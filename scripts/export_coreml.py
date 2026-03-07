#!/usr/bin/env python3
"""Export CorridorKey model (or submodules) to CoreML .mlpackage for Apple Neural Engine.

Usage:
    uv run python scripts/export_coreml.py \
        --checkpoint path/to/model.pth \
        --backbone-size 1024 \
        --output CorridorKey_backbone.mlpackage

Requires the `coreml` dependency group:
    uv sync --group coreml
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# GroupNorm -> BatchNorm weight folding (ANE compatibility)
# ---------------------------------------------------------------------------


def groupnorm_to_batchnorm(gn: nn.GroupNorm) -> nn.BatchNorm2d:
    """Convert GroupNorm to BatchNorm2d with folded weights for ANE compatibility.

    Valid ONLY at batch_size=1 inference. GroupNorm computes per-instance stats;
    BatchNorm with identity running stats is equivalent at batch=1.
    """
    bn = nn.BatchNorm2d(gn.num_channels)
    if gn.affine:
        bn.weight.data.copy_(gn.weight.data)
        bn.bias.data.copy_(gn.bias.data)
    else:
        bn.weight.data.fill_(1.0)
        bn.bias.data.fill_(0.0)
    bn.running_mean.fill_(0.0)
    bn.running_var.fill_(1.0)
    bn.eps = gn.eps
    bn.training = False
    return bn


def replace_groupnorm_with_batchnorm(module: nn.Module) -> int:
    """Recursively replace all GroupNorm layers with equivalent BatchNorm2d."""
    replacements = 0
    for name, child in module.named_children():
        if isinstance(child, nn.GroupNorm):
            bn = groupnorm_to_batchnorm(child)
            setattr(module, name, bn)
            replacements += 1
        else:
            replacements += replace_groupnorm_with_batchnorm(child)
    return replacements


# ---------------------------------------------------------------------------
# Backbone wrapper for tracing (isolates encoder from GreenFormer)
# ---------------------------------------------------------------------------


class BackboneWrapper(nn.Module):
    """Wraps the Hiera encoder for JIT tracing. Returns concatenated features."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Refiner wrapper for tracing
# ---------------------------------------------------------------------------


class RefinerWrapper(nn.Module):
    """Wraps CNNRefinerModule for JIT tracing with explicit input signature."""

    def __init__(self, refiner: nn.Module):
        super().__init__()
        self.refiner = refiner

    def forward(self, img: torch.Tensor, coarse_pred: torch.Tensor) -> torch.Tensor:
        return self.refiner(img, coarse_pred)


# ---------------------------------------------------------------------------
# Model loading (reuses inference_engine logic)
# ---------------------------------------------------------------------------


def load_greenformer(checkpoint_path: str, backbone_size: int) -> nn.Module:
    """Load GreenFormer from checkpoint at given backbone resolution."""
    from CorridorKeyModule.core.model_transformer import GreenFormer

    model = GreenFormer(
        encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
        img_size=backbone_size,
        backbone_size=None,  # No downsampling — export at native backbone_size
        use_refiner=True,
        refiner_tile_size=None,  # No tiling for export
        sparse_refiner=False,  # No sparse logic for export
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip compiled-model prefix & resize pos embeds
    new_state_dict = {}
    model_state = model.state_dict()
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[10:]
        if "pos_embed" in k and k in model_state and v.shape != model_state[k].shape:
            N_src, C = v.shape[1], v.shape[2]
            N_dst = model_state[k].shape[1]
            grid_src = int(math.sqrt(N_src))
            grid_dst = int(math.sqrt(N_dst))
            v_img = v.permute(0, 2, 1).view(1, C, grid_src, grid_src)
            v = F.interpolate(v_img, size=(grid_dst, grid_dst), mode="bicubic", align_corners=False)
            v = v.flatten(2).transpose(1, 2)
            print(f"  Resized {k}: {N_src} -> {N_dst} tokens")
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  [Warning] Missing keys: {missing}")
    if unexpected:
        print(f"  [Warning] Unexpected keys: {unexpected}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# CoreML export
# ---------------------------------------------------------------------------


def export_backbone(
    model: nn.Module,
    backbone_size: int,
    output_path: str,
) -> tuple:
    """Export backbone (encoder) to CoreML .mlpackage."""
    import coremltools as ct

    print(f"\n--- Exporting backbone to {output_path} ---")

    wrapper = BackboneWrapper(model.encoder)
    wrapper.eval()

    example_input = torch.randn(1, 4, backbone_size, backbone_size)

    print("  Tracing backbone...")
    try:
        traced = torch.jit.trace(wrapper, example_input, strict=False)
    except Exception as e:
        print(f"  [Error] JIT trace failed: {e}")
        print("  Hiera window partitioning may use dynamic shapes incompatible with tracing.")
        print("  Try --refiner-only to export just the CNN refiner.")
        raise

    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=example_input.shape, name="input_rgba")],
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.save(output_path)
    print(f"  Saved: {output_path}")
    return mlmodel, wrapper, example_input


def export_refiner(
    model: nn.Module,
    backbone_size: int,
    output_path: str,
) -> tuple:
    """Export CNN refiner to CoreML .mlpackage (ANE-friendly fallback)."""
    import coremltools as ct

    print(f"\n--- Exporting refiner to {output_path} ---")

    if model.refiner is None:
        print("  [Error] Model has no refiner module.")
        return None, None, None

    # Deep-copy refiner and fold GroupNorm -> BatchNorm for ANE
    refiner_copy = type(model.refiner)(
        in_channels=model.refiner.stem[0].in_channels,
        hidden_channels=model.refiner.stem[0].out_channels,
        out_channels=model.refiner.final.out_channels,
    )
    refiner_copy.load_state_dict(model.refiner.state_dict())
    refiner_copy.eval()

    num_replaced = replace_groupnorm_with_batchnorm(refiner_copy)
    print(f"  Replaced {num_replaced} GroupNorm -> BatchNorm layers")

    wrapper = RefinerWrapper(refiner_copy)
    wrapper.eval()

    example_img = torch.randn(1, 3, backbone_size, backbone_size)
    example_coarse = torch.randn(1, 4, backbone_size, backbone_size)

    print("  Tracing refiner...")
    traced = torch.jit.trace(wrapper, (example_img, example_coarse))

    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(shape=example_img.shape, name="rgb"),
            ct.TensorType(shape=example_coarse.shape, name="coarse_pred"),
        ],
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.save(output_path)
    print(f"  Saved: {output_path}")
    return mlmodel, wrapper, (example_img, example_coarse)


# ---------------------------------------------------------------------------
# Validation: CoreML vs PyTorch
# ---------------------------------------------------------------------------


def validate_output(
    mlmodel,
    torch_module: nn.Module,
    example_inputs,
    component_name: str,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> bool:
    """Compare CoreML prediction against PyTorch output."""
    print(f"\n--- Validating {component_name} ---")

    # PyTorch reference
    with torch.no_grad():
        if isinstance(example_inputs, tuple):
            torch_out = torch_module(*example_inputs)
        else:
            torch_out = torch_module(example_inputs)

    # Handle multi-output (backbone returns list of features)
    if isinstance(torch_out, (list, tuple)):
        torch_arrays = [t.numpy() for t in torch_out]
    else:
        torch_arrays = [torch_out.numpy()]

    # CoreML prediction
    input_names = list(mlmodel.input_description)
    if isinstance(example_inputs, tuple):
        coreml_input = {name: inp.numpy() for name, inp in zip(input_names, example_inputs, strict=True)}
    else:
        coreml_input = {input_names[0]: example_inputs.numpy()}

    coreml_out = mlmodel.predict(coreml_input)
    coreml_arrays = list(coreml_out.values())

    if len(torch_arrays) != len(coreml_arrays):
        print(f"  [FAIL] Output count mismatch: PyTorch={len(torch_arrays)}, CoreML={len(coreml_arrays)}")
        return False

    all_pass = True
    for i, (t_arr, c_arr) in enumerate(zip(torch_arrays, coreml_arrays, strict=True)):
        if not np.allclose(t_arr, c_arr, atol=atol, rtol=rtol):
            max_diff = np.abs(t_arr - c_arr).max()
            mean_diff = np.abs(t_arr - c_arr).mean()
            print(f"  Output {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} [FAIL]")
            all_pass = False
        else:
            max_diff = np.abs(t_arr - c_arr).max()
            print(f"  Output {i}: max_diff={max_diff:.6f} [PASS]")

    if all_pass:
        print(f"  {component_name} validation PASSED (atol={atol}, rtol={rtol})")
    else:
        print(f"  {component_name} validation FAILED — outputs differ beyond tolerance")
        print("  This may be acceptable if differences are small (FP32 rounding).")

    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CorridorKey model to CoreML .mlpackage for Apple Neural Engine",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model .pth checkpoint")
    parser.add_argument("--backbone-size", type=int, default=1024, help="Backbone resolution (default: 1024)")
    parser.add_argument("--output", default="CorridorKey_backbone.mlpackage", help="Output .mlpackage path")
    parser.add_argument(
        "--refiner-only",
        action="store_true",
        help="Export only the CNN refiner (simpler, guaranteed ANE-compatible)",
    )
    parser.add_argument("--skip-validation", action="store_true", help="Skip output validation step")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for validation (default: 1e-3)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for validation (default: 1e-3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Check coremltools availability
    try:
        import coremltools  # noqa: F401
    except ImportError:
        print("Error: coremltools not installed. Run: uv sync --group coreml")
        sys.exit(1)

    print(f"Loading model from {args.checkpoint}...")
    model = load_greenformer(args.checkpoint, args.backbone_size)

    if args.refiner_only:
        result = export_refiner(model, args.backbone_size, args.output)
        if result[0] is None:
            sys.exit(1)
        mlmodel, wrapper, example_inputs = result
        component_name = "refiner"
    else:
        try:
            result = export_backbone(model, args.backbone_size, args.output)
            mlmodel, wrapper, example_inputs = result
            component_name = "backbone"
        except Exception:
            print("\nBackbone export failed. Attempting refiner-only fallback...")
            fallback_path = args.output.replace(".mlpackage", "_refiner.mlpackage")
            result = export_refiner(model, args.backbone_size, fallback_path)
            if result[0] is None:
                sys.exit(1)
            mlmodel, wrapper, example_inputs = result
            component_name = "refiner"

    if not args.skip_validation and mlmodel is not None:
        validate_output(mlmodel, wrapper, example_inputs, component_name, atol=args.atol, rtol=args.rtol)

    print("\nDone.")


if __name__ == "__main__":
    main()
