#!/usr/bin/env python3
"""Phase 8: Detect MPS unsupported ops that silently fall back to CPU.

Runs inference with PYTORCH_ENABLE_MPS_FALLBACK=1 and captures all fallback
warnings. Also profiles the forward pass to identify CPU-dispatched ops.

Usage:
    python scripts/detect_mps_fallback.py --img-size 1024
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# Force MPS fallback mode to expose unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from CorridorKeyModule.core.model_transformer import GreenFormer  # noqa: E402

ENCODER_NAME = "hiera_base_plus_224.mae_in1k_ft_in1k"
INPUT_CHANNELS = 4
DEFAULT_IMG_SIZE = 1024


def capture_fallback_warnings(model, inp, device):
    """Run forward pass and capture any MPS fallback warnings."""
    fallback_warnings = []

    # Capture warnings
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        # Also capture stderr for PyTorch MPS fallback messages
        import io
        old_stderr = sys.stderr
        captured_stderr = io.StringIO()
        sys.stderr = captured_stderr

        try:
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
                _ = model(inp)
            torch.mps.synchronize()
        finally:
            sys.stderr = old_stderr

        stderr_output = captured_stderr.getvalue()

    # Collect fallback warnings from warnings module
    for w in caught_warnings:
        msg = str(w.message)
        if "MPS" in msg or "fallback" in msg.lower() or "not supported" in msg.lower():
            fallback_warnings.append(msg)

    # Collect fallback messages from stderr
    for line in stderr_output.splitlines():
        if "MPS" in line or "fallback" in line.lower() or "not supported" in line.lower():
            fallback_warnings.append(f"[stderr] {line}")

    return fallback_warnings, stderr_output


def profile_ops(model, inp, device):
    """Profile forward pass to identify op dispatch locations."""
    from torch.profiler import profile, ProfilerActivity

    activities = [ProfilerActivity.CPU]

    # Run profiled forward pass
    with profile(activities=activities, record_shapes=True) as prof:
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
            _ = model(inp)
        torch.mps.synchronize()

    return prof


def check_known_ops(device):
    """Test known potentially-unsupported ops on MPS."""
    results = {}
    test_tensor = torch.randn(1, 3, 64, 64, device=device)

    ops_to_check = {
        "F.interpolate bilinear": lambda: torch.nn.functional.interpolate(
            test_tensor, size=(128, 128), mode="bilinear", align_corners=False
        ),
        "F.interpolate bicubic": lambda: torch.nn.functional.interpolate(
            test_tensor, size=(128, 128), mode="bicubic", align_corners=False
        ),
        "torch.sigmoid": lambda: torch.sigmoid(test_tensor),
        "torch.cat": lambda: torch.cat([test_tensor, test_tensor], dim=1),
        "GroupNorm": lambda: torch.nn.GroupNorm(1, 3).to(device)(test_tensor),
        "BatchNorm2d": lambda: torch.nn.BatchNorm2d(3).to(device)(test_tensor),
        "Conv2d 3x3": lambda: torch.nn.Conv2d(3, 3, 3, padding=1).to(device)(test_tensor),
        "Conv2d dilated": lambda: torch.nn.Conv2d(3, 3, 3, padding=2, dilation=2).to(device)(test_tensor),
        "max_pool2d 3x3": lambda: torch.nn.functional.max_pool2d(test_tensor, 3, stride=1, padding=1),
        "max_pool2d 7x7": lambda: torch.nn.functional.max_pool2d(test_tensor, 7, stride=1, padding=3),
        "scaled_dot_product_attention": lambda: torch.nn.functional.scaled_dot_product_attention(
            torch.randn(1, 4, 16, 64, device=device),
            torch.randn(1, 4, 16, 64, device=device),
            torch.randn(1, 4, 16, 64, device=device),
        ),
        "Linear": lambda: torch.nn.Linear(3, 3).to(device)(test_tensor.view(1, -1, 3)),
        "ReLU inplace": lambda: torch.nn.ReLU(inplace=True)(test_tensor.clone()),
    }

    for name, op_fn in ops_to_check.items():
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _ = op_fn()
                torch.mps.synchronize()

            fallback = any("MPS" in str(warning.message) or "fallback" in str(warning.message).lower()
                          for warning in w)
            results[name] = "FALLBACK" if fallback else "OK"
        except Exception as exc:
            results[name] = f"ERROR: {exc}"

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 8: MPS fallback op detection")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    args = parser.parse_args()

    device = torch.device("mps")
    img_size = args.img_size

    print(f"{'=' * 60}")
    print("Phase 8: MPS Unsupported Ops & Fallback Detection")
    print(f"Device: {device}  Size: {img_size}x{img_size}")
    print(f"PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'not set')}")
    print(f"{'=' * 60}")

    # 1. Check known ops
    print("\n--- Known Op Support ---")
    op_results = check_known_ops(device)
    for name, status in op_results.items():
        indicator = "  " if status == "OK" else "!!"
        print(f"  {indicator} {name}: {status}")

    fallback_ops = [name for name, status in op_results.items() if status == "FALLBACK"]
    error_ops = [name for name, status in op_results.items() if status.startswith("ERROR")]

    # 2. Build model and run inference with fallback capture
    print(f"\n--- Building model (img_size={img_size}) ---")
    model = GreenFormer(
        encoder_name=ENCODER_NAME,
        in_channels=INPUT_CHANNELS,
        img_size=img_size,
        use_refiner=True,
    )
    model = model.to(device)
    model.eval()
    torch.mps.synchronize()

    inp = torch.randn(1, INPUT_CHANNELS, img_size, img_size, device=device)

    # Warmup
    print("Warmup (1 iteration)...")
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
        _ = model(inp)
    torch.mps.synchronize()

    # Capture fallback warnings
    print("\n--- Capturing fallback warnings ---")
    fallback_warnings, stderr_out = capture_fallback_warnings(model, inp, device)

    if fallback_warnings:
        print(f"  Found {len(fallback_warnings)} fallback warning(s):")
        for w in fallback_warnings:
            print(f"    !! {w}")
    else:
        print("  No fallback warnings detected (all ops running on MPS)")

    if stderr_out.strip():
        print("\n--- Stderr output ---")
        for line in stderr_out.strip().splitlines()[:20]:
            print(f"  {line}")

    # 3. Profile ops
    print("\n--- Profiling forward pass ---")
    try:
        prof = profile_ops(model, inp, device)

        # Print top ops by CPU time
        print("\nTop 20 ops by CPU time (look for ops that should be on MPS):")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    except Exception as exc:
        print(f"  Profiling failed: {exc}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Known op fallbacks: {len(fallback_ops)} {fallback_ops if fallback_ops else ''}")
    print(f"  Known op errors: {len(error_ops)} {error_ops if error_ops else ''}")
    print(f"  Runtime fallback warnings: {len(fallback_warnings)}")

    if not fallback_ops and not fallback_warnings:
        print("\n  All ops running natively on MPS. No substitutions needed.")
    else:
        print("\n  ACTION REQUIRED: Review fallback ops for hot-path impact.")


if __name__ == "__main__":
    main()
