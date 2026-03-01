#!/usr/bin/env python3
"""Full-pipeline inference benchmark for CorridorKey GreenFormer.

Measures per-stage latency at production resolution (2048x2048):
  1. Input generation (synthetic frame + mask)
  2. Preprocessing (cv2.resize + normalize + concat + tensor)
  3. CPU→MPS transfer (.to(device))
  4. Model forward (autocast + model(inp_t))
  5. MPS→CPU transfer (.cpu().numpy())
  6. Postprocessing (resize + clean_matte + despill + color + premul + checkerboard)
  7. Total pipeline (end-to-end process_frame())

Usage:
    uv run python scripts/benchmark_inference.py --device mps --random-weights
    uv run python scripts/benchmark_inference.py --device mps --checkpoint path/to/ckpt.pt
    uv run python scripts/benchmark_inference.py --device mps --random-weights --output-json results.json
    uv run python scripts/benchmark_inference.py --device mps --random-weights --skip-postprocess
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve project root so we can import CorridorKeyModule
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from CorridorKeyModule.core import color_utils as cu  # noqa: E402
from CorridorKeyModule.core.model_transformer import GreenFormer  # noqa: E402
from device_utils import clear_device_cache, memory_snapshot, resolve_device  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_IMG_SIZE = 2048
DEFAULT_ITERATIONS = 10
DEFAULT_WARMUP = 3
ENCODER_NAME = "hiera_base_plus_224.mae_in1k_ft_in1k"
INPUT_CHANNELS = 4  # RGB + coarse alpha
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync(device: torch.device) -> None:
    """Synchronize device before timing."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _make_synthetic_alpha(h: int, w: int, alpha_type: str) -> np.ndarray:
    """Generate synthetic alpha mask [H, W, 1] float32 in [0, 1].

    Types:
        gradient — radial gradient (realistic single-blob matte)
        flat     — all zeros
        binary   — step function (circle in center)
    """
    if alpha_type == "flat":
        return np.zeros((h, w, 1), dtype=np.float32)

    # Create coordinate grid normalized to [-1, 1]
    y_coords = np.linspace(-1, 1, h, dtype=np.float32)
    x_coords = np.linspace(-1, 1, w, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords)
    dist = np.sqrt(xx ** 2 + yy ** 2)

    if alpha_type == "binary":
        mask = (dist < 0.6).astype(np.float32)
    else:  # gradient (default)
        mask = np.clip(1.0 - dist, 0.0, 1.0)

    return mask[:, :, np.newaxis]


def _make_synthetic_image(h: int, w: int) -> np.ndarray:
    """Generate synthetic sRGB image [H, W, 3] float32 in [0, 1]."""
    rng = np.random.default_rng(42)
    return rng.random((h, w, 3), dtype=np.float32)


def _build_model(
    device: torch.device,
    img_size: int,
    checkpoint_path: str | None,
) -> GreenFormer:
    """Instantiate GreenFormer, optionally load checkpoint."""
    model = GreenFormer(
        encoder_name=ENCODER_NAME,
        in_channels=INPUT_CHANNELS,
        img_size=img_size,
        use_refiner=True,
    )
    model = model.to(device)
    model.eval()

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Strip _orig_mod. prefix from compiled checkpoints
        cleaned = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)
        print(f"  Loaded checkpoint: {ckpt_path}")

    return model


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_pipeline_benchmark(
    device_str: str,
    img_size: int,
    num_warmup: int,
    num_iters: int,
    checkpoint_path: str | None,
    alpha_type: str,
    skip_postprocess: bool,
    auto_despeckle: bool,
) -> dict:
    """Run full-pipeline benchmark, return results dict."""
    device = torch.device(device_str)
    label = f"despeckle={'on' if auto_despeckle else 'off'}"

    print(f"\n{'=' * 70}")
    print(f"Full-Pipeline Benchmark  device={device_str}  img_size={img_size}  "
          f"alpha={alpha_type}  {label}")
    print(f"{'=' * 70}")

    # --- Build model ---
    print("Loading model...")
    _sync(device)
    t0 = time.perf_counter()
    model = _build_model(device, img_size, checkpoint_path)
    _sync(device)
    model_load_sec = time.perf_counter() - t0
    mem_after_load = memory_snapshot(device)
    print(f"  Model load: {model_load_sec:.2f}s  |  "
          f"Memory: {mem_after_load['current_alloc_mb'] or 'N/A'} MB")

    # --- Generate synthetic inputs (once, reuse across iters) ---
    image = _make_synthetic_image(img_size, img_size)
    mask = _make_synthetic_alpha(img_size, img_size, alpha_type)

    # --- Warmup ---
    print(f"Warmup ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        # Preprocess
        img_resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]
        img_norm = (img_resized - IMAGENET_MEAN) / IMAGENET_STD
        inp_np = np.concatenate([img_norm, mask_resized], axis=-1).transpose((2, 0, 1))
        inp_t = torch.from_numpy(inp_np).float().unsqueeze(0).to(device, non_blocking=True)
        # Forward
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
            out = model(inp_t)
        if not skip_postprocess:
            alpha_np = out["alpha"][0].permute(1, 2, 0).float().cpu().numpy()
            fg_np = out["fg"][0].permute(1, 2, 0).float().cpu().numpy()
            res_alpha = cv2.resize(alpha_np, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
            if res_alpha.ndim == 2:
                res_alpha = res_alpha[:, :, np.newaxis]
            if auto_despeckle:
                cu.clean_matte(res_alpha, area_threshold=400, dilation=25, blur_size=5)
        _sync(device)
    clear_device_cache(device)

    mem_after_warmup = memory_snapshot(device)
    print(f"  Memory after warmup: {mem_after_warmup['current_alloc_mb'] or 'N/A'} MB")

    # --- Measured iterations ---
    stage_names = [
        "input_gen", "preprocess", "transfer_to_device",
        "model_forward", "transfer_to_cpu", "postprocess", "total",
    ]
    stage_latencies: dict[str, list[float]] = {name: [] for name in stage_names}

    print(f"Measuring ({num_iters} iterations)...")
    for i in range(num_iters):
        # Total timer — sync before start, sync after all stages
        _sync(device)
        t_total_start = time.perf_counter()

        # Stage 1: Input gen (reuse pre-generated — simulates frame buffer)
        t0 = time.perf_counter()
        t_input_gen = time.perf_counter() - t0

        # Stage 2: Preprocess (pure CPU)
        t0 = time.perf_counter()
        img_resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]
        img_norm = (img_resized - IMAGENET_MEAN) / IMAGENET_STD
        inp_np = np.concatenate([img_norm, mask_resized], axis=-1).transpose((2, 0, 1))
        t_preprocess = time.perf_counter() - t0

        # Stage 3: CPU→device transfer (non_blocking=False for per-stage accuracy)
        _sync(device)
        t0 = time.perf_counter()
        inp_t = torch.from_numpy(inp_np).float().unsqueeze(0).to(device, non_blocking=False)
        _sync(device)
        t_transfer_to = time.perf_counter() - t0

        # Stage 4: Model forward
        _sync(device)
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
            out = model(inp_t)
        _sync(device)
        t_forward = time.perf_counter() - t0

        if skip_postprocess:
            t_transfer_from = 0.0
            t_postprocess = 0.0
        else:
            # Stage 5: MPS→CPU transfer (.cpu().numpy() is implicit sync)
            _sync(device)
            t0 = time.perf_counter()
            alpha_np = out["alpha"][0].permute(1, 2, 0).float().cpu().numpy()
            fg_np = out["fg"][0].permute(1, 2, 0).float().cpu().numpy()
            t_transfer_from = time.perf_counter() - t0

            # Stage 6: Postprocess (pure CPU)
            t0 = time.perf_counter()
            res_alpha = cv2.resize(alpha_np, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
            res_fg = cv2.resize(fg_np, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
            if res_alpha.ndim == 2:
                res_alpha = res_alpha[:, :, np.newaxis]
            if auto_despeckle:
                processed_alpha = cu.clean_matte(res_alpha, area_threshold=400, dilation=25, blur_size=5)
            else:
                processed_alpha = res_alpha
            fg_despilled = cu.despill(res_fg, green_limit_mode="average", strength=1.0)
            fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
            fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)
            _ = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)
            bg_srgb = cu.create_checkerboard(img_size, img_size, checker_size=128, color1=0.15, color2=0.55)
            bg_lin = cu.srgb_to_linear(bg_srgb)
            comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
            _ = cu.linear_to_srgb(comp_lin)
            t_postprocess = time.perf_counter() - t0

        _sync(device)
        t_total = time.perf_counter() - t_total_start

        stage_latencies["input_gen"].append(t_input_gen)
        stage_latencies["preprocess"].append(t_preprocess)
        stage_latencies["transfer_to_device"].append(t_transfer_to)
        stage_latencies["model_forward"].append(t_forward)
        stage_latencies["transfer_to_cpu"].append(t_transfer_from)
        stage_latencies["postprocess"].append(t_postprocess)
        stage_latencies["total"].append(t_total)

    # --- Compute stats ---
    clear_device_cache(device)
    peak_snap = memory_snapshot(device)

    stage_stats: dict[str, dict] = {}
    for name in stage_names:
        lats = stage_latencies[name]
        if not lats or all(v == 0 for v in lats):
            stage_stats[name] = {"median_sec": 0, "mean_sec": 0, "min_sec": 0, "max_sec": 0}
            continue
        stage_stats[name] = {
            "median_sec": round(statistics.median(lats), 6),
            "mean_sec": round(statistics.mean(lats), 6),
            "min_sec": round(min(lats), 6),
            "max_sec": round(max(lats), 6),
        }

    total_median = stage_stats["total"]["median_sec"]
    throughput = round(1.0 / total_median, 2) if total_median > 0 else 0.0

    # --- Print summary ---
    print(f"\n{'─' * 70}")
    print(f"  Per-Stage Latency Breakdown (median of {num_iters} iters)")
    print(f"{'─' * 70}")
    for name in stage_names:
        s = stage_stats[name]
        med = s["median_sec"]
        pct = (med / total_median * 100) if total_median > 0 else 0
        bar = "█" * int(pct / 2)
        if name == "total":
            print(f"{'─' * 70}")
        print(f"  {name:<24s} {med:>8.4f}s  ({pct:5.1f}%)  {bar}")

    print(f"\n  Throughput: {throughput:.2f} FPS")
    print(f"  Memory after load:   {mem_after_load['current_alloc_mb'] or 'N/A'} MB")
    print(f"  Memory after warmup: {mem_after_warmup['current_alloc_mb'] or 'N/A'} MB")
    print(f"  Peak driver memory:  {peak_snap['driver_alloc_mb'] or 'N/A'} MB")
    print(f"  Recommended max:     {peak_snap['recommended_max_mb'] or 'N/A'} MB")

    results = {
        "device": device_str,
        "img_size": img_size,
        "alpha_type": alpha_type,
        "auto_despeckle": auto_despeckle,
        "skip_postprocess": skip_postprocess,
        "num_warmup": num_warmup,
        "num_iters": num_iters,
        "model_load_sec": round(model_load_sec, 3),
        "checkpoint": checkpoint_path or "random_weights",
        "stages": stage_stats,
        "throughput_fps": throughput,
        "memory": {
            "after_load": {k: round(v, 1) if v else None for k, v in mem_after_load.items()},
            "after_warmup": {k: round(v, 1) if v else None for k, v in mem_after_warmup.items()},
            "peak": {k: round(v, 1) if v else None for k, v in peak_snap.items()},
        },
    }
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full-pipeline inference benchmark for CorridorKey GreenFormer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--device", default="auto", choices=["mps", "cpu", "cuda", "auto"],
                    help="Device to benchmark (default: auto)")
    p.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE,
                    help=f"Input resolution (default: {DEFAULT_IMG_SIZE})")
    p.add_argument("--num-warmup", type=int, default=DEFAULT_WARMUP,
                    help=f"Warmup iterations (default: {DEFAULT_WARMUP})")
    p.add_argument("--num-iters", type=int, default=DEFAULT_ITERATIONS,
                    help=f"Measured iterations (default: {DEFAULT_ITERATIONS})")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to model checkpoint (.pt)")
    p.add_argument("--random-weights", action="store_true",
                    help="Use random weights (skip checkpoint, valid for timing only)")
    p.add_argument("--alpha-type", default="gradient", choices=["gradient", "flat", "binary"],
                    help="Synthetic alpha topology (default: gradient)")
    p.add_argument("--skip-postprocess", action="store_true",
                    help="Skip postprocessing stages (isolate model-forward)")
    p.add_argument("--output-json", type=str, default=None,
                    help="Path to write JSON results")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Validate checkpoint args
    if not args.random_weights and args.checkpoint is None:
        print("ERROR: Specify --checkpoint <path> or --random-weights")
        sys.exit(1)

    checkpoint_path = None if args.random_weights else args.checkpoint

    # Resolve device
    device_str = resolve_device(args.device)
    print(f"Device: {device_str}")

    all_results = []

    try:
        # Run with auto_despeckle=True
        if not args.skip_postprocess:
            results_despeckle_on = run_pipeline_benchmark(
                device_str=device_str,
                img_size=args.img_size,
                num_warmup=args.num_warmup,
                num_iters=args.num_iters,
                checkpoint_path=checkpoint_path,
                alpha_type=args.alpha_type,
                skip_postprocess=False,
                auto_despeckle=True,
            )
            all_results.append(results_despeckle_on)

            # Run with auto_despeckle=False
            results_despeckle_off = run_pipeline_benchmark(
                device_str=device_str,
                img_size=args.img_size,
                num_warmup=args.num_warmup,
                num_iters=args.num_iters,
                checkpoint_path=checkpoint_path,
                alpha_type=args.alpha_type,
                skip_postprocess=False,
                auto_despeckle=False,
            )
            all_results.append(results_despeckle_off)
        else:
            results_skip = run_pipeline_benchmark(
                device_str=device_str,
                img_size=args.img_size,
                num_warmup=args.num_warmup,
                num_iters=args.num_iters,
                checkpoint_path=checkpoint_path,
                alpha_type=args.alpha_type,
                skip_postprocess=True,
                auto_despeckle=False,
            )
            all_results.append(results_skip)

    except RuntimeError as e:
        err_msg = str(e).lower()
        if "out of memory" in err_msg or "mps" in err_msg:
            print(f"\nOOM at {args.img_size}x{args.img_size}: {e}")
            print("Suggestion: try --img-size 1024")
            oom_result = {
                "oom": True,
                "img_size": args.img_size,
                "device": device_str,
                "error": str(e),
            }
            if args.output_json:
                Path(args.output_json).write_text(json.dumps(oom_result, indent=2))
                print(f"OOM result written to {args.output_json}")
            sys.exit(0)
        raise

    # Write JSON output
    if args.output_json:
        output = all_results if len(all_results) > 1 else all_results[0]
        Path(args.output_json).write_text(json.dumps(output, indent=2))
        print(f"\nResults written to {args.output_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
