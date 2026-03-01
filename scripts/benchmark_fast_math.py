#!/usr/bin/env python3
"""MPS Fast Math parity and speed benchmark for CorridorKey GreenFormer.

PYTORCH_MPS_FAST_MATH=1 must be set BEFORE import torch (read at MPS init).
This script orchestrates two subprocess runs — one control, one fast math —
then compares raw model outputs and latency.

Usage:
    uv run python scripts/benchmark_fast_math.py --random-weights
    uv run python scripts/benchmark_fast_math.py --checkpoint path/to/ckpt.pt
    uv run python scripts/benchmark_fast_math.py --random-weights --img-size 1024 --num-iters 5
"""

from __future__ import annotations

# ── WORKER MODE ──────────────────────────────────────────────────────────
# When invoked with --_worker, this block runs BEFORE any torch import.
# The parent process sets PYTORCH_MPS_FAST_MATH via env before spawning.

import argparse
import sys

def _parse_worker_args() -> argparse.Namespace | None:
    """Check if we're in worker mode (--_worker flag present)."""
    if "--_worker" not in sys.argv:
        return None
    p = argparse.ArgumentParser()
    p.add_argument("--_worker", action="store_true")
    p.add_argument("--img-size", type=int, required=True)
    p.add_argument("--num-warmup", type=int, default=3)
    p.add_argument("--num-iters", type=int, default=10)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--random-weights", action="store_true")
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


_worker_args = _parse_worker_args()

if _worker_args is not None:
    # Worker path — torch not yet imported, env var already set by parent
    import json
    import os
    import statistics
    import time
    from pathlib import Path

    import numpy as np
    import torch

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    from CorridorKeyModule.core.model_transformer import GreenFormer  # noqa: E402
    from device_utils import clear_device_cache, memory_snapshot  # noqa: E402

    ENCODER_NAME = "hiera_base_plus_224.mae_in1k_ft_in1k"
    INPUT_CHANNELS = 4
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def _sync():
        torch.mps.synchronize()

    def _worker_main(args: argparse.Namespace) -> None:
        fast_math = os.environ.get("PYTORCH_MPS_FAST_MATH", "0")
        label = f"fast_math={fast_math}"
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("mps")
        img_size = args.img_size

        # Build model
        model = GreenFormer(
            encoder_name=ENCODER_NAME,
            in_channels=INPUT_CHANNELS,
            img_size=img_size,
            use_refiner=True,
        )
        model = model.to(device)
        model.eval()

        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device)
            state_dict = ckpt.get("state_dict", ckpt)
            cleaned = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned, strict=False)

        # Create deterministic input (same seed for both runs)
        rng = np.random.default_rng(42)
        image = rng.random((img_size, img_size, 3), dtype=np.float32)
        # Radial gradient mask
        y_coords = np.linspace(-1, 1, img_size, dtype=np.float32)
        x_coords = np.linspace(-1, 1, img_size, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)
        mask = np.clip(1.0 - np.sqrt(xx ** 2 + yy ** 2), 0, 1)[:, :, np.newaxis]

        # Preprocess
        img_norm = (image - IMAGENET_MEAN) / IMAGENET_STD
        inp_np = np.concatenate([img_norm, mask], axis=-1).transpose((2, 0, 1))
        inp_t = torch.from_numpy(inp_np).float().unsqueeze(0).to(device, non_blocking=True)
        _sync()

        # Warmup
        print(f"[{label}] Warmup ({args.num_warmup} iters)...")
        for _ in range(args.num_warmup):
            with torch.inference_mode(), torch.autocast(device_type="mps", dtype=torch.float16):
                _ = model(inp_t)
            _sync()
        clear_device_cache(device)

        # Measured runs — capture raw outputs from LAST iteration for parity
        latencies = []
        print(f"[{label}] Measuring ({args.num_iters} iters)...")
        for i in range(args.num_iters):
            _sync()
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type="mps", dtype=torch.float16):
                out = model(inp_t)
            _sync()
            latencies.append(time.perf_counter() - t0)

        # Save raw model outputs (pre-postprocess) from last iteration
        raw_alpha = out["alpha"][0].float().cpu().numpy()  # [1, H, W]
        raw_fg = out["fg"][0].float().cpu().numpy()        # [3, H, W]

        np.savez(
            output_dir / "raw_outputs.npz",
            alpha=raw_alpha,
            fg=raw_fg,
        )

        mem = memory_snapshot(device)

        results = {
            "fast_math": fast_math,
            "img_size": img_size,
            "num_iters": args.num_iters,
            "latency_median_sec": round(statistics.median(latencies), 6),
            "latency_mean_sec": round(statistics.mean(latencies), 6),
            "latency_min_sec": round(min(latencies), 6),
            "latency_max_sec": round(max(latencies), 6),
            "throughput_fps": round(1.0 / statistics.median(latencies), 2),
            "all_latencies_sec": [round(lat, 6) for lat in latencies],
            "memory": {k: round(v, 1) if v else None for k, v in mem.items()},
        }
        (output_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"[{label}] Median: {results['latency_median_sec']:.4f}s  "
              f"({results['throughput_fps']:.2f} FPS)")

    _worker_main(_worker_args)
    sys.exit(0)


# ── ORCHESTRATOR MODE ────────────────────────────────────────────────────

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _run_worker(
    fast_math: bool,
    img_size: int,
    num_warmup: int,
    num_iters: int,
    checkpoint: str | None,
    random_weights: bool,
    output_dir: Path,
) -> dict:
    """Spawn a worker subprocess with/without PYTORCH_MPS_FAST_MATH."""
    env = os.environ.copy()
    if fast_math:
        env["PYTORCH_MPS_FAST_MATH"] = "1"
    else:
        env.pop("PYTORCH_MPS_FAST_MATH", None)

    cmd = [
        sys.executable, __file__,
        "--_worker",
        "--img-size", str(img_size),
        "--num-warmup", str(num_warmup),
        "--num-iters", str(num_iters),
        "--output-dir", str(output_dir),
    ]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if random_weights:
        cmd += ["--random-weights"]

    label = "FAST_MATH=1" if fast_math else "FAST_MATH=0"
    print(f"\n{'─' * 60}")
    print(f"  Spawning worker: {label}")
    print(f"{'─' * 60}")

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Worker ({label}) failed with exit code {result.returncode}")

    results = json.loads((output_dir / "results.json").read_text())
    return results


def _compare_outputs(dir_control: Path, dir_fast: Path) -> dict:
    """Compare raw model outputs between control and fast math runs."""
    control = np.load(dir_control / "raw_outputs.npz")
    fast = np.load(dir_fast / "raw_outputs.npz")

    parity = {}
    for key in ["alpha", "fg"]:
        ctrl_arr = control[key]
        fast_arr = fast[key]
        diff = np.abs(ctrl_arr - fast_arr)
        max_abs_diff = float(diff.max())
        mean_abs_diff = float(diff.mean())
        # Fraction of pixels where diff > 1/255 (invisible in 8-bit)
        eightbit_threshold = 1.0 / 255.0
        pct_above_8bit = float((diff > eightbit_threshold).mean() * 100)

        parity[key] = {
            "max_abs_diff": round(max_abs_diff, 8),
            "mean_abs_diff": round(mean_abs_diff, 8),
            "pct_above_8bit_threshold": round(pct_above_8bit, 4),
            "shape": list(ctrl_arr.shape),
        }

    return parity


def main() -> None:
    p = argparse.ArgumentParser(
        description="MPS Fast Math parity + speed benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--img-size", type=int, default=1024,
                    help="Input resolution (default: 1024)")
    p.add_argument("--num-warmup", type=int, default=3,
                    help="Warmup iterations (default: 3)")
    p.add_argument("--num-iters", type=int, default=10,
                    help="Measured iterations (default: 10)")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to model checkpoint")
    p.add_argument("--random-weights", action="store_true",
                    help="Use random weights (valid for timing + relative parity)")
    p.add_argument("--output-json", type=str, default=None,
                    help="Path to write comparison JSON results")
    args = p.parse_args()

    if not args.random_weights and args.checkpoint is None:
        print("ERROR: Specify --checkpoint <path> or --random-weights")
        sys.exit(1)

    print("=" * 60)
    print("  MPS Fast Math Benchmark")
    print(f"  img_size={args.img_size}  iters={args.num_iters}")
    print("=" * 60)

    with tempfile.TemporaryDirectory(prefix="fast_math_bench_") as tmpdir:
        dir_control = Path(tmpdir) / "control"
        dir_fast = Path(tmpdir) / "fast_math"

        # Run control (no fast math)
        results_control = _run_worker(
            fast_math=False,
            img_size=args.img_size,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
            checkpoint=args.checkpoint,
            random_weights=args.random_weights,
            output_dir=dir_control,
        )

        # Run fast math
        results_fast = _run_worker(
            fast_math=True,
            img_size=args.img_size,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
            checkpoint=args.checkpoint,
            random_weights=args.random_weights,
            output_dir=dir_fast,
        )

        # Compare raw outputs
        parity = _compare_outputs(dir_control, dir_fast)

    # Compute deltas
    ctrl_med = results_control["latency_median_sec"]
    fast_med = results_fast["latency_median_sec"]
    speedup_pct = ((ctrl_med - fast_med) / ctrl_med * 100) if ctrl_med > 0 else 0

    # Determine recommendation
    alpha_max_diff = parity["alpha"]["max_abs_diff"]
    fg_max_diff = parity["fg"]["max_abs_diff"]
    max_diff = max(alpha_max_diff, fg_max_diff)
    eightbit_threshold = 1.0 / 255.0  # ~0.00392
    invisible_in_8bit = max_diff < eightbit_threshold

    if speedup_pct < 1.0:
        recommendation = "REJECT — negligible speed benefit, not worth precision risk"
    elif invisible_in_8bit:
        recommendation = "ADOPT — speed gain with invisible quality difference for 8-bit delivery"
    elif max_diff < 1e-3:
        recommendation = "ADOPT — speed gain with sub-1e-3 precision delta"
    elif max_diff < 0.004:
        recommendation = "ADOPT (gated) — gate behind --mps-fast-math flag, diffs invisible in 8-bit PNG"
    else:
        recommendation = "REJECT — quality degradation exceeds 8-bit invisibility threshold"

    # Print comparison
    print(f"\n{'=' * 60}")
    print("  COMPARISON: Control vs Fast Math")
    print(f"{'=' * 60}")
    print(f"\n  Latency (median):")
    print(f"    Control:    {ctrl_med:.4f}s  ({results_control['throughput_fps']:.2f} FPS)")
    print(f"    Fast Math:  {fast_med:.4f}s  ({results_fast['throughput_fps']:.2f} FPS)")
    print(f"    Delta:      {speedup_pct:+.2f}%")

    print(f"\n  Output Parity (raw model tensors, pre-postprocess):")
    for key in ["alpha", "fg"]:
        p_data = parity[key]
        print(f"    {key}:")
        print(f"      max_abs_diff:  {p_data['max_abs_diff']:.8f}")
        print(f"      mean_abs_diff: {p_data['mean_abs_diff']:.8f}")
        print(f"      pct > 1/255:   {p_data['pct_above_8bit_threshold']:.4f}%")

    print(f"\n  8-bit invisible (max_diff < 1/255): {'YES' if invisible_in_8bit else 'NO'}")
    print(f"\n  RECOMMENDATION: {recommendation}")

    # Compile full report
    report = {
        "control": results_control,
        "fast_math": results_fast,
        "parity": parity,
        "speedup_pct": round(speedup_pct, 2),
        "max_diff_any_channel": round(max_diff, 8),
        "invisible_in_8bit": invisible_in_8bit,
        "recommendation": recommendation,
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2))
        print(f"\nResults written to {args.output_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
