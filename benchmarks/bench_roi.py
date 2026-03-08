"""Benchmark: 3-way ROI method comparison (none / yolo / alpha_hint).

Measures per-frame timing, peak memory, and pixel-level quality (MAE, % > threshold)
against the full-frame baseline.

Usage:
  uv run python benchmarks/bench_roi.py --clip <video> --alpha <alpha_video>
  uv run python benchmarks/bench_roi.py --clip <video> --alpha <alpha_video> --methods none alpha_hint
  uv run python benchmarks/bench_roi.py --clip <video> --alpha <alpha_video> --max-frames 20
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from bench_phase import (  # noqa: E402
    create_engine,
    get_device,
    load_mask_frames,
    load_video_frames,
    measure_memory_after,
    measure_memory_before,
    print_memory_summary,
    print_timing_summary,
)

MAX_BENCHMARK_FRAMES = 10
QUALITY_THRESHOLD = 1e-2


# ── Device sync helpers ──────────────────────────────────────────────────────


def sync_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# ── Benchmark core ───────────────────────────────────────────────────────────


def run_method(
    engine,
    roi_manager,
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    device: str,
    method: str,
) -> tuple[list[dict], dict, list[float]]:
    """Run inference for one method.

    Returns (outputs, mem_info, frame_times) — same shape as bench_phase.run_benchmark.
    """
    if roi_manager is not None:
        roi_manager.reset()

    mem_before = measure_memory_before(device)
    outputs = []
    frame_times = []

    print(f"  Running benchmark ({len(frames)} frames, first is warmup)...")

    for i, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
        sync_device(device)
        t0 = time.perf_counter()

        if method == "none":
            result = engine.process_frame(frame, mask)
        else:
            result = roi_manager.process_with_roi(engine, frame, mask)

        sync_device(device)
        t1 = time.perf_counter()

        frame_times.append(t1 - t0)
        outputs.append(result)
        print(f"  Frame {i + 1}/{len(frames)}: {t1 - t0:.3f}s", end="\r")

    print()

    mem_info = measure_memory_after(device)
    mem_info["before_bytes"] = mem_before
    mem_info["delta_bytes"] = mem_info["allocated_bytes"] - mem_before

    return outputs, mem_info, frame_times


# ── Quality metrics ──────────────────────────────────────────────────────────


def compute_quality_metrics(
    baseline_outputs: list[dict],
    competitor_outputs: list[dict],
) -> dict[str, dict[str, float]]:
    """Compute MAE and % pixels > threshold per output channel."""
    channels = ["alpha", "fg"]
    start_idx = 1 if len(baseline_outputs) > 1 else 0
    metrics = {}

    for key in channels:
        maes = []
        pcts = []
        for i in range(start_idx, min(len(baseline_outputs), len(competitor_outputs))):
            base = baseline_outputs[i].get(key)
            comp = competitor_outputs[i].get(key)
            if base is None or comp is None:
                continue
            if base.shape != comp.shape:
                continue
            diff = np.abs(base.astype(np.float64) - comp.astype(np.float64))
            maes.append(float(diff.mean()))
            pcts.append(float((diff > QUALITY_THRESHOLD).mean() * 100))

        if maes:
            metrics[key] = {
                "mae": statistics.mean(maes),
                "pct_above": statistics.mean(pcts),
            }

    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Benchmark ROI methods: none / yolo / alpha_hint")
    parser.add_argument("--clip", required=True, help="Path to input RGB video")
    parser.add_argument("--alpha", required=True, help="Path to alpha hint video")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--device", default=None, help="Device override (cuda/mps/cpu)")
    parser.add_argument("--max-frames", type=int, default=MAX_BENCHMARK_FRAMES, help="Max frames (default 10)")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["none", "yolo", "alpha_hint"],
        default=["none", "yolo", "alpha_hint"],
        help="Which methods to benchmark (default: all three)",
    )
    parser.add_argument(
        "--shared-engine",
        action="store_true",
        help="Reuse one engine across methods (diagnostic: isolates MPS recompilation cost)",
    )

    args = parser.parse_args()

    device = args.device or get_device()

    # Load frames
    frames = load_video_frames(args.clip, max_frames=args.max_frames)
    masks = load_mask_frames(args.alpha, max_frames=args.max_frames)
    num_frames = min(len(frames), len(masks))
    frames = frames[:num_frames]
    masks = masks[:num_frames]

    frame_h, frame_w = frames[0].shape[:2]

    # Ensure "none" always runs first as baseline
    methods = list(args.methods)
    if "none" in methods:
        methods.remove("none")
        methods.insert(0, "none")
    else:
        methods.insert(0, "none")

    print(f"\n{'=' * 70}")
    print(f"ROI Benchmark — {num_frames} frames, {frame_w}x{frame_h}, device={device}")
    if args.shared_engine:
        print("  (shared engine mode — single engine reused across methods)")
    print(f"{'=' * 70}\n")

    from CorridorKeyModule.roi_manager import ROIManager

    results: dict[str, dict] = {}
    shared_engine = create_engine(device, args.checkpoint) if args.shared_engine else None

    for method in methods:
        print(f"\n{'=' * 70}")
        print(f"METHOD: {method}")
        print("=" * 70)

        engine = shared_engine if shared_engine else create_engine(device, args.checkpoint)

        roi_manager = None
        if method != "none":
            roi_manager = ROIManager(roi_method=method)

        outputs, mem_info, frame_times = run_method(engine, roi_manager, frames, masks, device, method)

        print("\n--- Timing ---")
        print_timing_summary(frame_times)

        print("\n--- Memory ---")
        print_memory_summary(mem_info)

        results[method] = {
            "outputs": outputs,
            "times": frame_times,
            "mem_info": mem_info,
        }

        # Free engine to release VRAM before next method (skip if shared)
        if not shared_engine:
            del engine
        del roi_manager
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # ── Summary table ────────────────────────────────────────────────────────

    baseline_times = results["none"]["times"]
    baseline_median = statistics.median(baseline_times[1:]) if len(baseline_times) > 1 else baseline_times[0]

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Method':<20} {'Median (s)':>12} {'Peak Mem (GB)':>14} {'Speedup':>10}")
    print("-" * 58)

    for method in methods:
        times = results[method]["times"]
        steady = times[1:] if len(times) > 1 else times
        med = statistics.median(steady) if steady else 0
        peak_gb = results[method]["mem_info"]["peak_bytes"] / 1e9
        if method == "none":
            speedup_str = "(baseline)"
        elif baseline_median > 0:
            speedup_str = f"{baseline_median / med:.2f}x"
        else:
            speedup_str = "—"
        print(f"{method:<20} {med:>12.3f} {peak_gb:>14.2f} {speedup_str:>10}")

    # Quality comparison
    baseline_outputs = results["none"]["outputs"]
    has_quality = False
    for method in methods:
        if method == "none":
            continue
        metrics = compute_quality_metrics(baseline_outputs, results[method]["outputs"])
        if metrics:
            if not has_quality:
                print(f"\n{'=' * 70}")
                print("QUALITY vs BASELINE (none)")
                has_quality = True
            print(f"  {method}:")
            for channel, m in metrics.items():
                print(f"    {channel:8s}: MAE={m['mae']:.4f}, >{QUALITY_THRESHOLD}: {m['pct_above']:.1f}%")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
