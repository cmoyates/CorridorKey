"""Benchmark: 3-way ROI method comparison (none / yolo / alpha_hint).

Measures per-frame timing, peak memory, and pixel-level quality (MAE, % > threshold)
against the full-frame baseline.

Usage:
  python benchmarks/bench_roi.py --clip <video> --alpha <alpha_video>
  python benchmarks/bench_roi.py --clip <video> --alpha <alpha_video> --methods none alpha_hint
  python benchmarks/bench_roi.py --clip <video> --alpha <alpha_video> --max-frames 20
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import cv2
import numpy as np
import torch

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ── Frame loading ─────────────────────────────────────────────────────────────

MAX_BENCHMARK_FRAMES = 10


def load_video_frames(video_path: str, max_frames: int = MAX_BENCHMARK_FRAMES) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb.astype(np.float32) / 255.0)
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from {video_path}")
    print(f"Loaded {len(frames)} frames from {os.path.basename(video_path)}")
    return frames


def load_mask_frames(video_path: str, max_frames: int = MAX_BENCHMARK_FRAMES) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open mask video: {video_path}")

    masks = []
    while len(masks) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        mask = frame[:, :, 2].astype(np.float32) / 255.0
        masks.append(mask)
    cap.release()

    if not masks:
        raise ValueError(f"No frames read from {video_path}")
    print(f"Loaded {len(masks)} mask frames from {os.path.basename(video_path)}")
    return masks


# ── Device helpers ─────────────────────────────────────────────────────────────


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def reset_memory(device: str) -> None:
    """Reset memory tracking between methods."""
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()


def measure_peak_memory(device: str) -> int:
    """Return peak memory in bytes."""
    if device == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()
    elif device == "mps":
        torch.mps.synchronize()
        return torch.mps.driver_allocated_memory()
    return 0


# ── Benchmark core ─────────────────────────────────────────────────────────────

QUALITY_THRESHOLD = 1e-2  # pixels with |diff| > this are "significant"


def run_method(
    engine,
    roi_manager,
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    device: str,
    method: str,
) -> tuple[list[dict], list[float], int]:
    """Run inference for one method.

    Returns (outputs, frame_times, peak_memory_bytes).
    """
    if roi_manager is not None:
        roi_manager.reset()

    reset_memory(device)

    outputs = []
    frame_times = []

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
        print(f"  [{method:11s}] Frame {i + 1}/{len(frames)}: {t1 - t0:.3f}s", end="\r")

    print()

    peak_mem = measure_peak_memory(device)
    return outputs, frame_times, peak_mem


def compute_quality_metrics(
    baseline_outputs: list[dict],
    competitor_outputs: list[dict],
) -> dict[str, dict[str, float]]:
    """Compute MAE and % pixels > threshold per output channel.

    Returns dict keyed by channel name, each with 'mae' and 'pct_above'.
    """
    channels = ["alpha", "fg"]
    # Skip warmup frame (index 0)
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


# ── Output formatting ─────────────────────────────────────────────────────────


def format_timing(method: str, frame_times: list[float], baseline_median: float | None = None) -> str:
    """Format timing line for one method."""
    if len(frame_times) <= 1:
        med = frame_times[0]
    else:
        med = statistics.median(frame_times[1:])  # exclude warmup

    line = f"  {method:15s}: {med:.1f}s median"
    if baseline_median is not None and baseline_median > 0 and method != "none (baseline)":
        speedup = baseline_median / med
        line += f" ({speedup:.2f}x faster)"
    return line


def format_memory(method: str, peak_bytes: int) -> str:
    """Format memory line for one method."""
    gb = peak_bytes / 1e9
    return f"  {method:15s}: peak={gb:.1f} GB"


def format_quality(method: str, metrics: dict[str, dict[str, float]]) -> list[str]:
    """Format quality lines for one method."""
    lines = [f"  {method}:"]
    for channel, m in metrics.items():
        lines.append(f"    {channel:8s}: MAE={m['mae']:.4f}, >{QUALITY_THRESHOLD}: {m['pct_above']:.1f}%")
    return lines


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Benchmark ROI methods: none / yolo / alpha_hint")
    parser.add_argument("--clip", required=True, help="Path to input RGB video")
    parser.add_argument("--alpha", required=True, help="Path to alpha hint video")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--device", default=None, help="Device override (cuda/mps/cpu)")
    parser.add_argument("--backend", choices=["auto", "torch", "mlx"], default="auto")
    parser.add_argument("--max-frames", type=int, default=MAX_BENCHMARK_FRAMES, help="Max frames (default 10)")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["none", "yolo", "alpha_hint"],
        default=["none", "yolo", "alpha_hint"],
        help="Which methods to benchmark (default: all three)",
    )

    args = parser.parse_args()

    device = args.device or get_device()
    print(f"Device: {device}")

    # Load frames
    frames = load_video_frames(args.clip, max_frames=args.max_frames)
    masks = load_mask_frames(args.alpha, max_frames=args.max_frames)
    num_frames = min(len(frames), len(masks))
    frames = frames[:num_frames]
    masks = masks[:num_frames]

    frame_h, frame_w = frames[0].shape[:2]
    print(f"Frame size: {frame_w}x{frame_h}, {num_frames} frames\n")

    # Create engine
    from CorridorKeyModule.backend import create_engine
    from CorridorKeyModule.roi_manager import ROIManager

    engine = create_engine(backend=args.backend, device=device)

    # Ensure "none" always runs first as baseline
    methods = list(args.methods)
    if "none" in methods:
        methods.remove("none")
        methods.insert(0, "none")
    else:
        # Need baseline for quality comparison — run it silently
        methods.insert(0, "none")

    # Warmup (1 frame, full-frame)
    print("Warming up...")
    engine.process_frame(frames[0], masks[0])
    sync_device(device)

    print(f"\n{'=' * 70}")
    print(f"ROI Benchmark — {num_frames} frames, {frame_w}x{frame_h}, device={device}")
    print(f"{'=' * 70}\n")

    # Run each method
    results: dict[str, dict] = {}

    for method in methods:
        roi_manager = None
        if method != "none":
            roi_manager = ROIManager(roi_method=method)

        print(f"--- {method} ---")
        outputs, times, peak_mem = run_method(
            engine, roi_manager, frames, masks, device, method
        )
        results[method] = {
            "outputs": outputs,
            "times": times,
            "peak_mem": peak_mem,
        }

    # ── Report ────────────────────────────────────────────────────────────────

    baseline_times = results["none"]["times"]
    baseline_median = statistics.median(baseline_times[1:]) if len(baseline_times) > 1 else baseline_times[0]

    print(f"\n{'=' * 70}")
    print("TIMING")
    for method in methods:
        times = results[method]["times"]
        label = f"{method} (baseline)" if method == "none" else method
        print(format_timing(label, times, baseline_median))

    print(f"\n{'=' * 70}")
    print("MEMORY")
    for method in methods:
        label = f"{method} (baseline)" if method == "none" else method
        print(format_memory(label, results[method]["peak_mem"]))

    print(f"\n{'=' * 70}")
    print("QUALITY vs BASELINE (none)")

    baseline_outputs = results["none"]["outputs"]
    for method in methods:
        if method == "none":
            continue
        metrics = compute_quality_metrics(baseline_outputs, results[method]["outputs"])
        if metrics:
            for line in format_quality(method, metrics):
                print(line)

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
