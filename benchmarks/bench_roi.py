"""Benchmark: ROI crop-and-paste vs full-frame processing.

Measures per-frame inference time and memory for both modes using real footage.

Usage:
  python benchmarks/bench_roi.py --clip <input_video> --alpha <alpha_video>
  python benchmarks/bench_roi.py --clip <input_video> --alpha <alpha_video> --max-frames 10
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


# ── Frame loading (matches bench_phase.py conventions) ─────────────────────────

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


def measure_memory_before(device: str) -> int:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated()
    elif device == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()
        return torch.mps.driver_allocated_memory()
    return 0


def measure_memory_after(device: str) -> dict:
    if device == "cuda":
        torch.cuda.synchronize()
        return {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "peak_bytes": torch.cuda.max_memory_allocated(),
        }
    elif device == "mps":
        torch.mps.synchronize()
        mem = torch.mps.driver_allocated_memory()
        return {"allocated_bytes": mem, "peak_bytes": mem}
    return {"allocated_bytes": 0, "peak_bytes": 0}


# ── Benchmark core ─────────────────────────────────────────────────────────────


def run_mode(
    engine,
    roi_manager,
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    device: str,
    use_roi: bool,
    label: str,
) -> tuple[list[dict], list[float], dict]:
    """Run inference in one mode, return (outputs, frame_times, mem_info)."""
    if roi_manager is not None and use_roi:
        roi_manager.reset()

    outputs = []
    frame_times = []

    mem_before = measure_memory_before(device)

    for i, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
        sync_device(device)
        t0 = time.perf_counter()

        if use_roi and roi_manager is not None:
            result = roi_manager.process_with_roi(engine, frame, mask)
        else:
            result = engine.process_frame(frame, mask)

        sync_device(device)
        t1 = time.perf_counter()

        frame_times.append(t1 - t0)
        outputs.append(result)
        print(f"  [{label}] Frame {i + 1}/{len(frames)}: {t1 - t0:.3f}s", end="\r")

    print()

    mem_info = measure_memory_after(device)
    mem_info["before_bytes"] = mem_before
    mem_info["delta_bytes"] = mem_info["allocated_bytes"] - mem_before

    return outputs, frame_times, mem_info


def print_timing(label: str, frame_times: list[float]) -> None:
    if len(frame_times) <= 1:
        print(f"  {label}: {frame_times[0]:.3f}s (single frame)")
        return

    steady = frame_times[1:]  # exclude warmup
    med = statistics.median(steady)
    mean = statistics.mean(steady)
    print(f"  {label}: {med:.3f}s median, {mean:.3f}s mean "
          f"(min={min(steady):.3f}, max={max(steady):.3f}, warmup={frame_times[0]:.3f})")


def print_memory(label: str, mem_info: dict) -> None:
    gb = 1e9
    print(f"  {label}: peak={mem_info['peak_bytes'] / gb:.2f} GB, "
          f"delta={mem_info['delta_bytes'] / gb:.2f} GB")


def pixel_diff_summary(full_outputs: list[dict], roi_outputs: list[dict]) -> None:
    """Compare ROI vs full-frame outputs to quantify quality impact."""
    num_frames = min(len(full_outputs), len(roi_outputs))
    channels = ["alpha", "fg", "comp", "processed"]

    print(f"\n  Quality comparison ({num_frames} frames):")
    for key in channels:
        maes = []
        max_errs = []
        for i in range(num_frames):
            full = full_outputs[i].get(key)
            roi = roi_outputs[i].get(key)
            if full is None or roi is None:
                continue
            # Shapes may differ if ROI returned full-frame fallback
            if full.shape != roi.shape:
                continue
            diff = np.abs(full.astype(np.float64) - roi.astype(np.float64))
            maes.append(float(diff.mean()))
            max_errs.append(float(diff.max()))

        if maes:
            avg_mae = statistics.mean(maes)
            worst_max = max(max_errs)
            mse = statistics.mean(m**2 for m in maes) if maes else 0
            psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))
            print(f"    {key:12s}: MAE={avg_mae:.6f}, max_err={worst_max:.6f}, PSNR={psnr:.1f} dB")


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Benchmark ROI vs full-frame inference")
    parser.add_argument("--clip", required=True, help="Path to input RGB video")
    parser.add_argument("--alpha", required=True, help="Path to alpha hint video")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--device", default=None, help="Device override (cuda/mps/cpu)")
    parser.add_argument("--backend", choices=["auto", "torch", "mlx"], default="auto")
    parser.add_argument("--max-frames", type=int, default=MAX_BENCHMARK_FRAMES, help="Max frames (default 10)")

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
    roi_manager = ROIManager()

    # Warmup (1 frame each mode)
    print("Warming up...")
    roi_manager.reset()
    engine.process_frame(frames[0], masks[0])
    sync_device(device)
    roi_manager.process_with_roi(engine, frames[0], masks[0])
    sync_device(device)

    print(f"\n{'='*70}")
    print(f"ROI Benchmark — {num_frames} frames, {frame_w}x{frame_h}, device={device}")
    print(f"{'='*70}\n")

    # Run full-frame mode
    print("--- Full Frame (no ROI) ---")
    full_outputs, full_times, full_mem = run_mode(
        engine, roi_manager, frames, masks, device, use_roi=False, label="Full"
    )

    # Run ROI mode
    print("\n--- With ROI ---")
    roi_outputs, roi_times, roi_mem = run_mode(
        engine, roi_manager, frames, masks, device, use_roi=True, label="ROI"
    )

    # Report ROI detection info
    print("\n  ROI detection summary:")
    from CorridorKeyModule.roi_manager import select_bucket

    roi_manager.reset()
    for i, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
        bbox = roi_manager._detector.detect(frame)
        if bbox:
            x1, y1, x2, y2 = bbox
            crop_w, crop_h = x2 - x1, y2 - y1
            bucket = select_bucket(int(crop_w * 1.4), int(crop_h * 1.4))
            bucket_str = f"{bucket}x{bucket}" if bucket else "fallback"
            print(f"    Frame {i + 1}: bbox={crop_w}x{crop_h} → bucket {bucket_str}")
        else:
            print(f"    Frame {i + 1}: no detection (full-frame fallback)")
        if i >= 4:
            remaining = len(frames) - 5
            if remaining > 0:
                print(f"    ... and {remaining} more frames")
            break

    # Timing comparison
    print(f"\n{'='*70}")
    print("TIMING")
    print(f"{'='*70}")
    print_timing("Full frame", full_times)
    print_timing("With ROI  ", roi_times)

    full_steady = full_times[1:] if len(full_times) > 1 else full_times
    roi_steady = roi_times[1:] if len(roi_times) > 1 else roi_times

    if roi_steady and full_steady:
        full_med = statistics.median(full_steady)
        roi_med = statistics.median(roi_steady)
        if roi_med > 0:
            speedup = full_med / roi_med
            saved_pct = (1 - roi_med / full_med) * 100
            direction = "faster" if speedup > 1 else "slower"
            print(f"\n  ROI is {speedup:.2f}x {direction} ({saved_pct:+.1f}%) by median")

    # Memory comparison
    print(f"\n{'='*70}")
    print("MEMORY")
    print(f"{'='*70}")
    print_memory("Full frame", full_mem)
    print_memory("With ROI  ", roi_mem)

    # Quality comparison
    print(f"\n{'='*70}")
    print("QUALITY (ROI vs Full Frame)")
    print(f"{'='*70}")
    pixel_diff_summary(full_outputs, roi_outputs)

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
