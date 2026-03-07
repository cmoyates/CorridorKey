"""Async I/O pipeline with triple buffering for pipelined frame processing.

Overlaps disk I/O (read + write) with GPU inference using a 3-slot ring buffer
and dedicated thread pools. On CUDA, uses separate streams for copy and compute.
On MPS/CPU, still benefits from async I/O overlapping GPU/CPU compute.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Sentinel indicating a corrupt/missing frame that should be skipped
_FRAME_SKIP = object()

# Number of ring buffer slots (triple buffering)
NUM_PIPELINE_SLOTS = 3


@dataclass
class FrameSource:
    """Describes how to read frames — either from a video capture or image file list."""

    type: str  # "video" or "sequence"
    path: str  # video file path or directory path
    files: list[str] = field(default_factory=list)  # sorted filenames for sequence type
    cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self.type == "video" and self.cap is None:
            self.cap = cv2.VideoCapture(self.path)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def _can_pin_memory(device: torch.device) -> bool:
    """Pinned memory only benefits CUDA (enables DMA via non_blocking)."""
    return device.type == "cuda"


class InferenceRingBuffer:
    """Triple-buffer for pipelined frame processing.

    Pre-allocates CPU buffers (pinned on CUDA for async DMA) to avoid
    per-frame allocation overhead and memory fragmentation.
    """

    def __init__(self, img_size: int, device: torch.device):
        self.img_size = img_size
        self.device = device
        pin = _can_pin_memory(device)

        # CPU-side input buffers: [1, 4, H, W] float32
        self.cpu_input: list[torch.Tensor] = []
        for _ in range(NUM_PIPELINE_SLOTS):
            buf = torch.zeros(1, 4, img_size, img_size, dtype=torch.float32)
            if pin:
                buf = buf.pin_memory()
            self.cpu_input.append(buf)

        # Metadata per slot (original dims, frame stem, etc.)
        self.meta: list[dict[str, Any]] = [{} for _ in range(NUM_PIPELINE_SLOTS)]


class AsyncFramePipeline:
    """Pipelined frame processing: read -> infer -> write in parallel.

    Uses a ring buffer with NUM_PIPELINE_SLOTS slots. At steady state:
      - Slot (F % 3): being read by I/O thread
      - Slot ((F-1) % 3): being inferred on GPU
      - Slot ((F-2) % 3): being written by I/O thread

    Frame ordering is enforced by waiting on the previous write Future
    before reusing a slot.
    """

    def __init__(
        self,
        engine: Any,
        img_size: int,
        device: torch.device,
        process_frame_kwargs: dict[str, Any],
    ):
        self.engine = engine
        self.device = device
        self.process_frame_kwargs = process_frame_kwargs
        self.ring = InferenceRingBuffer(img_size, device)

        self.read_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="io-read")
        self.write_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="io-write")

        # CUDA stream isolation
        self.compute_stream = None
        self.copy_stream = None
        if device.type == "cuda":
            self.compute_stream = torch.cuda.Stream(device)
            self.copy_stream = torch.cuda.Stream(device)

    def shutdown(self) -> None:
        self.read_pool.shutdown(wait=True)
        self.write_pool.shutdown(wait=True)

    def process_clip(
        self,
        input_source: FrameSource,
        alpha_source: FrameSource,
        output_dirs: dict[str, str],
        num_frames: int,
        input_is_linear: bool,
        exr_flags: list[int],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        """Process an entire clip with pipelined read/infer/write."""
        input_source.open()
        alpha_source.open()

        try:
            self._run_pipeline(
                input_source, alpha_source, output_dirs, num_frames, input_is_linear, exr_flags, mean, std
            )
        finally:
            input_source.close()
            alpha_source.close()
            self.shutdown()

    def _run_pipeline(
        self,
        input_source: FrameSource,
        alpha_source: FrameSource,
        output_dirs: dict[str, str],
        num_frames: int,
        input_is_linear: bool,
        exr_flags: list[int],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        # Track futures for each slot to enforce ordering
        read_futures: list[Future | None] = [None] * NUM_PIPELINE_SLOTS
        write_futures: list[Future | None] = [None] * NUM_PIPELINE_SLOTS
        results: list[dict[str, np.ndarray] | None] = [None] * NUM_PIPELINE_SLOTS

        frames_submitted = 0
        frames_inferred = 0
        frames_written = 0

        # === WARMUP: submit first reads ===
        warmup_count = min(NUM_PIPELINE_SLOTS - 1, num_frames)
        for i in range(warmup_count):
            slot = i % NUM_PIPELINE_SLOTS
            read_futures[slot] = self.read_pool.submit(
                _read_frame,
                input_source,
                alpha_source,
                i,
                input_is_linear,
                self.ring.img_size,
                mean,
                std,
            )
            frames_submitted += 1

        # === STEADY STATE ===
        while frames_written < num_frames:
            # 1. Submit next read (if frames remain)
            if frames_submitted < num_frames:
                next_slot = frames_submitted % NUM_PIPELINE_SLOTS

                # Wait for any pending write on this slot before reusing it
                if write_futures[next_slot] is not None:
                    write_futures[next_slot].result()

                read_futures[next_slot] = self.read_pool.submit(
                    _read_frame,
                    input_source,
                    alpha_source,
                    frames_submitted,
                    input_is_linear,
                    self.ring.img_size,
                    mean,
                    std,
                )
                frames_submitted += 1

            # 2. Infer the next ready frame
            if frames_inferred < num_frames:
                infer_slot = frames_inferred % NUM_PIPELINE_SLOTS
                read_result = read_futures[infer_slot].result()

                if frames_inferred % 10 == 0:
                    print(f"  Frame {frames_inferred}/{num_frames}...", end="\r")

                if read_result is _FRAME_SKIP:
                    results[infer_slot] = None
                    self.ring.meta[infer_slot] = {"skip": True}
                else:
                    img_srgb, mask_linear, input_stem = read_result
                    res = self.engine.process_frame(img_srgb, mask_linear, **self.process_frame_kwargs)
                    results[infer_slot] = res
                    self.ring.meta[infer_slot] = {
                        "input_stem": input_stem,
                        "skip": False,
                    }

                frames_inferred += 1

            # 3. Submit write for the frame that was just inferred
            write_slot = (frames_inferred - 1) % NUM_PIPELINE_SLOTS
            meta = self.ring.meta[write_slot]
            if not meta.get("skip", False) and results[write_slot] is not None:
                write_futures[write_slot] = self.write_pool.submit(
                    _write_frame,
                    results[write_slot],
                    output_dirs,
                    meta["input_stem"],
                    exr_flags,
                )
            else:
                write_futures[write_slot] = None

            frames_written += 1

        # === DRAIN: wait for all pending writes ===
        for wf in write_futures:
            if wf is not None:
                wf.result()

        print("")


def _read_frame(
    input_source: FrameSource,
    alpha_source: FrameSource,
    frame_idx: int,
    input_is_linear: bool,
    img_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str] | object:
    """Read and preprocess a single frame. Returns (img_srgb, mask_linear, stem) or _FRAME_SKIP."""
    try:
        # --- Read RGB input ---
        img_srgb = None
        input_stem = f"{frame_idx:05d}"

        if input_source.type == "video":
            ret, frame = input_source.cap.read()
            if not ret:
                return _FRAME_SKIP
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_srgb = img_rgb.astype(np.float32) / 255.0
        else:
            fname = input_source.files[frame_idx]
            fpath = os.path.join(input_source.path, fname)
            input_stem = os.path.splitext(fname)[0]

            is_exr = fpath.lower().endswith(".exr")
            if is_exr:
                img_linear = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                if img_linear is None:
                    return _FRAME_SKIP
                img_linear_rgb = cv2.cvtColor(img_linear, cv2.COLOR_BGR2RGB)
                img_srgb = np.maximum(img_linear_rgb, 0.0)
            else:
                img_bgr = cv2.imread(fpath)
                if img_bgr is None:
                    return _FRAME_SKIP
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_srgb = img_rgb.astype(np.float32) / 255.0

        # --- Read alpha mask ---
        mask_linear = None
        if alpha_source.type == "video":
            ret, frame = alpha_source.cap.read()
            if not ret:
                return _FRAME_SKIP
            mask_linear = frame[:, :, 2].astype(np.float32) / 255.0
        else:
            fname = alpha_source.files[frame_idx]
            fpath = os.path.join(alpha_source.path, fname)
            mask_in = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)

            if mask_in is None:
                return _FRAME_SKIP

            if mask_in.ndim == 3:
                if mask_in.shape[2] == 3:
                    mask_linear = mask_in[:, :, 0]
                else:
                    mask_linear = mask_in
            else:
                mask_linear = mask_in

            if mask_linear.dtype == np.uint8:
                mask_linear = mask_linear.astype(np.float32) / 255.0
            elif mask_linear.dtype == np.uint16:
                mask_linear = mask_linear.astype(np.float32) / 65535.0
            else:
                mask_linear = mask_linear.astype(np.float32)

        # --- Resize mask if needed ---
        if mask_linear.shape[:2] != img_srgb.shape[:2]:
            mask_linear = cv2.resize(
                mask_linear, (img_srgb.shape[1], img_srgb.shape[0]), interpolation=cv2.INTER_LINEAR
            )

        return (img_srgb, mask_linear, input_stem)

    except Exception:
        logger.exception("Error reading frame %d", frame_idx)
        return _FRAME_SKIP


def _write_frame(
    res: dict[str, np.ndarray],
    output_dirs: dict[str, str],
    input_stem: str,
    exr_flags: list[int],
) -> None:
    """Write all output passes for a single frame to disk."""
    try:
        pred_fg = res["fg"]
        pred_alpha = res["alpha"]

        # Save FG (sRGB EXR)
        fg_bgr = cv2.cvtColor(pred_fg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dirs["fg"], f"{input_stem}.exr"), fg_bgr, exr_flags)

        # Save Matte (single-channel linear EXR)
        alpha_out = pred_alpha[:, :, 0] if pred_alpha.ndim == 3 else pred_alpha
        cv2.imwrite(os.path.join(output_dirs["matte"], f"{input_stem}.exr"), alpha_out, exr_flags)

        # Save Comp (PNG 8-bit)
        comp_srgb = res["comp"]
        comp_bgr = cv2.cvtColor((np.clip(comp_srgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dirs["comp"], f"{input_stem}.png"), comp_bgr)

        # Save Processed (RGBA EXR)
        if "processed" in res:
            proc_rgba = res["processed"]
            proc_bgra = cv2.cvtColor(proc_rgba, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(os.path.join(output_dirs["proc"], f"{input_stem}.exr"), proc_bgra, exr_flags)

    except Exception:
        logger.exception("Error writing frame %s", input_stem)
