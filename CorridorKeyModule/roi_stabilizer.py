"""Temporal stabilization for ROI bounding boxes.

Smooths raw YOLO detections across frames using a 1-Euro low-pass filter,
applies padding, and locks crop windows to prevent ViT patch-grid jitter.

Reference: Casiez et al. 2012, "1€ Filter: A Simple Speed-based Low-pass Filter
for Noisy Input in Interactive Systems"
"""

from __future__ import annotations

import math

# ── 1-Euro Filter ──────────────────────────────────────────────────────────────

class OneEuroFilter:
    """1-Euro adaptive low-pass filter for a single scalar signal.

    Smooths noisy input with a cutoff frequency that adapts to signal speed:
    slow movements get heavy smoothing, fast movements pass through.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _smoothing_factor(te: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * te
        return r / (r + 1.0)

    def reset(self) -> None:
        """Clear filter state (e.g. on scene cut)."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, timestamp: float, value: float) -> float:
        """Filter a new sample.

        Args:
            timestamp: Monotonically increasing time (e.g. frame index or seconds).
            value: Raw noisy measurement.

        Returns:
            Smoothed value.
        """
        if self._t_prev is None:
            # First sample — no smoothing possible
            self._x_prev = value
            self._dx_prev = 0.0
            self._t_prev = timestamp
            return value

        te = timestamp - self._t_prev
        if te <= 0:
            te = 1e-6  # avoid division by zero for duplicate timestamps

        # Derivative estimation
        a_d = self._smoothing_factor(te, self.d_cutoff)
        dx = (value - self._x_prev) / te
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(te, cutoff)
        x_hat = a * value + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = timestamp

        return x_hat


# ── Bbox smoothing helpers ─────────────────────────────────────────────────────

# Scene-cut threshold: centroid shift > 30% of frame dimension triggers reset
SCENE_CUT_THRESHOLD = 0.30

# Padding fraction applied to each side of the smoothed bbox
PADDING_FRACTION = 0.20


def _bbox_centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _pad_bbox(
    bbox: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    fraction: float = PADDING_FRACTION,
) -> tuple[int, int, int, int]:
    """Expand bbox by *fraction* on each side, clamp to frame bounds, return ints."""
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * fraction
    pad_y = (y2 - y1) * fraction

    x1_pad = max(0, int(round(x1 - pad_x)))
    y1_pad = max(0, int(round(y1 - pad_y)))
    x2_pad = min(frame_width, int(round(x2 + pad_x)))
    y2_pad = min(frame_height, int(round(y2 + pad_y)))

    return (x1_pad, y1_pad, x2_pad, y2_pad)


# ── Crop Lock Manager ─────────────────────────────────────────────────────────

# Minimum frames before a locked crop can be re-evaluated
MIN_LOCK_FRAMES = 10

# If the subject bbox edge is within this fraction of the locked crop edge, unlock
BOUNDARY_THRESHOLD = 0.05


class CropLockManager:
    """Locks and manages a stable crop window across frames.

    Prevents rapid lock/unlock oscillation via a minimum lock duration
    and only unlocks when the subject approaches the crop boundary.
    """

    def __init__(
        self,
        min_lock_frames: int = MIN_LOCK_FRAMES,
        boundary_threshold: float = BOUNDARY_THRESHOLD,
    ) -> None:
        self.min_lock_frames = min_lock_frames
        self.boundary_threshold = boundary_threshold

        self.locked_crop: tuple[int, int, int, int] | None = None
        self._frames_since_lock: int = 0

    def reset(self) -> None:
        """Clear lock state (e.g. on scene cut)."""
        self.locked_crop = None
        self._frames_since_lock = 0

    def _subject_near_boundary(
        self,
        subject_bbox: tuple[float, float, float, float],
    ) -> bool:
        """Check if the unpadded subject bbox is within threshold of the locked crop edge."""
        if self.locked_crop is None:
            return False

        lx1, ly1, lx2, ly2 = self.locked_crop
        sx1, sy1, sx2, sy2 = subject_bbox

        crop_w = lx2 - lx1
        crop_h = ly2 - ly1
        thresh_x = crop_w * self.boundary_threshold
        thresh_y = crop_h * self.boundary_threshold

        # Subject edge within threshold of *any* locked crop edge → near boundary
        return (
            (sx1 - lx1) < thresh_x
            or (lx2 - sx2) < thresh_x
            or (sy1 - ly1) < thresh_y
            or (ly2 - sy2) < thresh_y
        )

    def update(
        self,
        smoothed_bbox: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        """Given a smoothed (unpadded) bbox, return the crop to use this frame.

        Handles locking, unlock evaluation, and re-locking.

        Returns:
            Integer (x1, y1, x2, y2) crop coordinates.
        """
        if self.locked_crop is None:
            # First detection — lock immediately
            padded = _pad_bbox(smoothed_bbox, frame_width, frame_height)
            self.locked_crop = padded
            self._frames_since_lock = 0
            return padded

        self._frames_since_lock += 1

        # Check unlock conditions
        can_re_evaluate = self._frames_since_lock >= self.min_lock_frames
        near_edge = self._subject_near_boundary(smoothed_bbox)

        if can_re_evaluate and near_edge:
            # Unlock and re-lock with new padded bbox
            padded = _pad_bbox(smoothed_bbox, frame_width, frame_height)
            self.locked_crop = padded
            self._frames_since_lock = 0
            return padded

        # Stay locked
        return self.locked_crop

    def get_current_crop(self) -> tuple[int, int, int, int] | None:
        """Return the current locked crop, or None if not locked."""
        return self.locked_crop


# ── ROI Stabilizer (top-level API) ─────────────────────────────────────────────

class ROIStabilizer:
    """Combines 1-Euro filtering, padding, and crop locking into a single API.

    Usage:
        stabilizer = ROIStabilizer(frame_width=3840, frame_height=2160)
        for frame_idx, raw_bbox in enumerate(detections):
            crop = stabilizer.update(frame_idx, raw_bbox)
            # crop is (x1, y1, x2, y2) or None (full-frame fallback)
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        *,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        min_lock_frames: int = MIN_LOCK_FRAMES,
        boundary_threshold: float = BOUNDARY_THRESHOLD,
        scene_cut_threshold: float = SCENE_CUT_THRESHOLD,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.scene_cut_threshold = scene_cut_threshold

        # 4 independent 1-Euro filters: x1, y1, x2, y2
        self._filters = [
            OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(4)
        ]
        self._lock = CropLockManager(min_lock_frames, boundary_threshold)
        self._prev_centroid: tuple[float, float] | None = None

    def reset(self) -> None:
        """Full state reset (scene cut or new clip)."""
        for f in self._filters:
            f.reset()
        self._lock.reset()
        self._prev_centroid = None

    def _detect_scene_cut(self, bbox: tuple[int, int, int, int]) -> bool:
        """Return True if the bbox centroid jumped more than threshold * frame size."""
        if self._prev_centroid is None:
            return False

        cx, cy = _bbox_centroid(bbox)
        px, py = self._prev_centroid

        dx = abs(cx - px) / self.frame_width
        dy = abs(cy - py) / self.frame_height

        return dx > self.scene_cut_threshold or dy > self.scene_cut_threshold

    def update(
        self,
        timestamp: float,
        raw_bbox: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        """Process a single frame's detection and return a stable crop region.

        Args:
            timestamp: Monotonic time value (frame index works fine).
            raw_bbox: YOLO detection (x1, y1, x2, y2) or None if no detection.

        Returns:
            Stable crop (x1, y1, x2, y2) in frame coordinates, or None if
            no crop is available (no prior lock and no detection).
        """
        if raw_bbox is None:
            # No detection — use locked crop if available
            return self._lock.get_current_crop()

        # Scene cut detection
        if self._detect_scene_cut(raw_bbox):
            self.reset()

        # Smooth bbox coordinates through 1-Euro filters
        smoothed = tuple(
            f(timestamp, float(v)) for f, v in zip(self._filters, raw_bbox)
        )

        # Update centroid for next frame's scene-cut detection
        self._prev_centroid = _bbox_centroid(smoothed)

        # Lock manager handles padding and locking
        return self._lock.update(smoothed, self.frame_width, self.frame_height)
