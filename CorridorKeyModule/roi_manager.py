"""ROI crop-and-paste pipeline: bucket padding, engine call, and reintegration.

Orchestrates the full ROI lifecycle:
  detection → stabilization → crop → bucket pad → engine → extract → reintegrate

Steps 3 and 4 of the Dynamic ROI plan.
"""

from __future__ import annotations

import cv2
import numpy as np

from .roi_detector import SubjectDetector
from .roi_stabilizer import ROIStabilizer


# ── Bucket selection ───────────────────────────────────────────────────────────

BUCKET_SIZES = [512, 1024, 2048]

# Green screen fill for padding (sRGB normalized [0,1])
PAD_FILL_RGB = (0.0, 0.69, 0.25)
# Alpha hint padding = no mask signal
PAD_FILL_MASK = 0.0

# Gaussian feather sigma for reintegration blending (pixels)
FEATHER_SIGMA = 15


def select_bucket(crop_width: int, crop_height: int) -> int | None:
    """Return the smallest power-of-2 bucket that fits the crop, or None if too large."""
    for size in BUCKET_SIZES:
        if crop_width <= size and crop_height <= size:
            return size
    return None


# ── Center-pad helpers ─────────────────────────────────────────────────────────


def center_pad_rgb(
    crop: np.ndarray,
    bucket_size: int,
    fill: tuple[float, float, float] = PAD_FILL_RGB,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Center-pad an RGB crop into a square bucket with green fill.

    Args:
        crop: [H, W, 3] float32 image.
        bucket_size: Target square dimension.
        fill: RGB fill color (normalized 0-1).

    Returns:
        (padded_image, (offset_x, offset_y)) — offset for un-padding later.
    """
    h, w = crop.shape[:2]
    padded = np.full((bucket_size, bucket_size, 3), fill, dtype=np.float32)
    offset_x = (bucket_size - w) // 2
    offset_y = (bucket_size - h) // 2
    padded[offset_y : offset_y + h, offset_x : offset_x + w] = crop
    return padded, (offset_x, offset_y)


def center_pad_mask(
    crop: np.ndarray,
    bucket_size: int,
    fill: float = PAD_FILL_MASK,
) -> np.ndarray:
    """Center-pad a single-channel mask into a square bucket.

    Args:
        crop: [H, W] or [H, W, 1] float32 mask.
        bucket_size: Target square dimension.
        fill: Fill value for padding region.

    Returns:
        Padded mask [bucket_size, bucket_size].
    """
    if crop.ndim == 3:
        crop = crop[:, :, 0]

    h, w = crop.shape[:2]
    padded = np.full((bucket_size, bucket_size), fill, dtype=np.float32)
    offset_x = (bucket_size - w) // 2
    offset_y = (bucket_size - h) // 2
    padded[offset_y : offset_y + h, offset_x : offset_x + w] = crop
    return padded


# ── Extract valid region from padded output ────────────────────────────────────


def extract_valid_region(
    padded_output: np.ndarray,
    crop_width: int,
    crop_height: int,
    offset: tuple[int, int],
) -> np.ndarray:
    """Slice the valid crop region from a padded bucket output.

    Works for any channel count: [H, W], [H, W, 1], [H, W, 3], [H, W, 4].
    """
    ox, oy = offset
    if padded_output.ndim == 2:
        return padded_output[oy : oy + crop_height, ox : ox + crop_width]
    return padded_output[oy : oy + crop_height, ox : ox + crop_width]


# ── Gaussian feathered reintegration ───────────────────────────────────────────


def create_feather_mask(
    frame_height: int,
    frame_width: int,
    crop_box: tuple[int, int, int, int],
    sigma: int = FEATHER_SIGMA,
) -> np.ndarray:
    """Create a soft-edged blend mask for reintegrating a crop into the full frame.

    Returns:
        [H, W] float32 mask — 1.0 inside crop, soft falloff at edges.
    """
    x1, y1, x2, y2 = crop_box
    mask = np.zeros((frame_height, frame_width), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0

    if sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)

    return mask


def reintegrate(
    full_frame_shape: tuple[int, int],
    crop_box: tuple[int, int, int, int],
    crop_result: dict[str, np.ndarray],
    sigma: int = FEATHER_SIGMA,
) -> dict[str, np.ndarray]:
    """Paste processed crop results back into full-frame outputs with feathered edges.

    Args:
        full_frame_shape: (height, width) of the original full-resolution frame.
        crop_box: (x1, y1, x2, y2) crop coordinates in full-frame space.
        crop_result: Engine output dict with 'alpha', 'fg', 'comp', 'processed' keys.
        sigma: Gaussian blur sigma for feathered blending.

    Returns:
        Dict matching engine output format, at full-frame resolution.
    """
    frame_h, frame_w = full_frame_shape
    x1, y1, x2, y2 = crop_box

    feather = create_feather_mask(frame_h, frame_w, crop_box, sigma)
    feather_3d = feather[:, :, np.newaxis]  # [H, W, 1] for broadcasting

    output = {}

    for key, crop_data in crop_result.items():
        if crop_data is None:
            output[key] = None
            continue

        channels = crop_data.shape[-1] if crop_data.ndim == 3 else 1

        # Create full-frame canvas (zeros = transparent/black background)
        if crop_data.ndim == 3:
            canvas = np.zeros((frame_h, frame_w, channels), dtype=np.float32)
            canvas[y1:y2, x1:x2] = crop_data
            output[key] = canvas * feather_3d
        else:
            canvas = np.zeros((frame_h, frame_w), dtype=np.float32)
            canvas[y1:y2, x1:x2] = crop_data
            output[key] = canvas * feather

    return output


# ── ROI Manager (top-level orchestrator) ───────────────────────────────────────


class ROIManager:
    """Orchestrates the full ROI crop-and-paste pipeline.

    Owns: detection → stabilization → crop → bucket pad → engine call → reintegration.

    Usage:
        manager = ROIManager()
        # In frame loop:
        result = manager.process_with_roi(engine, img_srgb, mask_linear, **kwargs)
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.3,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        feather_sigma: int = FEATHER_SIGMA,
    ) -> None:
        self._detector = SubjectDetector(confidence_threshold=confidence_threshold)
        self._stabilizer: ROIStabilizer | None = None
        self._frame_idx: int = 0
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._feather_sigma = feather_sigma

    def reset(self) -> None:
        """Reset all state for a new clip."""
        self._stabilizer = None
        self._frame_idx = 0

    def process_with_roi(
        self,
        engine,
        image: np.ndarray,
        mask_linear: np.ndarray,
        **engine_kwargs,
    ) -> dict[str, np.ndarray]:
        """Process a frame through the ROI pipeline, falling back to full-frame if needed.

        Args:
            engine: CorridorKeyEngine (or any object with process_frame method).
            image: [H, W, 3] float32 sRGB image (0-1).
            mask_linear: [H, W] or [H, W, 1] float32 linear mask (0-1).
            **engine_kwargs: Passed through to engine.process_frame().

        Returns:
            Engine result dict at original frame resolution.
        """
        frame_h, frame_w = image.shape[:2]

        # Lazily initialize stabilizer with frame dimensions
        if self._stabilizer is None:
            self._stabilizer = ROIStabilizer(
                frame_w,
                frame_h,
                min_cutoff=self._min_cutoff,
                beta=self._beta,
            )

        # Step 1: Detect subject
        raw_bbox = self._detector.detect(image)

        # Step 2: Stabilize + lock
        crop_box = self._stabilizer.update(float(self._frame_idx), raw_bbox)
        self._frame_idx += 1

        # No crop available → full-frame fallback
        if crop_box is None:
            return engine.process_frame(image, mask_linear, **engine_kwargs)

        x1, y1, x2, y2 = crop_box
        crop_w = x2 - x1
        crop_h = y2 - y1

        # Step 3: Select bucket
        bucket = select_bucket(crop_w, crop_h)

        if bucket is None:
            # Crop exceeds all buckets → full-frame fallback
            return engine.process_frame(image, mask_linear, **engine_kwargs)

        # Crop the frame and mask
        img_crop = image[y1:y2, x1:x2].copy()
        mask_2d = mask_linear[:, :, 0] if mask_linear.ndim == 3 else mask_linear
        mask_crop = mask_2d[y1:y2, x1:x2].copy()

        # Center-pad into bucket
        img_padded, offset = center_pad_rgb(img_crop, bucket)
        mask_padded = center_pad_mask(mask_crop, bucket)

        # Call engine at bucket size (img_size override → resize is a no-op)
        bucket_result = engine.process_frame(
            img_padded,
            mask_padded,
            img_size=bucket,
            **engine_kwargs,
        )

        # Extract valid region from padded output
        extracted = {}
        for key, data in bucket_result.items():
            if data is not None:
                extracted[key] = extract_valid_region(data, crop_w, crop_h, offset)
            else:
                extracted[key] = None

        # Step 4: Reintegrate into full-frame with feathered blending
        return reintegrate(
            (frame_h, frame_w),
            crop_box,
            extracted,
            sigma=self._feather_sigma,
        )
