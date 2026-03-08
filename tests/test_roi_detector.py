"""Tests for the ROI subject detector (YOLOv11 CPU localization)."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def detector():
    """Load the SubjectDetector once for all tests in this module."""
    from CorridorKeyModule.roi_detector import SubjectDetector

    return SubjectDetector()


class TestSubjectDetector:
    """Step 1: CPU-Bound Subject Localization tests."""

    def test_returns_none_for_blank_frame(self, detector):
        """A pure green frame should return None (no subject)."""
        green_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        green_frame[:, :, 1] = 200  # Green channel
        result = detector.detect(green_frame)
        assert result is None

    def test_bbox_within_frame_bounds(self, detector):
        """Any detected bbox must be within [0, W) x [0, H)."""
        # Create a simple test image with a dark rectangle on green (simulating a subject)
        h, w = 2160, 3840
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 1] = 200  # Green background

        # Draw a dark rectangle to simulate a person-like shape
        cx, cy = w // 2, h // 2
        rect_w, rect_h = 400, 1200
        frame[cy - rect_h // 2 : cy + rect_h // 2, cx - rect_w // 2 : cx + rect_w // 2] = [80, 50, 50]

        result = detector.detect(frame)

        if result is not None:
            x1, y1, x2, y2 = result
            assert 0 <= x1 < x2 <= w, f"x coords out of bounds: {x1}, {x2}"
            assert 0 <= y1 < y2 <= h, f"y coords out of bounds: {y1}, {y2}"

    def test_accepts_float32_input(self, detector):
        """Detector should handle float32 [0-1] frames (our pipeline format)."""
        frame = np.zeros((1080, 1920, 3), dtype=np.float32)
        frame[:, :, 1] = 0.8  # Green
        # Should not raise
        result = detector.detect(frame)
        # Result can be None (no subject in plain green) — just verify no crash
        assert result is None or len(result) == 4

    def test_coordinate_scaling_4k(self, detector):
        """Bounding box coordinates should scale correctly from 640 to 4K."""
        # Use a 4K frame — if detection occurs, coords must be in 4K space
        h, w = 2160, 3840
        frame = np.full((h, w, 3), 200, dtype=np.uint8)  # Gray frame

        result = detector.detect(frame)

        if result is not None:
            x1, y1, x2, y2 = result
            # Coords must be in original 4K space, not 640x640
            assert x2 <= w
            assert y2 <= h

    def test_model_is_on_cpu(self, detector):
        """YOLO model must be forced to CPU (zero GPU VRAM)."""
        # Access the internal model's device
        model_device = str(detector._model.device)
        assert "cpu" in model_device.lower(), f"Model is on {model_device}, expected CPU"
