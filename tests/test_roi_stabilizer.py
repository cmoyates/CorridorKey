"""Tests for the ROI temporal stabilizer (1-Euro filter + crop locking)."""

from __future__ import annotations

import pytest

from CorridorKeyModule.roi_stabilizer import (
    BOUNDARY_THRESHOLD,
    MIN_LOCK_FRAMES,
    PADDING_FRACTION,
    CropLockManager,
    OneEuroFilter,
    ROIStabilizer,
    _pad_bbox,
)


# ── 1-Euro Filter ──────────────────────────────────────────────────────────────


class TestOneEuroFilter:
    """Verify the 1-Euro low-pass filter smooths noisy signals."""

    def test_first_sample_passes_through(self):
        f = OneEuroFilter()
        assert f(0.0, 100.0) == 100.0

    def test_smooths_noisy_signal(self):
        """A constant signal with noise should converge near the true value."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
        true_value = 500.0
        noise = [3, -2, 5, -1, 4, -3, 2, -4, 1, -2]

        smoothed = []
        for i, n in enumerate(noise):
            smoothed.append(f(float(i), true_value + n))

        # Last few values should be closer to true_value than the raw noise
        last_errors = [abs(s - true_value) for s in smoothed[-3:]]
        raw_errors = [abs(n) for n in noise[-3:]]
        assert max(last_errors) < max(raw_errors)

    def test_fast_movement_passes_through(self):
        """High beta means fast movements should barely be smoothed."""
        f = OneEuroFilter(min_cutoff=1.0, beta=100.0)  # very high speed response
        f(0.0, 0.0)
        result = f(1.0, 1000.0)
        # With high beta the cutoff is very high → low smoothing
        assert result > 900.0

    def test_reset_clears_state(self):
        f = OneEuroFilter()
        f(0.0, 100.0)
        f(1.0, 200.0)
        f.reset()
        # After reset, next call behaves like first sample
        assert f(2.0, 50.0) == 50.0


# ── Padding ────────────────────────────────────────────────────────────────────


class TestPadding:
    """Verify bbox padding and clamping."""

    def test_20_percent_padding(self):
        bbox = (100.0, 100.0, 200.0, 300.0)  # 100w x 200h
        padded = _pad_bbox(bbox, 3840, 2160)
        # Expected: x ±20, y ±40
        assert padded == (80, 60, 220, 340)

    def test_clamp_to_frame_bounds(self):
        bbox = (10.0, 10.0, 50.0, 50.0)  # near top-left corner
        padded = _pad_bbox(bbox, 100, 100)
        x1, y1, x2, y2 = padded
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 100
        assert y2 <= 100

    def test_full_frame_bbox_clamps(self):
        bbox = (0.0, 0.0, 3840.0, 2160.0)
        padded = _pad_bbox(bbox, 3840, 2160)
        assert padded == (0, 0, 3840, 2160)


# ── CropLockManager ───────────────────────────────────────────────────────────


class TestCropLockManager:
    """Verify crop locking, unlock, and oscillation prevention."""

    def test_first_detection_locks(self):
        lock = CropLockManager()
        crop = lock.update((500.0, 500.0, 700.0, 900.0), 3840, 2160)
        assert crop is not None
        assert lock.locked_crop == crop

    def test_lock_persists_across_frames(self):
        lock = CropLockManager()
        # Initial lock
        first = lock.update((500.0, 500.0, 700.0, 900.0), 3840, 2160)

        # Same-ish bbox on subsequent frames — should stay locked
        for _ in range(5):
            crop = lock.update((510.0, 510.0, 690.0, 890.0), 3840, 2160)
            assert crop == first, "Lock should persist when subject is centered"

    def test_min_lock_prevents_early_unlock(self):
        lock = CropLockManager(min_lock_frames=10)
        first = lock.update((500.0, 500.0, 700.0, 900.0), 3840, 2160)

        # Subject at edge but within min_lock_frames — should NOT unlock
        for i in range(9):
            # Push subject bbox to the very edge of the locked crop
            crop = lock.update(
                (float(first[0]), float(first[1]), float(first[2]), float(first[3])),
                3840,
                2160,
            )
            assert crop == first, f"Should not unlock on frame {i+1}"

    def test_unlock_when_near_boundary_after_min_frames(self):
        lock = CropLockManager(min_lock_frames=3, boundary_threshold=0.05)
        first = lock.update((500.0, 500.0, 700.0, 900.0), 3840, 2160)

        # Advance past min_lock_frames with centered bbox
        for _ in range(3):
            lock.update((600.0, 700.0, 650.0, 750.0), 3840, 2160)

        # Now push subject to very edge of locked crop → unlock
        lx1, ly1, lx2, ly2 = first
        edge_bbox = (float(lx1 + 1), float(ly1 + 1), float(lx2 - 1), float(ly2 - 1))
        new_crop = lock.update(edge_bbox, 3840, 2160)
        assert new_crop != first, "Should unlock when subject near boundary"

    def test_reset_clears_lock(self):
        lock = CropLockManager()
        lock.update((100.0, 100.0, 200.0, 200.0), 3840, 2160)
        assert lock.locked_crop is not None
        lock.reset()
        assert lock.locked_crop is None


# ── ROIStabilizer (integration) ────────────────────────────────────────────────


class TestROIStabilizer:
    """Integration tests for the full smoothing + locking pipeline."""

    def test_first_detection_returns_padded_crop(self):
        stab = ROIStabilizer(3840, 2160)
        crop = stab.update(0.0, (500, 500, 700, 900))
        assert crop is not None
        x1, y1, x2, y2 = crop
        # Padded crop should be larger than raw bbox
        assert x1 < 500
        assert y1 < 500
        assert x2 > 700
        assert y2 > 900

    def test_no_detection_no_lock_returns_none(self):
        stab = ROIStabilizer(3840, 2160)
        assert stab.update(0.0, None) is None

    def test_no_detection_with_lock_returns_locked_crop(self):
        stab = ROIStabilizer(3840, 2160)
        crop = stab.update(0.0, (500, 500, 700, 900))
        # Next frame: no detection → keep locked crop
        assert stab.update(1.0, None) == crop

    def test_stable_detections_produce_identical_crop(self):
        """Repeated identical detections should converge to the same locked crop."""
        stab = ROIStabilizer(3840, 2160, min_lock_frames=5)
        bbox = (500, 500, 700, 900)

        crops = []
        for i in range(20):
            crops.append(stab.update(float(i), bbox))

        # After initial lock, all crops should be identical
        unique = set(crops)
        assert len(unique) == 1, "Stable input should keep a single locked crop"

    def test_scene_cut_resets_state(self):
        stab = ROIStabilizer(3840, 2160, scene_cut_threshold=0.30)

        # Establish lock on left side of frame
        for i in range(5):
            stab.update(float(i), (100, 100, 300, 500))

        first_crop = stab._lock.locked_crop

        # Huge jump to right side → scene cut
        new_crop = stab.update(5.0, (3000, 1500, 3500, 2000))

        # After scene cut, should have a new lock (not the old one)
        assert new_crop != first_crop

    def test_crop_within_frame_bounds(self):
        """Crop coordinates must always be within frame bounds."""
        stab = ROIStabilizer(1920, 1080)

        # Subject near edge of frame
        for i, bbox in enumerate([
            (0, 0, 100, 200),
            (1800, 900, 1920, 1080),
            (900, 500, 1000, 600),
        ]):
            crop = stab.update(float(i), bbox)
            if crop:
                x1, y1, x2, y2 = crop
                assert 0 <= x1 < x2 <= 1920
                assert 0 <= y1 < y2 <= 1080

    def test_reset_allows_fresh_start(self):
        stab = ROIStabilizer(3840, 2160)
        stab.update(0.0, (500, 500, 700, 900))
        stab.reset()
        # After reset, no detection → None (no lock)
        assert stab.update(1.0, None) is None
