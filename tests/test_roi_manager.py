"""Tests for the ROI manager (bucket padding, reintegration, orchestration)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from CorridorKeyModule.roi_manager import (
    ALPHA_HINT_THRESHOLD,
    BUCKET_SIZES,
    PAD_FILL_MASK,
    PAD_FILL_RGB,
    ROIManager,
    bbox_from_alpha_hint,
    center_pad_mask,
    center_pad_rgb,
    create_feather_mask,
    extract_valid_region,
    reintegrate,
    select_bucket,
)

# ── Bucket Selection ──────────────────────────────────────────────────────────


class TestBucketSelection:
    def test_small_crop_gets_512(self):
        assert select_bucket(300, 400) == 512

    def test_exact_512_gets_512(self):
        assert select_bucket(512, 512) == 512

    def test_medium_crop_gets_1024(self):
        assert select_bucket(600, 800) == 1024

    def test_large_crop_gets_2048(self):
        assert select_bucket(1200, 1500) == 2048

    def test_exact_2048_gets_2048(self):
        assert select_bucket(2048, 2048) == 2048

    def test_exceeds_all_buckets_returns_none(self):
        assert select_bucket(2049, 2049) is None

    def test_width_exceeds_but_height_fits(self):
        assert select_bucket(2049, 512) is None

    def test_smallest_bucket_selected(self):
        """Should pick the smallest bucket that fits, not a larger one."""
        assert select_bucket(100, 100) == 512
        assert select_bucket(513, 513) == 1024


# ── Center Padding ─────────────────────────────────────────────────────────────


class TestCenterPadRGB:
    def test_output_shape(self):
        crop = np.ones((200, 300, 3), dtype=np.float32)
        padded, offset = center_pad_rgb(crop, 512)
        assert padded.shape == (512, 512, 3)

    def test_offset_centers_crop(self):
        crop = np.ones((200, 300, 3), dtype=np.float32)
        _, (ox, oy) = center_pad_rgb(crop, 512)
        assert ox == (512 - 300) // 2
        assert oy == (512 - 200) // 2

    def test_fill_is_green(self):
        crop = np.zeros((100, 100, 3), dtype=np.float32)
        padded, (ox, oy) = center_pad_rgb(crop, 512)
        # Check a padded pixel (outside the crop)
        corner = padded[0, 0]
        np.testing.assert_array_almost_equal(corner, PAD_FILL_RGB)

    def test_crop_preserved_in_center(self):
        crop = np.full((100, 100, 3), 0.5, dtype=np.float32)
        padded, (ox, oy) = center_pad_rgb(crop, 512)
        center_region = padded[oy : oy + 100, ox : ox + 100]
        np.testing.assert_array_equal(center_region, crop)

    def test_exact_fit_no_padding(self):
        crop = np.ones((512, 512, 3), dtype=np.float32)
        padded, (ox, oy) = center_pad_rgb(crop, 512)
        assert ox == 0 and oy == 0
        np.testing.assert_array_equal(padded, crop)


class TestCenterPadMask:
    def test_output_shape(self):
        mask = np.ones((200, 300), dtype=np.float32)
        padded = center_pad_mask(mask, 512)
        assert padded.shape == (512, 512)

    def test_fill_is_zero(self):
        mask = np.ones((100, 100), dtype=np.float32)
        padded = center_pad_mask(mask, 512)
        assert padded[0, 0] == PAD_FILL_MASK

    def test_handles_3d_mask(self):
        mask = np.ones((100, 100, 1), dtype=np.float32)
        padded = center_pad_mask(mask, 512)
        assert padded.shape == (512, 512)

    def test_rgb_and_mask_identical_placement(self):
        """RGB and mask crops must be placed at the same offset."""
        h, w = 200, 300
        rgb_crop = np.ones((h, w, 3), dtype=np.float32)
        mask_crop = np.ones((h, w), dtype=np.float32)

        _, (rgb_ox, rgb_oy) = center_pad_rgb(rgb_crop, 1024)
        # Mask uses same centering formula
        mask_ox = (1024 - w) // 2
        mask_oy = (1024 - h) // 2

        assert rgb_ox == mask_ox
        assert rgb_oy == mask_oy


# ── Extract Valid Region ───────────────────────────────────────────────────────


class TestExtractValidRegion:
    def test_extracts_correct_region(self):
        padded = np.zeros((512, 512, 3), dtype=np.float32)
        # Place known values at offset
        ox, oy = 106, 156  # (512-300)//2, (512-200)//2
        padded[oy : oy + 200, ox : ox + 300] = 0.75
        extracted = extract_valid_region(padded, 300, 200, (ox, oy))
        assert extracted.shape == (200, 300, 3)
        np.testing.assert_array_almost_equal(extracted, 0.75)

    def test_works_with_2d(self):
        padded = np.zeros((512, 512), dtype=np.float32)
        ox, oy = 100, 100
        padded[oy : oy + 50, ox : ox + 50] = 1.0
        extracted = extract_valid_region(padded, 50, 50, (ox, oy))
        assert extracted.shape == (50, 50)


# ── Feather Mask ───────────────────────────────────────────────────────────────


class TestFeatherMask:
    def test_shape_matches_frame(self):
        mask = create_feather_mask(1080, 1920, (100, 100, 500, 500))
        assert mask.shape == (1080, 1920)

    def test_center_is_one(self):
        mask = create_feather_mask(1080, 1920, (100, 100, 500, 500), sigma=15)
        # Center of crop should be ~1.0 (far from blurred edges)
        assert mask[300, 300] > 0.99

    def test_outside_is_zero(self):
        mask = create_feather_mask(1080, 1920, (200, 200, 400, 400), sigma=15)
        # Far from crop boundary should be ~0
        assert mask[0, 0] < 0.01
        assert mask[1079, 1919] < 0.01

    def test_edge_is_soft(self):
        mask = create_feather_mask(1080, 1920, (200, 200, 800, 800), sigma=15)
        # Right at the crop boundary should be between 0 and 1 (soft edge)
        edge_val = mask[200, 500]  # top edge, middle x
        assert 0.1 < edge_val < 0.9

    def test_zero_sigma_hard_edge(self):
        mask = create_feather_mask(100, 100, (20, 20, 80, 80), sigma=0)
        assert mask[50, 50] == 1.0
        assert mask[0, 0] == 0.0
        # Hard edge: just inside = 1, just outside = 0
        assert mask[20, 50] == 1.0
        assert mask[19, 50] == 0.0


# ── Reintegration ─────────────────────────────────────────────────────────────


class TestReintegration:
    def test_output_matches_frame_dimensions(self):
        crop_result = {
            "alpha": np.ones((100, 200, 1), dtype=np.float32),
            "fg": np.ones((100, 200, 3), dtype=np.float32),
            "comp": np.ones((100, 200, 3), dtype=np.float32),
            "processed": np.ones((100, 200, 4), dtype=np.float32),
        }
        out = reintegrate((1080, 1920), (500, 400, 700, 500), crop_result, sigma=15)
        for key in ("alpha", "fg", "comp", "processed"):
            h, w = out[key].shape[:2]
            assert h == 1080 and w == 1920, f"{key} wrong shape: {out[key].shape}"

    def test_alpha_zero_outside_crop(self):
        crop_result = {
            "alpha": np.ones((100, 200, 1), dtype=np.float32),
        }
        out = reintegrate((1080, 1920), (500, 400, 700, 500), crop_result, sigma=0)
        # Far outside crop → 0
        assert out["alpha"][0, 0, 0] == 0.0
        # Inside crop → 1
        assert out["alpha"][450, 600, 0] == 1.0

    def test_feathered_transition_smooth(self):
        crop_result = {
            "alpha": np.ones((200, 200, 1), dtype=np.float32),
        }
        out = reintegrate((500, 500), (150, 150, 350, 350), crop_result, sigma=15)
        # Inside → high, boundary → intermediate, far outside → ~0
        assert out["alpha"][250, 250, 0] > 0.95
        # At edge
        edge_val = out["alpha"][150, 250, 0]
        assert 0.1 < edge_val < 0.9

    def test_handles_none_values(self):
        crop_result = {"alpha": np.ones((50, 50, 1), dtype=np.float32), "fg": None}
        out = reintegrate((200, 200), (10, 10, 60, 60), crop_result, sigma=0)
        assert out["fg"] is None
        assert out["alpha"] is not None


# ── Alpha Hint Bounding Box ───────────────────────────────────────────────────


class TestBboxFromAlphaHint:
    def test_returns_none_for_empty_mask(self):
        mask = np.zeros((1080, 1920), dtype=np.float32)
        assert bbox_from_alpha_hint(mask) is None

    def test_returns_none_for_below_threshold(self):
        mask = np.full((100, 100), ALPHA_HINT_THRESHOLD * 0.5, dtype=np.float32)
        assert bbox_from_alpha_hint(mask) is None

    def test_single_pixel(self):
        mask = np.zeros((100, 200), dtype=np.float32)
        mask[50, 75] = 1.0
        bbox = bbox_from_alpha_hint(mask)
        assert bbox == (75, 50, 76, 51)

    def test_typical_subject_mask(self):
        mask = np.zeros((1080, 1920), dtype=np.float32)
        mask[200:800, 600:1200] = 0.8
        bbox = bbox_from_alpha_hint(mask)
        assert bbox == (600, 200, 1200, 800)

    def test_full_frame_mask(self):
        mask = np.ones((1080, 1920), dtype=np.float32)
        bbox = bbox_from_alpha_hint(mask)
        assert bbox == (0, 0, 1920, 1080)

    def test_handles_3d_mask(self):
        mask = np.zeros((100, 200, 1), dtype=np.float32)
        mask[10:50, 30:80, 0] = 1.0
        bbox = bbox_from_alpha_hint(mask)
        assert bbox == (30, 10, 80, 50)

    def test_exclusive_end_coords(self):
        """End coords should be exclusive (matching YOLO convention)."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[10:20, 30:40] = 1.0
        bbox = bbox_from_alpha_hint(mask)
        x1, y1, x2, y2 = bbox
        assert x2 - x1 == 10
        assert y2 - y1 == 10

    def test_custom_threshold(self):
        mask = np.full((100, 100), 0.05, dtype=np.float32)
        # Default threshold (0.01) → should detect
        assert bbox_from_alpha_hint(mask) is not None
        # Higher threshold → should not detect
        assert bbox_from_alpha_hint(mask, threshold=0.1) is None


# ── ROIManager Integration ─────────────────────────────────────────────────────


class TestROIManager:
    """Integration tests using a mock engine and mock detector."""

    @staticmethod
    def _make_mock_engine():
        """Create a mock engine whose process_frame returns correctly shaped output."""
        engine = MagicMock()

        def fake_process_frame(image, mask, img_size=None, **kwargs):
            sz = img_size if img_size is not None else 2048
            return {
                "alpha": np.ones((sz, sz, 1), dtype=np.float32) * 0.9,
                "fg": np.ones((sz, sz, 3), dtype=np.float32) * 0.5,
                "comp": np.ones((sz, sz, 3), dtype=np.float32) * 0.3,
                "processed": np.ones((sz, sz, 4), dtype=np.float32) * 0.7,
            }

        engine.process_frame = MagicMock(side_effect=fake_process_frame)
        return engine

    @staticmethod
    def _make_manager_with_mock_detector(roi_method="yolo"):
        """Create an ROIManager with the SubjectDetector mocked out."""
        with patch("CorridorKeyModule.roi_manager.SubjectDetector") as mock_cls:
            mock_detector = MagicMock()
            mock_cls.return_value = mock_detector
            manager = ROIManager(roi_method=roi_method)
        return manager, mock_detector

    def test_full_pipeline_returns_correct_shape(self):
        """ROI pipeline should return full-frame sized output."""
        manager, mock_det = self._make_manager_with_mock_detector()
        mock_det.detect.return_value = (500, 300, 800, 900)
        engine = self._make_mock_engine()

        frame = np.zeros((1080, 1920, 3), dtype=np.float32)
        mask = np.zeros((1080, 1920), dtype=np.float32)

        result = manager.process_with_roi(engine, frame, mask)

        for key in ("alpha", "fg", "comp", "processed"):
            h, w = result[key].shape[:2]
            assert h == 1080 and w == 1920, f"{key}: {result[key].shape}"

    def test_engine_called_with_bucket_img_size(self):
        """Engine should receive img_size matching the bucket."""
        manager, mock_det = self._make_manager_with_mock_detector()
        mock_det.detect.return_value = (800, 400, 1000, 600)
        engine = self._make_mock_engine()

        frame = np.zeros((1080, 1920, 3), dtype=np.float32)
        mask = np.zeros((1080, 1920), dtype=np.float32)

        manager.process_with_roi(engine, frame, mask)

        call_kwargs = engine.process_frame.call_args
        img_size_used = call_kwargs.kwargs.get("img_size") or call_kwargs[1].get("img_size")
        assert img_size_used in BUCKET_SIZES

    def test_no_detection_falls_back_to_full_frame(self):
        """No detection + no lock → full-frame engine call (no img_size override)."""
        manager, mock_det = self._make_manager_with_mock_detector()
        mock_det.detect.return_value = None
        engine = self._make_mock_engine()

        frame = np.zeros((1080, 1920, 3), dtype=np.float32)
        mask = np.zeros((1080, 1920), dtype=np.float32)

        result = manager.process_with_roi(engine, frame, mask)
        assert result is not None

    def test_padded_input_uses_green_fill(self):
        """The padded region of the engine input should be green, not black."""
        manager, mock_det = self._make_manager_with_mock_detector()
        mock_det.detect.return_value = (800, 400, 1000, 600)
        engine = self._make_mock_engine()

        frame = np.zeros((1080, 1920, 3), dtype=np.float32)
        mask = np.zeros((1080, 1920), dtype=np.float32)

        captured_inputs = {}

        def capture_process_frame(image, mask_arg, img_size=None, **kwargs):
            captured_inputs["image"] = image.copy()
            sz = img_size or 2048
            return {
                "alpha": np.ones((sz, sz, 1), dtype=np.float32),
                "fg": np.ones((sz, sz, 3), dtype=np.float32),
                "comp": np.ones((sz, sz, 3), dtype=np.float32),
                "processed": np.ones((sz, sz, 4), dtype=np.float32),
            }

        engine.process_frame = MagicMock(side_effect=capture_process_frame)

        manager.process_with_roi(engine, frame, mask)

        # Check corner pixel (padding region) is green, not black
        img = captured_inputs["image"]
        corner = img[0, 0]
        np.testing.assert_array_almost_equal(corner, PAD_FILL_RGB)

    def test_oversized_crop_falls_back(self):
        """Crop larger than 2048 should fall back to full-frame processing."""
        manager, mock_det = self._make_manager_with_mock_detector()
        mock_det.detect.return_value = (100, 100, 3000, 3000)
        engine = self._make_mock_engine()

        # Very large frame with bbox that exceeds 2048 after padding
        frame = np.zeros((4320, 7680, 3), dtype=np.float32)
        mask = np.zeros((4320, 7680), dtype=np.float32)

        result = manager.process_with_roi(engine, frame, mask)
        # Should still succeed (full-frame fallback)
        assert result is not None

    def test_alpha_hint_method_does_not_load_yolo(self):
        """alpha_hint method should not instantiate SubjectDetector."""
        with patch("CorridorKeyModule.roi_manager.SubjectDetector") as mock_cls:
            ROIManager(roi_method="alpha_hint")
            mock_cls.assert_not_called()

    def test_alpha_hint_returns_correct_shape(self):
        """alpha_hint ROI pipeline should return full-frame sized output."""
        manager, _ = self._make_manager_with_mock_detector(roi_method="alpha_hint")
        engine = self._make_mock_engine()

        frame = np.zeros((1080, 1920, 3), dtype=np.float32)
        mask = np.zeros((1080, 1920), dtype=np.float32)
        # Paint a subject region in the mask
        mask[300:700, 600:1200] = 0.9

        result = manager.process_with_roi(engine, frame, mask)
        for key in ("alpha", "fg", "comp", "processed"):
            h, w = result[key].shape[:2]
            assert h == 1080 and w == 1920, f"{key}: {result[key].shape}"

    def test_alpha_hint_empty_mask_falls_back(self):
        """Empty alpha hint mask → full-frame fallback."""
        manager, _ = self._make_manager_with_mock_detector(roi_method="alpha_hint")
        engine = self._make_mock_engine()

        frame = np.zeros((1080, 1920, 3), dtype=np.float32)
        mask = np.zeros((1080, 1920), dtype=np.float32)

        result = manager.process_with_roi(engine, frame, mask)
        assert result is not None
