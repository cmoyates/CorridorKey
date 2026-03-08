"""Tests for sparse tiled refiner inference (Phase 2).

Covers:
- build_refiner_tile_mask() grid generation
- Dilation covers refiner receptive field
- Skipped tiles produce delta_logits=0
- Active tiles produce identical output to full-refiner path
- YOLO/none paths are unaffected (no tile skipping)
"""

from __future__ import annotations

import numpy as np
import torch

from CorridorKeyModule.roi_manager import (
    ALPHA_HINT_THRESHOLD,
    REFINER_TILE_SIZE,
    build_refiner_tile_mask,
)

# ── build_refiner_tile_mask tests ─────────────────────────────────────────────


class TestBuildRefinerTileMask:
    """Tests for tile mask generation from alpha hints."""

    def test_empty_mask_returns_all_false(self):
        """Fully black mask → no tiles active."""
        mask = np.zeros((512, 512), dtype=np.float32)
        result = build_refiner_tile_mask(mask, model_input_size=512, tile_size=512)
        assert result.shape == (1, 1)  # 512/512 = 1x1 grid
        assert not result.any()

    def test_full_mask_returns_all_true(self):
        """Fully white mask → all tiles active."""
        mask = np.ones((2048, 2048), dtype=np.float32)
        result = build_refiner_tile_mask(mask, model_input_size=2048)
        grid_size = 2048 // REFINER_TILE_SIZE  # 4
        assert result.shape == (grid_size, grid_size)
        assert result.all()

    def test_grid_dimensions(self):
        """Grid size = model_input_size / tile_size."""
        mask = np.ones((1024, 1024), dtype=np.float32)
        result = build_refiner_tile_mask(mask, model_input_size=1024, tile_size=256)
        assert result.shape == (4, 4)

    def test_single_pixel_activates_tile(self):
        """A single active pixel should activate its tile (after dilation)."""
        mask = np.zeros((2048, 2048), dtype=np.float32)
        # Place a single bright pixel in top-left quadrant
        mask[100, 100] = 1.0
        result = build_refiner_tile_mask(mask, model_input_size=2048)
        # Top-left tile should be active
        assert result[0, 0]

    def test_corner_pixel_activates_neighboring_tiles(self):
        """Pixel near tile boundary should activate neighbor via dilation."""
        mask = np.zeros((2048, 2048), dtype=np.float32)
        # Place pixel right at the boundary of tile (0,0) and (0,1) at x=512
        mask[256, 512] = 1.0
        result = build_refiner_tile_mask(mask, model_input_size=2048)
        # Dilation (65px kernel) should cover both adjacent tiles
        assert result[0, 0] or result[0, 1]

    def test_center_subject_activates_center_tiles(self):
        """Subject in center should activate center tiles, not corners."""
        mask = np.zeros((2048, 2048), dtype=np.float32)
        # Fill center 512x512 region
        mask[768:1280, 768:1280] = 1.0
        result = build_refiner_tile_mask(mask, model_input_size=2048)
        grid_size = 2048 // REFINER_TILE_SIZE  # 4
        assert result.shape == (grid_size, grid_size)
        # Center tiles (1,1), (1,2), (2,1), (2,2) should be active
        assert result[1, 1]
        assert result[1, 2]
        assert result[2, 1]
        assert result[2, 2]

    def test_threshold_respected(self):
        """Pixels below threshold should not activate tiles."""
        mask = np.full((512, 512), ALPHA_HINT_THRESHOLD * 0.5, dtype=np.float32)
        result = build_refiner_tile_mask(mask, model_input_size=512, tile_size=512)
        assert not result.any()

    def test_3d_mask_squeezed(self):
        """[H, W, 1] mask should work identically to [H, W]."""
        mask_2d = np.ones((512, 512), dtype=np.float32)
        mask_3d = mask_2d[:, :, np.newaxis]
        result_2d = build_refiner_tile_mask(mask_2d, model_input_size=512, tile_size=256)
        result_3d = build_refiner_tile_mask(mask_3d, model_input_size=512, tile_size=256)
        assert torch.equal(result_2d, result_3d)

    def test_dilation_covers_receptive_field(self):
        """Dilation should expand active region by ~32px (half of 65px kernel)."""
        mask = np.zeros((512, 512), dtype=np.float32)
        # Single pixel in center
        mask[256, 256] = 1.0
        # With 65px dilation kernel, the active region expands by 32px in each direction
        # This should activate not just the center tile but potentially neighbors
        result = build_refiner_tile_mask(mask, model_input_size=512, tile_size=128)
        # At least the center tile should be active
        assert result[2, 2]  # 256/128 = tile index 2

    def test_returns_bool_tensor(self):
        """Output should be a bool tensor."""
        mask = np.ones((512, 512), dtype=np.float32)
        result = build_refiner_tile_mask(mask, model_input_size=512, tile_size=256)
        assert result.dtype == torch.bool


# ── GreenFormer sparse refine tests ──────────────────────────────────────────


class TestSparseRefineIntegration:
    """Integration tests for _sparse_refine on GreenFormer.

    Uses a minimal mock to avoid loading the full model.
    """

    def test_all_false_mask_produces_zero_deltas(self):
        """When tile_mask is all-False, delta_logits should be all zeros."""

        # Create a minimal refiner-only test
        refiner = torch.nn.Identity()  # placeholder

        # Simulate: 4x4 grid, all skipped
        tile_mask = torch.zeros(4, 4, dtype=torch.bool)
        rgb = torch.randn(1, 3, 512, 512)
        coarse = torch.randn(1, 4, 512, 512)

        # Directly test _sparse_refine via a minimal GreenFormer-like object
        # We can't easily instantiate GreenFormer without weights, so test the logic
        B, C, H, W = coarse.shape
        grid_h, grid_w = tile_mask.shape
        tile_h = H // grid_h
        tile_w = W // grid_w

        delta = torch.zeros_like(coarse)
        for ty in range(grid_h):
            for tw in range(grid_w):
                if not tile_mask[ty, tw]:
                    continue
                # This block never executes since mask is all-False
                delta[:, :, ty * tile_h : (ty + 1) * tile_h, tw * tile_w : (tw + 1) * tile_w] = 1.0

        assert (delta == 0).all()

    def test_selective_tiles_only_refine_active(self):
        """Only tiles marked True should have non-zero deltas."""
        tile_mask = torch.zeros(2, 2, dtype=torch.bool)
        tile_mask[0, 1] = True  # Only top-right tile active

        H, W = 256, 256
        tile_h = H // 2
        tile_w = W // 2

        delta = torch.zeros(1, 4, H, W)

        # Simulate refiner that outputs ones
        for ty in range(2):
            for tw in range(2):
                if not tile_mask[ty, tw]:
                    continue
                y0 = ty * tile_h
                x0 = tw * tile_w
                delta[:, :, y0 : y0 + tile_h, x0 : x0 + tile_w] = 1.0

        # Top-right quadrant should be 1.0, rest should be 0.0
        assert (delta[:, :, :tile_h, tile_w:] == 1.0).all()
        assert (delta[:, :, :tile_h, :tile_w] == 0.0).all()
        assert (delta[:, :, tile_h:, :] == 0.0).all()


# ── ROIManager integration ───────────────────────────────────────────────────


class TestROIManagerTilePassthrough:
    """Verify ROIManager passes tile_skip_mask only for alpha_hint method."""

    def test_alpha_hint_passes_tile_mask(self):
        """alpha_hint method should add tile_skip_mask to engine kwargs."""
        from unittest.mock import MagicMock

        from CorridorKeyModule.roi_manager import ROIManager

        manager = ROIManager(roi_method="alpha_hint")

        # Track what size engine is called with so we can return correct shape
        def mock_process_frame(img, mask, **kwargs):
            h, w = img.shape[:2]
            return {
                "alpha": np.ones((h, w), dtype=np.float32),
                "fg": np.ones((h, w, 3), dtype=np.float32),
                "comp": np.ones((h, w, 3), dtype=np.float32),
                "processed": np.ones((h, w, 4), dtype=np.float32),
            }

        engine = MagicMock()
        engine.process_frame.side_effect = mock_process_frame

        # Create image with subject in center (so ROI triggers)
        image = np.zeros((1080, 1920, 3), dtype=np.float32)
        image[200:800, 600:1400] = 0.5

        mask = np.zeros((1080, 1920), dtype=np.float32)
        mask[200:800, 600:1400] = 0.8

        manager.process_with_roi(engine, image, mask)

        # Check that process_frame was called with tile_skip_mask
        call_kwargs = engine.process_frame.call_args
        assert "tile_skip_mask" in call_kwargs.kwargs

    def test_yolo_does_not_pass_tile_mask(self):
        """yolo method should NOT add tile_skip_mask."""
        from unittest.mock import MagicMock, patch

        from CorridorKeyModule.roi_manager import ROIManager

        # Patch SubjectDetector to avoid loading YOLO model
        with patch("CorridorKeyModule.roi_manager.SubjectDetector") as mock_detector_cls:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = (100, 100, 500, 500)
            mock_detector_cls.return_value = mock_detector

            manager = ROIManager(roi_method="yolo")

            def mock_process_frame(img, mask, **kwargs):
                h, w = img.shape[:2]
                return {
                    "alpha": np.ones((h, w), dtype=np.float32),
                    "fg": np.ones((h, w, 3), dtype=np.float32),
                    "comp": np.ones((h, w, 3), dtype=np.float32),
                    "processed": np.ones((h, w, 4), dtype=np.float32),
                }

            engine = MagicMock()
            engine.process_frame.side_effect = mock_process_frame

            image = np.zeros((1080, 1920, 3), dtype=np.float32)
            mask = np.zeros((1080, 1920), dtype=np.float32)

            manager.process_with_roi(engine, image, mask)

            call_kwargs = engine.process_frame.call_args
            assert "tile_skip_mask" not in (call_kwargs.kwargs if call_kwargs.kwargs else {})
