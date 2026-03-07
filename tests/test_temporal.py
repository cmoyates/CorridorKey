"""Tests for CorridorKeyModule.temporal — temporal video coherence.

Tests cover TemporalCache keyframe logic, derive_keyframe_interval,
warp_features shapes, adaptive keyframe triggers, and FlowEstimator
Farneback fallback. No GPU or model weights needed.
"""

from __future__ import annotations

import torch

from CorridorKeyModule.temporal import (
    LARGE_MOTION_THRESHOLD,
    SCENE_CUT_THRESHOLD,
    FlowEstimator,
    TemporalCache,
    derive_keyframe_interval,
    warp_features,
)

# ---------------------------------------------------------------------------
# derive_keyframe_interval
# ---------------------------------------------------------------------------


class TestDeriveKeyframeInterval:
    def test_short_clip_gets_small_interval(self):
        assert derive_keyframe_interval(10) == 2

    def test_medium_clip(self):
        assert derive_keyframe_interval(50) == 5

    def test_long_clip_caps_at_max(self):
        assert derive_keyframe_interval(200) == 10

    def test_very_short_clip_floors_at_2(self):
        assert derive_keyframe_interval(5) == 2

    def test_custom_max_interval(self):
        assert derive_keyframe_interval(200, max_interval=5) == 5

    def test_30_frames(self):
        assert derive_keyframe_interval(30) == 3

    def test_100_frames(self):
        assert derive_keyframe_interval(100) == 10


# ---------------------------------------------------------------------------
# TemporalCache
# ---------------------------------------------------------------------------


class TestTemporalCache:
    def test_first_frame_is_always_keyframe(self):
        cache = TemporalCache(keyframe_interval=5)
        assert cache.is_keyframe()

    def test_keyframe_at_interval(self):
        cache = TemporalCache(keyframe_interval=3)
        # Simulate having cached features (non-first-frame)
        cache.cached_features = [torch.zeros(1)]
        cache.frame_index = 0
        assert cache.is_keyframe()
        cache.frame_index = 1
        assert not cache.is_keyframe()
        cache.frame_index = 2
        assert not cache.is_keyframe()
        cache.frame_index = 3
        assert cache.is_keyframe()

    def test_advance_increments_index(self):
        cache = TemporalCache()
        assert cache.frame_index == 0
        cache.advance()
        assert cache.frame_index == 1
        cache.advance()
        assert cache.frame_index == 2

    def test_reset_clears_state(self):
        cache = TemporalCache()
        cache.cached_features = [torch.zeros(1)]
        cache.cached_frame = torch.zeros(1)
        cache.frame_index = 42
        cache.reset()
        assert cache.cached_features is None
        assert cache.cached_frame is None
        assert cache.frame_index == 0

    def test_update_cache_stores_detached(self):
        cache = TemporalCache()
        feat = torch.randn(1, 64, 8, 8, requires_grad=True)
        frame = torch.randn(1, 3, 32, 32, requires_grad=True)
        cache.update_cache([feat], frame)
        assert not cache.cached_features[0].requires_grad
        assert not cache.cached_frame.requires_grad

    def test_no_cache_forces_keyframe(self):
        """Without cached features, any frame is a keyframe."""
        cache = TemporalCache(keyframe_interval=100)
        cache.frame_index = 50
        assert cache.is_keyframe()


# ---------------------------------------------------------------------------
# Adaptive keyframe triggers
# ---------------------------------------------------------------------------


class TestAdaptiveKeyframe:
    def test_scene_cut_triggers_keyframe(self):
        cache = TemporalCache()
        flow = torch.zeros(1, 2, 8, 8)
        # Create frames with large difference
        prev = torch.zeros(1, 3, 8, 8)
        curr = torch.full((1, 3, 8, 8), SCENE_CUT_THRESHOLD + 0.1)
        assert cache.should_force_keyframe(flow, prev, curr)

    def test_small_change_no_trigger(self):
        cache = TemporalCache()
        flow = torch.zeros(1, 2, 8, 8)
        prev = torch.full((1, 3, 8, 8), 0.5)
        curr = torch.full((1, 3, 8, 8), 0.51)
        assert not cache.should_force_keyframe(flow, prev, curr)

    def test_large_motion_triggers_keyframe(self):
        cache = TemporalCache()
        # Flow with large magnitude
        flow = torch.full((1, 2, 8, 8), LARGE_MOTION_THRESHOLD + 10.0)
        prev = torch.zeros(1, 3, 8, 8)
        curr = torch.zeros(1, 3, 8, 8)
        assert cache.should_force_keyframe(flow, prev, curr)

    def test_small_motion_no_trigger(self):
        cache = TemporalCache()
        flow = torch.full((1, 2, 8, 8), 1.0)  # 1px displacement
        prev = torch.zeros(1, 3, 8, 8)
        curr = torch.zeros(1, 3, 8, 8)
        assert not cache.should_force_keyframe(flow, prev, curr)


# ---------------------------------------------------------------------------
# warp_features
# ---------------------------------------------------------------------------


class TestWarpFeatures:
    def test_zero_flow_preserves_features(self):
        """Zero flow should return features close to original."""
        feat = torch.randn(1, 64, 16, 16)
        flow = torch.zeros(1, 2, 32, 32)
        warped = warp_features([feat], flow)
        assert len(warped) == 1
        assert warped[0].shape == feat.shape
        # With zero flow, warped should be very close to original
        # (small numeric differences from grid_sample interpolation)
        torch.testing.assert_close(warped[0], feat, atol=1e-4, rtol=1e-4)

    def test_multi_scale_features(self):
        """Should handle multi-scale feature list."""
        features = [
            torch.randn(1, 112, 32, 32),
            torch.randn(1, 224, 16, 16),
            torch.randn(1, 448, 8, 8),
            torch.randn(1, 896, 4, 4),
        ]
        flow = torch.zeros(1, 2, 64, 64)
        warped = warp_features(features, flow)
        assert len(warped) == 4
        for orig, w in zip(features, warped, strict=True):
            assert w.shape == orig.shape

    def test_nonzero_flow_changes_features(self):
        """Nonzero flow should produce different features."""
        feat = torch.randn(1, 32, 16, 16)
        flow = torch.full((1, 2, 16, 16), 5.0)  # 5px displacement
        warped = warp_features([feat], flow)
        # Features should be different due to warping
        assert not torch.allclose(warped[0], feat, atol=1e-3)


# ---------------------------------------------------------------------------
# FlowEstimator (Farneback fallback only — no RAFT weights in CI)
# ---------------------------------------------------------------------------


class TestFlowEstimatorFarneback:
    def test_farneback_returns_correct_shape(self):
        estimator = FlowEstimator(device=torch.device("cpu"), use_raft=False)
        prev = torch.rand(1, 3, 32, 32)
        curr = torch.rand(1, 3, 32, 32)
        flow = estimator.estimate(prev, curr)
        assert flow.shape == (1, 2, 32, 32)

    def test_identical_frames_near_zero_flow(self):
        estimator = FlowEstimator(device=torch.device("cpu"), use_raft=False)
        frame = torch.rand(1, 3, 32, 32)
        flow = estimator.estimate(frame, frame)
        # Identical frames should produce near-zero flow
        assert flow.abs().mean() < 1.0

    def test_farneback_on_cpu(self):
        """Farneback should always work on CPU without GPU."""
        estimator = FlowEstimator(device=torch.device("cpu"), use_raft=False)
        assert estimator.model is None
