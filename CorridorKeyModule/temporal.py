"""Temporal video coherence: feature caching + optical flow warping.

Caches encoder features on keyframes and warps them to non-keyframes
using optical flow, bypassing the heavy Hiera backbone. Target: 5-20x
throughput on smooth video with keyframe every 5-20 frames.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Adaptive keyframe thresholds
SCENE_CUT_THRESHOLD = 0.15  # ~15% average pixel change
LARGE_MOTION_THRESHOLD = 50.0  # >50px average displacement


def derive_keyframe_interval(num_frames: int, max_interval: int = 10) -> int:
    """Derive keyframe interval from clip length.

    Shorter clips get more frequent keyframes to preserve quality.
    Returns interval clamped to [2, max_interval].

    Examples:
        10 frames  -> interval 2  (5 keyframes)
        30 frames  -> interval 3  (10 keyframes)
        50 frames  -> interval 5  (10 keyframes)
        100+ frames -> interval 10 (10+ keyframes)
    """
    return min(max_interval, max(2, num_frames // 10))


def warp_features(
    cached_features: list[torch.Tensor],
    flow: torch.Tensor,
) -> list[torch.Tensor]:
    """Warp cached encoder features using optical flow (backward warp).

    For each pixel in the target (current) frame, looks up where it came
    from in the source (keyframe). Uses grid_sample with border padding
    to handle out-of-bounds regions.

    Args:
        cached_features: List of [B, C, H_i, W_i] feature tensors at different scales.
        flow: [B, 2, H_flow, W_flow] in pixel units.

    Returns:
        List of warped feature tensors at same scales.
    """
    warped = []
    for feat in cached_features:
        _, _, fh, fw = feat.shape
        # Downsample flow to feature resolution
        flow_ds = F.interpolate(flow, size=(fh, fw), mode="bilinear", align_corners=True)
        # Scale flow values proportionally
        flow_ds = flow_ds.clone()
        flow_ds[:, 0] *= fw / flow.shape[3]
        flow_ds[:, 1] *= fh / flow.shape[2]

        # Build sampling grid: identity + flow displacement
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, fh, device=flow.device),
            torch.linspace(-1, 1, fw, device=flow.device),
            indexing="ij",
        )
        # Add normalized flow displacement
        grid_x = grid_x.unsqueeze(0) + flow_ds[:, 0] * 2.0 / fw
        grid_y = grid_y.unsqueeze(0) + flow_ds[:, 1] * 2.0 / fh
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, fh, fw, 2]

        warped_feat = F.grid_sample(feat, grid, mode="bilinear", align_corners=True, padding_mode="border")
        warped.append(warped_feat)
    return warped


class FlowEstimator:
    """Optical flow estimation using RAFT small (torchvision) with Farneback fallback."""

    def __init__(self, device: torch.device, use_raft: bool = True):
        self.device = device
        self.model = None

        if use_raft:
            try:
                from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

                weights = Raft_Small_Weights.DEFAULT
                self.model = raft_small(weights=weights).to(device)
                self.model.eval()
                self.model.requires_grad_(False)
                logger.info("RAFT small flow estimator loaded")
            except Exception as e:
                logger.warning("RAFT unavailable (%s), using Farneback fallback", e)

        if self.model is None:
            logger.info("Using OpenCV Farneback optical flow (CPU)")

    @torch.no_grad()
    def estimate(self, prev_frame: torch.Tensor, curr_frame: torch.Tensor) -> torch.Tensor:
        """Estimate dense optical flow between two frames.

        Args:
            prev_frame: [B, 3, H, W] float in [0, 1]
            curr_frame: [B, 3, H, W] float in [0, 1]

        Returns:
            flow: [B, 2, H, W] in pixel units
        """
        if self.model is not None:
            return self._estimate_raft(prev_frame, curr_frame)
        return self._estimate_farneback(prev_frame, curr_frame)

    def _estimate_raft(self, prev_frame: torch.Tensor, curr_frame: torch.Tensor) -> torch.Tensor:
        # RAFT expects [B, 3, H, W] in [0, 255]
        prev_255 = prev_frame * 255.0
        curr_255 = curr_frame * 255.0
        flow_predictions = self.model(prev_255, curr_255)
        return flow_predictions[-1]  # Last iteration is most refined

    def _estimate_farneback(self, prev_frame: torch.Tensor, curr_frame: torch.Tensor) -> torch.Tensor:
        import cv2
        import numpy as np

        # Convert to grayscale numpy
        prev_gray = prev_frame[0].mean(dim=0).cpu().numpy()
        curr_gray = curr_frame[0].mean(dim=0).cpu().numpy()

        prev_u8 = (prev_gray * 255).clip(0, 255).astype(np.uint8)
        curr_u8 = (curr_gray * 255).clip(0, 255).astype(np.uint8)

        flow_np = cv2.calcOpticalFlowFarneback(
            prev_u8,
            curr_u8,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # [H, W, 2] -> [1, 2, H, W]
        flow_t = torch.from_numpy(flow_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return flow_t


class TemporalCache:
    """Manages feature caching and keyframe scheduling for temporal coherence.

    Stores encoder features from keyframes and the corresponding RGB frame
    (at backbone resolution) for optical flow estimation. The flow estimator
    is bundled here so GreenFormer.forward() only needs one extra argument.
    """

    def __init__(
        self,
        keyframe_interval: int = 10,
        flow_estimator: FlowEstimator | None = None,
    ):
        self.keyframe_interval = keyframe_interval
        self.flow_estimator = flow_estimator
        self.cached_features: list[torch.Tensor] | None = None
        self.cached_frame: torch.Tensor | None = None  # [B, 3, H, W] RGB at backbone res
        self.frame_index: int = 0

    def is_keyframe(self) -> bool:
        """Check if current frame should be a keyframe."""
        if self.cached_features is None:
            return True  # First frame always keyframe
        return self.frame_index % self.keyframe_interval == 0

    def should_force_keyframe(self, flow: torch.Tensor, prev_frame: torch.Tensor, curr_frame: torch.Tensor) -> bool:
        """Check adaptive triggers for forced keyframe."""
        # Scene cut detection
        frame_diff = (prev_frame - curr_frame).abs().mean()
        if frame_diff > SCENE_CUT_THRESHOLD:
            logger.debug("Forced keyframe: scene cut (diff=%.3f)", frame_diff.item())
            return True

        # Large motion detection
        flow_mag = flow.norm(dim=1).mean()
        if flow_mag > LARGE_MOTION_THRESHOLD:
            logger.debug("Forced keyframe: large motion (mag=%.1f)", flow_mag.item())
            return True

        return False

    def update_cache(self, features: list[torch.Tensor], frame_rgb: torch.Tensor) -> None:
        """Store encoder features and frame for future warping."""
        self.cached_features = [f.detach() for f in features]
        self.cached_frame = frame_rgb.detach()

    def advance(self) -> None:
        """Advance frame counter. Call after each frame is processed."""
        self.frame_index += 1

    def reset(self) -> None:
        """Reset cache for a new clip."""
        self.cached_features = None
        self.cached_frame = None
        self.frame_index = 0
