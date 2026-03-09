"""Utilities for patching third-party model internals."""

from __future__ import annotations

import logging
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def patch_hiera_global_attention(hiera_model: nn.Module) -> int:
    """Patch global attention blocks to use 4D contiguous tensors for SDPA.

    timm's Hiera passes 5D non-contiguous tensors to PyTorch's SDPA during
    global attention, causing silent fallback to the Math backend (~8GB extra
    VRAM). This replaces those forward methods with a 4D contiguous version
    that enables FlashAttention / memory-efficient kernels.
    """
    patched = 0
    for blk in hiera_model.blocks:
        attn = blk.attn
        if attn.use_mask_unit_attn:
            continue

        def _make_patched_forward(original_attn: nn.Module) -> types.MethodType:
            def _patched_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
                B, N, _ = x.shape
                qkv = self.qkv(x)
                qkv = qkv.reshape(B, N, 3, self.heads, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                if self.q_stride > 1:
                    q = q.view(B, self.heads, self.q_stride, -1, self.head_dim).amax(dim=2)
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()
                x = F.scaled_dot_product_attention(q, k, v)
                x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
                x = self.proj(x)
                return x

            return types.MethodType(_patched_forward, original_attn)

        attn.forward = _make_patched_forward(attn)
        patched += 1

    logger.info("Patched %d Hiera global attention blocks (5D→4D contiguous)", patched)
    return patched
