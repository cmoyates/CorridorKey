"""Benchmark: Hiera SDPA 4D contiguous patch vs unpatched (5D) memory usage.

Measures peak memory on MPS/CUDA for a single forward pass at 2048x2048.
"""

from __future__ import annotations

import gc
import time

import torch
import timm

from CorridorKeyModule.model_utils import patch_hiera_global_attention


IMG_SIZE = 2048
IN_CHANNELS = 4
ENCODER_NAME = "hiera_base_plus_224.mae_in1k_ft_in1k"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def reset_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


def get_peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    elif device.type == "mps":
        return torch.mps.current_allocated_memory() / 1024**2
    return 0.0


def run_forward(device: torch.device, *, apply_patch: bool) -> tuple[float, float]:
    """Run a forward pass, return (peak_memory_mb, elapsed_seconds)."""
    reset_memory(device)

    model = timm.create_model(ENCODER_NAME, pretrained=False, features_only=True, img_size=IMG_SIZE)

    # Patch input layer for 4 channels
    patch_embed = model.model.patch_embed.proj
    weight = patch_embed.weight.data
    out_ch, _, k, _ = weight.shape
    new_conv = torch.nn.Conv2d(IN_CHANNELS, out_ch, kernel_size=k, stride=patch_embed.stride, padding=patch_embed.padding)
    new_conv.weight.data[:, :3] = weight
    new_conv.weight.data[:, 3:] = 0.0
    if patch_embed.bias is not None:
        new_conv.bias.data = patch_embed.bias.data
    model.model.patch_embed.proj = new_conv

    if apply_patch:
        count = patch_hiera_global_attention(model.model)
        print(f"  Patched {count} global attention blocks")

    model = model.to(device).half()
    model.requires_grad_(False)

    dummy = torch.randn(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=device, dtype=torch.float16)

    reset_memory(device)

    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model(dummy)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    peak_mb = get_peak_memory_mb(device)

    # Cleanup
    del model, dummy
    reset_memory(device)

    return peak_mb, elapsed


def main() -> None:
    device = get_device()
    print(f"Device: {device}")
    print(f"Resolution: {IMG_SIZE}x{IMG_SIZE}")
    print()

    print("=== UNPATCHED (5D, timm default) ===")
    mem_before, time_before = run_forward(device, apply_patch=False)
    print(f"  Peak memory: {mem_before:.1f} MB")
    print(f"  Forward time: {time_before:.3f}s")
    print()

    print("=== PATCHED (4D contiguous) ===")
    mem_after, time_after = run_forward(device, apply_patch=True)
    print(f"  Peak memory: {mem_after:.1f} MB")
    print(f"  Forward time: {time_after:.3f}s")
    print()

    if mem_before > 0:
        reduction = mem_before - mem_after
        pct = (reduction / mem_before) * 100
        print(f"Memory reduction: {reduction:.1f} MB ({pct:.1f}%)")
    print(f"Speed change: {time_before:.3f}s -> {time_after:.3f}s")


if __name__ == "__main__":
    main()
