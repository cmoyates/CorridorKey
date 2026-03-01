# Phase 8: Unsupported Ops & Fallback Detection Results

**Date:** 2026-03-01
**Device:** MPS (Apple Silicon) | **Size:** 1024x1024 | **PyTorch:** 2.9.1

## Verdict: No action needed — all ops run natively on MPS

## Known Op Support

| Op | Status |
|---|---|
| F.interpolate bilinear | OK |
| F.interpolate bicubic | OK |
| torch.sigmoid | OK |
| torch.cat | OK |
| GroupNorm | OK |
| BatchNorm2d | OK |
| Conv2d 3x3 | OK |
| Conv2d dilated | OK |
| max_pool2d 3x3 | OK |
| max_pool2d 7x7 | OK |
| scaled_dot_product_attention | OK |
| Linear | OK |
| ReLU inplace | OK |

## Runtime Fallback Warnings

**None.** Running with `PYTORCH_ENABLE_MPS_FALLBACK=1` produced zero fallback warnings during full model inference.

## Profiler Top Ops (by CPU time)

| Op | Self CPU % | # Calls | Notes |
|---|---|---|---|
| aten::copy_ | 50.57% | 405 | dtype/device transfers (autocast) |
| aten::_scaled_dot_product_attention_math_for_mps | 21.25% | 24 | Hiera attention — native MPS |
| aten::relu_ | 11.66% | 11 | Activation — native MPS |
| aten::_mps_convolution | 9.70% | 15 | CNN ops — native MPS |
| aten::group_norm | 0.40% | 9 | Refiner norms — native MPS |

## Analysis

All ops in the GreenFormer inference path dispatch natively to MPS Metal shaders. The high `aten::copy_` percentage reflects autocast dtype conversions (fp32→fp16) which are expected and unavoidable under `torch.autocast`.

The SDPA attention runs via the dedicated `_scaled_dot_product_attention_math_for_mps` kernel, confirming efficient Metal dispatch for Hiera's attention blocks.

## Conclusion

No op substitutions needed. The MPS backend in PyTorch 2.9.1 fully supports all ops used by GreenFormer (Hiera backbone + DecoderHead + CNNRefiner).
