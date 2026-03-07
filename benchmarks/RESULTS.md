# Benchmark Results

Reference clip: `ClipsForInference/BetterGreenScreenTest_BASE/Input.mp4` (20 frames, 1920x1080 input, 2048x2048 inference)
Alpha hint: `ClipsForInference/BetterGreenScreenTest_BASE/AlphaHint/BetterGreenScreenTest_MASK.mp4`
Settings: sRGB colorspace, all other defaults (despill=1.0, auto_despeckle=True, refiner_scale=1.0)
Hardware: Apple M3, MPS backend

## Results Table

| Phase | Median Frame Time | Mean Frame Time | MPS Mem (post-inference) | Mem Delta | Notes |
|-------|------------------|-----------------|------------------------|-----------|-------|
| 0 -- Baseline | 7.85s | 13.10s* | 32.23 GB | 30.95 GB | *Mean skewed by MPS shader compilation on frames 2-3 (30.8s, 76.4s) |
| 1 -- FP16 weights | 5.70s | 5.66s | 25.02 GB | 23.90 GB | -7.21 GB mem, -27% time vs baseline |
| 2 -- GPU math + caching | 5.42s | 5.51s | 26.10 GB | 24.98 GB | -6.13 GB mem, -31% time vs baseline; +1.08 GB vs P1 (GPU holds post-proc tensors) |
| 3 -- Backbone 1024 | 1.53s | 1.53s | 8.18 GB | 7.06 GB | -80.5% time, -74.6% mem vs baseline; quality lossy w/o retrain |
| 4 -- Tiled refiner | | | | | |

## Phase 0 -- Baseline (unoptimized)

**Date:** 2026-03-07
**Commit:** 995bd25

### Timing

| Metric | Value |
|--------|-------|
| Warmup (frame 1) | 10.61s |
| Mean (excl. warmup) | 13.10s |
| Median (excl. warmup) | 7.85s |
| Stdev | 16.23s |
| Min | 7.17s |
| Max | 76.40s |

Frames 2-3 were extreme outliers (30.8s, 76.4s) due to MPS shader compilation/caching.
Steady-state frames (4-20) ranged 7.2-12.5s with median ~7.8s.

### Memory (MPS unified)

| Metric | Value |
|--------|-------|
| Before inference | 1.28 GB |
| After inference | 32.23 GB |
| Delta | 30.95 GB |

MPS unified memory numbers reflect shared CPU/GPU pool. Useful for relative comparison between phases, not absolute VRAM claims.

### Quality

Baseline vs itself: all pixel diffs are 0. Quality gate tests pass trivially.

## Phase 1 -- FP16 Weight Casting

**Date:** 2026-03-07
**Commit:** 136a18f

### Change

Added `model.half()` after `load_state_dict` in `_load_model`. Autocast already ran FP16 activations; aligning weight storage halves static footprint and reduces activation memory.

### Timing

| Metric | Value |
|--------|-------|
| Warmup (frame 1) | 5.98s |
| Mean (excl. warmup) | 5.66s |
| Median (excl. warmup) | 5.70s |
| Stdev | 0.13s |
| Min | 5.47s |
| Max | 5.85s |

No MPS compilation outliers this run — shader cache likely warm from Phase 0.

### Memory (MPS unified)

| Metric | Value | Delta vs Baseline |
|--------|-------|-------------------|
| Before inference | 1.12 GB | -0.16 GB |
| After inference | 25.02 GB | -7.21 GB (-22.4%) |
| Delta | 23.90 GB | -7.05 GB |

Savings far exceed the estimated ~400MB. FP16 weights produce FP16 intermediates more efficiently under autocast, reducing activation memory too.

### Quality (vs Phase 0 Baseline)

| Channel | MAE | Max Err | PSNR | Pixels > 1e-4 | Pixels > 1e-2 |
|---------|-----|---------|------|---------------|---------------|
| Alpha | 0.000007 | 0.0057 | 83.7 dB | 1.55% | 0.00% |
| FG | 0.000035 | 0.0324 | 78.8 dB | 11.69% | 0.00% |
| Processed | 0.000017 | 0.0057 | 83.4 dB | 5.00% | 0.00% |
| Composite | 0.000029 | 0.0033 | 82.7 dB | 10.90% | 0.00% |

FP16 rounding introduces tiny precision diffs but zero pixels exceed 1e-2. Quality gate tests pass with fp16 thresholds (plan's original "lossless" 1e-4 max_err too tight for FP16).

## Phase 2 -- GPU Color Math + Asset Caching

**Date:** 2026-03-07
**Commit:** 59927d1

### Changes

- `F.interpolate(bicubic)` replaces `cv2.resize(Lanczos4)` for prediction upscaling — stays on GPU
- `despill`, `srgb_to_linear`, `premultiply`, compositing all run as GPU tensor ops
- `.cpu().numpy()` deferred to final return (single batch transfer)
- `clean_matte` stays CPU (cv2.connectedComponents) — only alpha transferred
- Checkerboard + linear variant cached per resolution as GPU tensors
- Dilation kernel cached in module-level dict
- Bicubic clamp added (can overshoot [0,1])

### Timing

| Metric | Value |
|--------|-------|
| Warmup (frame 1) | 5.81s |
| Mean (excl. warmup) | 5.51s |
| Median (excl. warmup) | 5.42s |
| Stdev | 0.16s |
| Min | 5.31s |
| Max | 5.74s |

~5% faster than Phase 1. Modest gain — inference dominates; post-processing was already a small fraction.

### Memory (MPS unified)

| Metric | Value | Delta vs Baseline | Delta vs Phase 1 |
|--------|-------|-------------------|-------------------|
| Before inference | 1.12 GB | -0.16 GB | 0 |
| After inference | 26.10 GB | -6.13 GB (-19.0%) | +1.08 GB |
| Delta | 24.98 GB | -5.97 GB | +1.08 GB |

Memory slightly higher than Phase 1 because GPU now holds post-processing tensors (checkerboard, color math intermediates) that previously lived on CPU numpy. Net still 6.13 GB below baseline.

### Quality (vs Phase 0 Baseline)

| Channel | MAE | Max Err | PSNR | Pixels > 1e-4 | Pixels > 1e-2 |
|---------|-----|---------|------|---------------|---------------|
| Alpha | 0.000041 | 0.0204 | 68.8 dB | 2.77% | 0.01% |
| FG | 0.000125 | 0.0826 | 64.6 dB | 21.29% | 0.06% |
| Processed | 0.000065 | 0.0250 | 70.3 dB | 11.45% | 0.00% |
| Composite | 0.000092 | 0.0185 | 70.7 dB | 20.09% | 0.00% |

Quality diffs larger than Phase 1 due to Lanczos4→bicubic interpolation change (different algorithm, not just floating point ordering). FG channel most affected (3 channels of color data). However, pixels > 1e-2 remains near 0% across all channels — visually indistinguishable.

**Note:** FG max err 0.083 exceeds plan's Phase 1-2 threshold of 0.04. This is cumulative: FP16 rounding (Phase 1) + Lanczos4→bicubic (Phase 2). The threshold assumed Phase 2 would only introduce "floating-point ordering differences" but the interpolation algorithm change is more significant. Practically lossless — 0.06% pixels > 1e-2 in FG, 0% elsewhere.

## Phase 3 -- Backbone 1024

**Date:** 2026-03-07
**Commit:** 50dcd7b

### Changes

- `GreenFormer` accepts `backbone_size` param (default None = same as img_size)
- Encoder initialized at `backbone_size` (1024); pos_embed resized during checkpoint loading
- `forward()` downsamples input to 1024 before encoder, upsamples decoder outputs to full 2048
- Refiner receives original full-res RGB + upsampled coarse predictions
- `CorridorKeyEngine` passes `backbone_size` through to model
- Benchmark script gains `--backbone-size` CLI arg

### Timing

| Metric | Value |
|--------|-------|
| Warmup (frame 1) | 1.62s |
| Mean (excl. warmup) | 1.53s |
| Median (excl. warmup) | 1.53s |
| Stdev | 0.007s |
| Min | 1.52s |
| Max | 1.54s |

5.3x faster than baseline, 3.5x faster than Phase 2. Backbone at 1024 processes 4x fewer tokens.

### Memory (MPS unified)

| Metric | Value | Delta vs Baseline | Delta vs Phase 2 |
|--------|-------|-------------------|-------------------|
| Before inference | 1.12 GB | -0.16 GB | 0 |
| After inference | 8.18 GB | -24.05 GB (-74.6%) | -17.92 GB |
| Delta | 7.06 GB | -23.89 GB | -17.92 GB |

Massive reduction — backbone at 1024 uses ~2GB vs ~8GB at 2048 (4x fewer tokens = 4x less activation memory). Refiner at 2048 still contributes ~4GB.

### Quality (vs Phase 0 Baseline)

| Channel | MAE | Max Err | PSNR | Pixels > 1e-4 | Pixels > 1e-2 |
|---------|-----|---------|------|---------------|---------------|
| Alpha | 0.003336 | 0.9011 | 32.1 dB | 6.22% | 3.18% |
| FG | 0.002658 | 0.7906 | 36.5 dB | 31.23% | 4.57% |
| Processed | 0.001898 | 0.9011 | 37.2 dB | 22.18% | 3.18% |
| Composite | 0.001515 | 0.3544 | 43.3 dB | 29.24% | 3.26% |

Quality is lossy as expected — model was trained at 2048 and backbone now runs at 1024 without retraining. Alpha PSNR 32.1 dB and max err 0.90 exceed plan's lossy thresholds (PSNR > 40, max err < 0.02). The refiner compensates partially but can't fully recover fine detail lost by the coarser backbone.

**Verdict:** Performance gains are exceptional (5.3x speed, 74.6% memory reduction). Quality requires retraining with mixed-resolution backbone to meet production thresholds. Viable as a "fast preview" mode without retraining.
