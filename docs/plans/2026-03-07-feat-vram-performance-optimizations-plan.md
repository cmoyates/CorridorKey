---
title: "feat: VRAM & Performance Optimizations for Inference Engine"
type: feat
date: 2026-03-07
---

# VRAM & Performance Optimizations for CorridorKey Inference Engine

## Overview

Four-phase optimization plan targeting memory reduction and throughput improvement in the core CorridorKey inference pipeline. **Primary test target: MPS (Apple Silicon)**. CUDA compatibility maintained but not the primary validation environment. Scope is strictly `CorridorKeyModule/` and `clip_manager.py` — GVM and VideoMaMa untouched.

## Problem Statement

Current pain points at 2048x2048 inference:
- **~22.7GB VRAM** required (CUDA), limiting GPU accessibility
- **FP32 weights** consuming double the necessary memory (~400MB wasted)
- **Premature CPU transfers** at `inference_engine.py:175-176` — full 2048x2048 tensors pulled to NumPy before post-processing, creating PCIe bottleneck every frame
- **Per-frame reallocation** of checkerboard backgrounds and morphological kernels
- **Hiera backbone at full 2048x2048** consuming ~8GB VRAM when lower res suffices
- **CNN Refiner 4GB spike** from processing full 2048x2048 in one shot

## Implementation Phases

### Phase 1: "Free" VRAM & Overhead Fixes

Trivial changes, immediate VRAM savings, zero risk to output quality.

#### 1a. FP16 Weight Casting

**File:** `inference_engine.py:29-84` (`_load_model`)

After `model.load_state_dict()` completes (line 78), cast all parameters to FP16:

```python
model = model.half()
```

**Why:** Model loads FP32 weights but only uses `torch.autocast` for activations (line 164). Casting weights to FP16 halves static VRAM footprint (~400MB savings). The existing `autocast` context already handles mixed-precision math.

**Risk:** None — `autocast` already runs FP16 activations; weight casting aligns storage.

#### 1b. Verify No-Grad Context

**File:** `inference_engine.py:86`

- `@torch.no_grad()` decorator already present on `process_frame` — confirmed.
- `model` set to inference mode already at line 36.

**Action:** Verify `clip_manager.py` doesn't call model methods outside this decorator. Current code calls `engine.process_frame()` directly (line 683), which is decorated — confirmed safe.

**No code change needed.**

### Acceptance Criteria — Phase 1

- [ ] `model.half()` added after `load_state_dict` in `_load_model`
- [ ] `@torch.no_grad()` confirmed on `process_frame`
- [ ] Model set to inference mode confirmed in `_load_model`
- [ ] Test script validates memory drop (MPS: `torch.mps.driver_allocated_size()`, CUDA: `torch.cuda.memory_allocated()`)
- [ ] Output quality unchanged (pixel-diff test against reference frame)

---

### Phase 2: Eliminate CPU/GPU Bottlenecks & Cache Assets

#### 2a. Keep Color Math on GPU

**Files:** `inference_engine.py:173-217`, `color_utils.py`

**Current flow (lines 175-176):**
```python
res_alpha = pred_alpha[0].permute(1, 2, 0).float().cpu().numpy()
res_fg = pred_fg[0].permute(1, 2, 0).float().cpu().numpy()
```

All subsequent operations (`clean_matte`, `despill`, `srgb_to_linear`, `premultiply`, `create_checkerboard`, compositing) run on CPU NumPy.

**Proposed flow:**
1. Keep `pred_alpha` and `pred_fg` as GPU tensors after inference
2. Use `F.interpolate` instead of `cv2.resize` for upscale back to original resolution (replacing lines 177-178)
3. Run `despill()`, `srgb_to_linear()`, `premultiply()`, `composite_straight()` as tensor ops on GPU — `color_utils.py` already supports both tensor and numpy via `_is_tensor()` dispatch (lines 11-32)
4. Handle `clean_matte` separately (see below)
5. Call `.cpu().numpy()` only at final dict return (line 219)

**Key concern:** `clean_matte` uses `cv2.connectedComponentsWithStats` (line 265) — CPU-only. Options:
- Keep `clean_matte` as the one CPU roundtrip (transfer alpha only, not FG)
- Or implement pure-torch approximation using `torch.nn.functional.max_pool2d`

**Recommendation:** Keep `clean_matte` on CPU (only touches single-channel alpha), move everything else to GPU.

#### 2b. Cache Static Variables

**Files:** `inference_engine.py`, `color_utils.py`

Currently recreated every frame:
- Checkerboard background (`color_utils.py:298-323`, called at `inference_engine.py:208`)
- Morphological dilation kernel in `clean_matte` (`color_utils.py:277-278`)

**Proposed:**
- Cache checkerboard in `CorridorKeyEngine.__init__` keyed by `(width, height)` — only regenerate if resolution changes
- Cache dilation kernel in `clean_matte` or pass from engine

### Acceptance Criteria — Phase 2

- [ ] `F.interpolate` replaces `cv2.resize` for upscaling predictions
- [ ] `despill`, `srgb_to_linear`, `premultiply`, compositing all run on GPU tensors
- [ ] `.cpu().numpy()` only called at end of `process_frame`
- [ ] Checkerboard cached per resolution
- [ ] Dilation kernel cached
- [ ] Throughput improvement measured (frames/sec before vs after)
- [ ] Output pixel-identical to Phase 1 baseline

---

### Phase 3: Decouple Backbone and Refiner Resolutions

#### 3a. Modify `GreenFormer.forward()`

**File:** `model_transformer.py:238-293`

**Current:** Backbone and refiner both operate at input tensor's resolution (2048x2048).

**Proposed:**
1. Add `backbone_size: int = 1024` parameter to `forward()` (or `__init__`)
2. Before encoding, downsample input with `F.interpolate`:
   ```python
   x_backbone = F.interpolate(x, size=(backbone_size, backbone_size),
                               mode="bilinear", align_corners=False)
   ```
3. Run encoder on `x_backbone` (1024x1024) — Hiera pos_embed interpolation already supported (lines 54-74 in `inference_engine.py`)
4. Upsample coarse decoder outputs back to original input size (2048x2048)
5. Feed original 2048x2048 RGB + upsampled coarse predictions into `CNNRefinerModule`

**VRAM impact:** Backbone at 1024x1024 = ~2GB vs ~8GB at 2048x2048. ~6GB savings.

**Risk:** Coarse predictions lose fine detail — refiner exists to recover this. May need retraining/fine-tuning to compensate.

### Acceptance Criteria — Phase 3

- [ ] `GreenFormer.forward()` accepts `backbone_size` parameter
- [ ] `F.interpolate` downsamples to 1024 before encoder
- [ ] Decoder outputs upsampled to original resolution
- [ ] Refiner receives full-res RGB + upsampled coarse predictions
- [ ] VRAM measured — expect ~6GB reduction
- [ ] Visual quality comparison (side-by-side with Phase 2 output)

---

### Phase 4: Tiled Inference for CNN Refiner

#### 4a. Tile the CNNRefinerModule

**File:** `model_transformer.py:95-138` (`CNNRefinerModule`)

**Current:** Processes full 2048x2048 in one forward pass, spiking ~4GB VRAM.

**Proposed tiling strategy:**
1. Tile size: 512x512 with 64px overlap on each edge
2. Process tiles sequentially on GPU
3. After each tile: move result to CPU NumPy accumulator, flush device cache (CUDA: `torch.cuda.empty_cache()`, MPS: `torch.mps.empty_cache()`)
4. Blend overlapping seams using 2D tent (linear ramp) weight map

**Tent weight map construction:**
```python
# 1D ramp: 0->1 over overlap, 1 in center, 1->0 over overlap
ramp = np.linspace(0, 1, overlap)
weights_1d = np.concatenate([ramp, np.ones(tile_size - 2*overlap), ramp[::-1]])
# 2D: outer product
weights_2d = np.outer(weights_1d, weights_1d)
```

**Tile grid for 2048x2048 with 512 tiles + 64 overlap:**
- Stride = 512 - 64 = 448
- Tiles: ceil(2048 / 448) = 5 per axis = 25 tiles total

**Risk:** Seam artifacts if tent blending isn't precise. Refiner receptive field is ~65px (dilated convs d=1,2,4,8), close to 64px overlap — should suffice but needs visual validation.

### Acceptance Criteria — Phase 4

- [ ] `CNNRefinerModule` tiled with 512x512 tiles, 64px overlap
- [ ] Each tile processed individually, moved to CPU immediately
- [ ] Device cache flushed after each tile (MPS/CUDA-aware)
- [ ] Tent weight map blends seams in CPU accumulator
- [ ] Peak VRAM during refiner pass measured — expect ~500MB vs ~4GB
- [ ] Visual comparison: no seam artifacts at tile boundaries
- [ ] Edge case: non-2048 resolutions handled correctly

---

## Quality Validation Strategy

Every phase must prove it hasn't degraded output quality. We establish a baseline once, then gate each phase against it.

### Baseline Capture (Pre-Phase 1)

Before any changes, run inference on a small set of reference frames and save the raw outputs:

```
tests/fixtures/quality_baseline/
  frame_001_alpha.npy      # Linear alpha [H, W, 1]
  frame_001_fg.npy         # sRGB FG [H, W, 3]
  frame_001_processed.npy  # Linear premul RGBA [H, W, 4]
  frame_001_comp.npy       # sRGB composite [H, W, 3]
```

Use `.npy` (not EXR/PNG) to avoid lossy format round-trips. These are the ground truth.

### Per-Phase Quality Gate

Each phase runs the same reference frames and compares against baseline:

#### Metrics

| Metric | What it catches | Threshold |
|--------|----------------|-----------|
| **Max absolute error** | Catastrophic single-pixel blowups | Phase 1-2: `< 1e-4`, Phase 3-4: `< 0.02` |
| **Mean absolute error (MAE)** | Systematic drift across the image | Phase 1-2: `< 1e-5`, Phase 3-4: `< 0.005` |
| **PSNR** | Overall signal fidelity | Phase 1-2: `> 80 dB` (near-identical), Phase 3-4: `> 40 dB` |
| **SSIM** | Structural/perceptual similarity | Phase 1-2: `> 0.9999`, Phase 3-4: `> 0.95` |

Phase 1-2 thresholds are tight because these changes should be **mathematically lossless** (just FP16 casting and moving ops to GPU — same math, same precision). Phase 3-4 thresholds are relaxed because resolution decoupling and tiling introduce inherent approximation.

#### Per-Channel Validation

Don't just compare the final composite — validate each output channel independently:
- **Alpha channel:** Most sensitive. Crushed shadows or inflated edges are immediately visible in compositing.
- **FG color:** sRGB values. Check R, G, B channels separately — despill regressions show up as green channel drift.
- **Processed RGBA:** Validates the full premultiply + despill + despeckle chain.

#### Difference Visualization

For any frame that fails thresholds, generate a heat map:
```python
diff = np.abs(baseline - result)
# Amplify 10x for visibility, clamp to [0, 1]
diff_vis = np.clip(diff * 10.0, 0.0, 1.0)
cv2.imwrite("diff_alpha.png", (diff_vis * 255).astype(np.uint8))
```

This makes subtle regressions (dark fringes, edge artifacts, tile seams) immediately visible.

### Test Script Structure

```python
# tests/test_quality_gate.py

@pytest.fixture(scope="session")
def baseline_outputs():
    """Load pre-computed baseline .npy files."""
    ...

@pytest.fixture(scope="session")
def current_outputs(engine):
    """Run inference on same reference frames with current code."""
    ...

def test_alpha_pixel_diff(baseline_outputs, current_outputs):
    """Alpha channel max/mean absolute error within threshold."""
    ...

def test_fg_pixel_diff(baseline_outputs, current_outputs):
    """FG color per-channel max/mean absolute error within threshold."""
    ...

def test_processed_rgba_diff(baseline_outputs, current_outputs):
    """Full pipeline RGBA output within threshold."""
    ...

def test_psnr(baseline_outputs, current_outputs):
    """PSNR above minimum dB threshold."""
    ...

def test_ssim(baseline_outputs, current_outputs):
    """SSIM above minimum structural similarity threshold."""
    ...

def test_color_space_integrity(current_outputs):
    """FG values in valid sRGB range [0, 1]. Alpha in linear [0, 1]."""
    ...

def test_no_nan_or_inf(current_outputs):
    """No NaN or Inf in any output channel."""
    ...
```

### Phase-Specific Quality Concerns

| Phase | What could go wrong | How to catch it |
|-------|-------------------|-----------------|
| 1 (FP16) | Precision loss in low-alpha regions, dark fringe artifacts | Alpha channel MAE, check values near 0.0 and 1.0 specifically |
| 2 (GPU math) | Floating-point ordering differences between NumPy and PyTorch | Should be negligible; tight thresholds will catch any drift |
| 3 (Backbone 1024) | Loss of fine edge detail in coarse predictions, blockier alpha | SSIM on alpha edges, visual inspection of hair/fine detail |
| 4 (Tiling) | Seam artifacts at tile boundaries, color discontinuities | Pixel diff specifically at tile boundary locations, SSIM per-tile-overlap-region |

### CI Integration

- Quality gate tests marked `@pytest.mark.gpu` — skip in CI (no GPU), run locally before each phase merge
- Baseline `.npy` files gitignored (too large) — generated locally via `pytest --generate-baseline`
- A lightweight CPU smoke test (tiny synthetic input, check no NaN/Inf/OOB) runs in CI

---

## Technical Considerations

### Performance Targets
| Phase | VRAM Savings (est.) | Throughput Impact |
|-------|-------------------|-------------------|
| 1     | ~400MB            | Neutral           |
| 2     | Minimal           | +20-40% (fewer PCIe transfers) |
| 3     | ~6GB              | ~Neutral (downsample + upsample vs full-res backbone) |
| 4     | ~3.5GB            | -10-20% (tile overhead) |

**Total estimated VRAM: ~22.7GB -> ~12-13GB**

### Critical Color Math Rules
Per CLAUDE.md: FG is sRGB, alpha is linear. All sRGB-to-linear conversions must use piecewise transfer functions from `color_utils.py`. Phase 2 GPU migration must preserve this exactly — the `_is_tensor` dispatch in `color_utils.py` already handles this.

### MPS as Primary Target
All phases must run correctly on MPS (Apple Silicon). Key considerations:
- **Phase 1:** `model.half()` works on MPS. FP16 on MPS uses Apple's Metal shader cores natively. Watch for any precision regressions specific to MPS FP16 path.
- **Phase 2:** `F.interpolate` works on MPS. Tensor ops in `color_utils.py` all MPS-compatible.
- **Phase 3:** `F.interpolate` for backbone downsampling works on MPS.
- **Phase 4:** Cache flushing must be device-aware — use `torch.mps.empty_cache()` on MPS, `torch.cuda.empty_cache()` on CUDA. MPS unified memory means CPU/GPU transfers are cheaper (shared address space), so per-tile offload overhead is lower than on CUDA.
- **Memory measurement:** MPS uses `torch.mps.driver_allocated_size()` or `torch.mps.current_allocated_memory()`. No equivalent to CUDA's `torch.cuda.max_memory_allocated()`.

## Dependencies & Risks

- **Phase 3 risk:** Backbone at 1024 may produce noticeably worse coarse predictions. Refiner may not fully compensate without fine-tuning. Need A/B comparison before committing.
- **Phase 4 risk:** Tent blending seams. 64px overlap matches refiner's ~65px receptive field — tight margin. May need 96px if artifacts appear.
- **No model retraining** assumed for any phase. If Phase 3 degrades quality, retraining with backbone at 1024 would be the fix.

## Open Questions

- Phase 3: retrain refiner at mixed resolution or just test zero-shot?
- Phase 4: 64px overlap sufficient given 65px receptive field, or use 96px?
- Phase 4: cache flush per tile — worth overhead or only every N tiles? (less costly on MPS due to unified memory)
- MPS FP16: any precision issues with Apple Silicon's Metal FP16 path?

## References

- `inference_engine.py` — model loading, frame processing, post-processing
- `color_utils.py` — sRGB/linear conversion, despill, checkerboard, clean_matte
- `model_transformer.py` — GreenFormer, DecoderHead, CNNRefinerModule
- `clip_manager.py:610-738` — per-frame inference loop
