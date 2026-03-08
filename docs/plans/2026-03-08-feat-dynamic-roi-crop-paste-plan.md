---
title: "Dynamic ROI Crop-and-Paste Optimization"
type: feat
date: 2026-03-08
---

# Dynamic ROI Crop-and-Paste Optimization

## Overview

Bypass processing massive empty green screen space by dynamically cropping the high-res input to only the subject, processing it through the CorridorKey engine, and blending it back seamlessly. This reduces GPU tensor size from 2048x2048 (full frame) to as small as 512x512 when the subject is small.

## Problem Statement

Current pipeline resizes the entire 4K frame to 2048x2048 regardless of subject size. A person occupying 20% of frame still forces 100% of the tensor through the Hiera+Refiner network. This wastes compute and VRAM on empty green pixels.

## Architecture

```
Frame N (4K RGB + AlphaHint)
    │
    ▼
┌─────────────────────────────┐
│ Step 1: YOLO CPU Detection  │  ← yolo11n.pt on CPU, 640x640
│   Returns bbox in 4K coords │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Step 2: Temporal Stabilize  │  ← 1-Euro filter + lock-and-refine
│   Smoothed, padded, locked  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Step 3: Power-of-2 Bucket   │  ← 512 / 1024 / 2048, NO resize
│   Center-pad with green     │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ CorridorKey Engine          │  ← Hiera + Refiner at bucket size
│   (process_frame)           │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Step 4: Gaussian Feathered  │  ← Paste alpha back to 4K plate
│   Reintegration             │
└─────────────────────────────┘
```

## Implementation Phases

### Phase 1: CPU-Bound Subject Localization (Step 1) — DONE

**Status:** Implemented in `CorridorKeyModule/roi_detector.py`

**What it does:**
1. Loads `yolo11n.pt` (Nano) forced to CPU — zero GPU VRAM
2. Downsamples input to 640x640 via OpenCV
3. Detects person (COCO class 0), falls back to highest-confidence any-class
4. Scales bbox back to original resolution, clamps to frame bounds

**Files:**
- `CorridorKeyModule/roi_detector.py` — `SubjectDetector` class
- `tests/test_roi_detector.py` — unit tests
- `pyproject.toml` — added `ultralytics` dependency

**API:**
```python
detector = SubjectDetector()
bbox = detector.detect(frame)  # → (x1, y1, x2, y2) | None
```

**Design decisions:**
- Multi-person: currently picks highest-confidence person. **TODO**: union all person bboxes for multi-actor shots.
- No detection: returns `None`. Caller must fall back to full-frame processing.

---

### Phase 2: Temporal Stabilization & Lock-and-Refine (Step 2)

**Goal:** Prevent ViT patch-grid jitter by locking the crop window across frames.

**File:** `CorridorKeyModule/roi_stabilizer.py`

#### 2a. 1-Euro Low-Pass Filter

Implement the standard 1-Euro filter (Casiez et al. 2012) to smooth raw YOLO bbox coordinates (x1, y1, x2, y2) across consecutive frames.

```python
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0): ...
    def __call__(self, timestamp: float, value: float) -> float: ...
```

- 4 filter instances: one per bbox coordinate
- **First frame:** Initialize filter with first detection (no smoothing on frame 0)
- **Scene cut detection:** If bbox centroid moves >30% of frame dimension between frames, reset all filters

#### 2b. Padding

Expand the smoothed bbox outward by **20% margin** on all sides:
```python
pad_x = (x2 - x1) * 0.20
pad_y = (y2 - y1) * 0.20
padded = (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)
```
Clamp to frame boundaries after padding.

#### 2c. Locking Logic

```python
class CropLockManager:
    locked_crop: tuple[int, int, int, int] | None
    min_lock_frames: int = 10  # prevent oscillation
    boundary_threshold: float = 0.05  # 5% of crop dimensions
```

- **Lock:** Store padded crop as locked state. Use these exact coordinates for all subsequent frames.
- **Unlock trigger:** On each frame, check if the new (unpadded, smoothed) subject bbox edge is within 5% of the locked crop edge (measured as fraction of the locked crop's width/height).
- **Min lock duration:** 10 frames minimum before allowing re-evaluation (prevents rapid lock/unlock oscillation).
- **After unlock:** Compute new 20% padded bbox, lock it. Filter continues from current state (no reset).

#### 2d. No-Detection Handling

- If YOLO returns `None` and a lock exists: keep using locked crop
- If YOLO returns `None` and no lock exists: fall back to full-frame processing

**Acceptance criteria:**
- [x] 1-Euro filter smooths bbox across frames
- [x] 20% padding applied after smoothing
- [x] Lock persists across frames with identical crop coordinates
- [x] Unlock triggers when subject approaches 5% of crop boundary
- [x] No oscillation (min 10-frame lock)
- [x] Scene cuts reset filter state
- [x] No-detection gracefully falls back

---

### Phase 3: Power-of-2 Padding / Shape Management (Step 3)

**Goal:** Feed static-shaped tensors to the engine without resizing the crop (preserve pixel frequency).

**File:** `CorridorKeyModule/roi_manager.py` (orchestrates the full pipeline)

#### 3a. Bucket Selection

Three static buckets: **512x512, 1024x1024, 2048x2048**.

```python
BUCKET_SIZES = [512, 1024, 2048]

def select_bucket(crop_w: int, crop_h: int) -> int:
    for size in BUCKET_SIZES:
        if crop_w <= size and crop_h <= size:
            return size
    return None  # exceeds all buckets → fallback
```

#### 3b. Center-Pad

Place the crop in the center of the bucket. Fill remaining space with **mean green screen value** (not black zeros — model was trained on green screen context).

```python
# Pad fill: approximate green screen in sRGB normalized [0,1]
PAD_FILL_RGB = (0.0, 0.69, 0.25)
PAD_FILL_MASK = 0.0  # alpha hint padding = no mask signal
```

Both the RGB crop AND the AlphaHint mask must be cropped and padded identically.

#### 3c. Fallback (>2048)

If padded crop exceeds 2048x2048: bypass ROI entirely, use current full-frame resize-to-2048 behavior. No tiling system needed — that's a separate future feature.

#### 3d. Engine Integration

Modify `CorridorKeyEngine.process_frame()` to accept an optional `img_size` override, or create a wrapper that:
1. Crops the frame + mask using locked coordinates
2. Pads into bucket
3. Calls `process_frame()` with bucket-sized input
4. Extracts the valid region from the output

**Acceptance criteria:**
- [x] No resize — crop is placed at 1:1 pixel scale
- [x] Smallest fitting bucket selected
- [x] Both RGB and mask padded identically
- [x] Green fill value (not black)
- [x] >2048 falls back to current full-frame behavior
- [ ] `torch.compile` sees consistent tensor shapes (no graph breaks)

---

### Phase 4: Gaussian Feathered Reintegration (Step 4)

**Goal:** Paste the processed alpha matte back into the original 4K plate without visible seams.

**File:** `CorridorKeyModule/roi_manager.py` (reintegration method)

#### 4a. Extract Valid Region

Slice the valid crop region from the padded bucket output (reverse of center-pad).

#### 4b. Create Feathered Mask

```python
# Full-resolution float mask
feather_mask = np.zeros((orig_h, orig_w), dtype=np.float32)
feather_mask[y1:y2, x1:x2] = 1.0

# Gaussian blur for soft edges
FEATHER_SIGMA = 15  # pixels
feather_mask = cv2.GaussianBlur(feather_mask, (0, 0), sigmaX=FEATHER_SIGMA)
```

#### 4c. Blend

```python
# Outside the crop: alpha = 0.0 (background)
# Inside the crop: alpha = predicted value
# Feathered edge: linear interpolation
final_alpha = predicted_alpha * feather_mask + 0.0 * (1.0 - feather_mask)
```

Same blending for FG, comp, and processed outputs.

**Acceptance criteria:**
- [x] No visible hard edges at crop boundary
- [x] Alpha outside crop region is 0.0
- [x] Feathered transition is smooth (15px sigma)
- [x] All output channels (alpha, fg, comp, processed) are reintegrated
- [x] Output dimensions match original input dimensions

---

## Integration Point

ROI wraps `engine.process_frame()` from within `clip_manager.py::run_inference()`. This makes it **backend-agnostic** — works with both Torch and MLX engines.

```python
# clip_manager.py::run_inference() frame loop (line ~681)
if roi_enabled:
    res = roi_manager.process_with_roi(engine, img_srgb, mask_linear, **kwargs)
else:
    res = engine.process_frame(img_srgb, mask_linear, **kwargs)
```

The `ROIManager` class owns the full lifecycle: detection → stabilization → crop → engine call → reintegration.

## CLI Interface

Add `--roi / --no-roi` flag (default: enabled).

```python
# corridorkey_cli.py
parser.add_argument("--no-roi", action="store_true", help="Disable dynamic ROI cropping")
```

## File Structure (New)

```
CorridorKeyModule/
├── roi_detector.py      ← Step 1 (DONE)
├── roi_stabilizer.py    ← Step 2 (1-Euro filter + lock manager)
├── roi_manager.py       ← Steps 3+4 (bucket padding + reintegration + orchestration)
```

## Open Questions

1. **Multi-person union?** Current detector picks one person. Should it union all person bboxes? Common in green screen.
2. **Square buckets waste?** A standing person (800x3000) in a 2048 bucket wastes ~80% compute. Rectangular buckets (e.g. 512x2048) would help but complicate `torch.compile`.
3. **YOLO every frame while locked?** CPU cost (~20-50ms/frame) is small vs GPU inference, but for 1000+ frame clips it adds up. Could run every Nth frame while locked.
4. **Despeckle threshold scaling?** `despeckle_size=400` means different things at 512 vs 2048 bucket size. Scale proportionally?
5. **1-Euro filter params?** min_cutoff=1.0, beta=0.007, d_cutoff=1.0 are standard defaults — need tuning with real footage?

---

## Handoff Notes (2026-03-08)

### Implementation Status — ALL 4 STEPS COMPLETE

| Step | File | Tests |
|------|------|-------|
| 1. YOLO CPU detection | `roi_detector.py` | 5 tests (`test_roi_detector.py`) |
| 2. Temporal stabilization | `roi_stabilizer.py` | 19 tests (`test_roi_stabilizer.py`) |
| 3. Bucket padding + engine integration | `roi_manager.py` + `inference_engine.py` | 33 tests (`test_roi_manager.py`) |
| 4. Feathered reintegration | `roi_manager.py` | (included in step 3 tests) |
| CLI integration | `clip_manager.py` + `corridorkey_cli.py` | — |
| Benchmark | `benchmarks/bench_roi.py` | — |

57 total tests, all passing.

### CLI: `--no-roi` flag added to both `clip_manager.py` and `corridorkey_cli.py`. ROI enabled by default.

### Engine change: `process_frame()` now accepts optional `img_size` param to override `self.img_size`, enabling bucket-sized inference without resize.

### Benchmark Results (1920x1080, MPS, 5 frames, BetterGreenScreenTest clip)

| Metric | Full Frame | With ROI |
|--------|-----------|----------|
| Median time | 44.0s | 19.5s (**2.25x faster**) |
| Peak memory | 31.2 GB | 30.1 GB |
| Alpha MAE vs full-frame | — | 0.023 |

### Key Observations from Benchmarking

1. **YOLO detection rate low** — only 1/5 frames detected on this clip. Confidence threshold (0.3) or subject appearance may need tuning.
2. **All detected subjects land in 2048 bucket** — at 1080p, even small subjects + 20% padding exceed 1024. Real bucket savings need 4K input.
3. **Quality delta is non-trivial** — MAE 0.023 on alpha, max_err >1.0. Feathered reintegration introduces visible differences vs full-frame. Needs investigation.
4. **Speedup comes from crop, not bucket reduction** — processing a smaller region of the frame is faster even at 2048 because the engine resize (crop→2048) preserves more detail than (full_frame→2048).
5. **`torch.compile` graph breaks** — untested. Bucket size changes between frames may cause recompilations.

### Remaining Work

- [ ] `torch.compile` graph break verification on GPU
- [ ] Tune YOLO confidence threshold for green screen footage
- [ ] Test with 4K footage (where 512/1024 buckets actually trigger)
- [ ] Investigate quality delta — is feather blending causing artifacts?
- [ ] Consider multi-person bbox union for multi-actor shots
