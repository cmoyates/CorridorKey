---
title: "Alpha Hint ROI + Sparse Tiled Refiner + ROI Benchmark Harness"
type: feat
date: 2026-03-08
---

# Alpha Hint ROI + Sparse Tiled Refiner + ROI Benchmark Harness

## Overview

Expand the ROI pipeline with an alternative bounding-box source (Alpha Hint mask scanning) alongside the existing YOLO detector, add compute-skipping in the CNN refiner via sparse tiling when using the Alpha Hint path, and build a 3-way benchmark harness to compare `none` / `yolo` / `alpha_hint` on time, memory, and pixel-level quality.

## Problem Statement

1. **YOLO detection rate is low on green screen footage** — benchmark showed only 1/5 frames detected. Alpha hints are *always* available (required input), making them a reliable bbox source.
2. **Refiner runs on 100% of tiles** even when large regions are pure green screen with no subject — wasted compute.
3. **No structured benchmark** compares ROI methods against the full-frame baseline with quality metrics (MAE, % pixels > threshold).

## Architecture

```
Frame N (4K RGB + AlphaHint)
    │
    ▼
┌────────────────────────────────────┐
│ --roi-method selector              │
│   "none"  → full-frame             │
│   "yolo"  → SubjectDetector (CPU)  │
│   "alpha_hint" → mask bbox scan    │
└──────────┬─────────────────────────┘
           │
           ▼
┌────────────────────────────────────┐
│ Stabilizer (1-Euro + Lock-Refine) │  ← shared by both yolo & alpha_hint
└──────────┬─────────────────────────┘
           │
           ▼
┌────────────────────────────────────┐
│ Bucket Pad → Engine               │
│ (alpha_hint: sparse refiner tiles)│
└──────────┬─────────────────────────┘
           │
           ▼
┌────────────────────────────────────┐
│ Feathered Reintegration           │
└────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Alpha Hint Bounding Box + CLI Toggle

**Goal:** Build a secondary bbox generator from the alpha hint mask. Wire `--roi-method {yolo,alpha_hint,none}` into both CLIs.

**Files modified:**
- `CorridorKeyModule/roi_manager.py` — add `bbox_from_alpha_hint()` function, update `ROIManager` to accept `roi_method` param
- `clip_manager.py` — replace `--no-roi` with `--roi-method`, update `run_inference()` signature
- `corridorkey_cli.py` — replace `--no-roi` with `--roi-method`
- `tests/test_roi_manager.py` — add tests for alpha hint bbox extraction

#### 1a. `bbox_from_alpha_hint(mask)` function

```python
# CorridorKeyModule/roi_manager.py

def bbox_from_alpha_hint(
    mask: np.ndarray,
    threshold: float = 0.01,
) -> tuple[int, int, int, int] | None:
    """Find bounding box of non-black pixels in an alpha hint mask.

    Args:
        mask: [H, W] or [H, W, 1] float32 mask (0-1).
        threshold: Pixel intensity below this is considered black/empty.

    Returns:
        (x1, y1, x2, y2) or None if mask is entirely black.
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Binary mask of "interesting" pixels
    active = mask > threshold
    if not active.any():
        return None

    # Find min/max coordinates
    rows = np.any(active, axis=1)
    cols = np.any(active, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return (int(x1), int(y1), int(x2 + 1), int(y2 + 1))
```

Key points:
- Threshold 0.01 ignores near-black noise in alpha hints
- Returns exclusive end coords (matching YOLO bbox convention)
- Returns `None` for empty masks → triggers full-frame fallback

#### 1b. Update `ROIManager`

```python
class ROIManager:
    def __init__(
        self,
        *,
        roi_method: str = "yolo",  # "yolo" | "alpha_hint"
        confidence_threshold: float = 0.3,
        # ... existing params
    ) -> None:
        self._roi_method = roi_method
        if roi_method == "yolo":
            self._detector = SubjectDetector(confidence_threshold=confidence_threshold)
        else:
            self._detector = None  # alpha_hint doesn't need YOLO

    def process_with_roi(self, engine, image, mask_linear, **engine_kwargs):
        # Step 1: Get raw bbox
        if self._roi_method == "yolo":
            raw_bbox = self._detector.detect(image)
        else:  # alpha_hint
            raw_bbox = bbox_from_alpha_hint(mask_linear)

        # Steps 2-4: identical to current implementation
        # ...
```

#### 1c. CLI: `--roi-method` replaces `--no-roi`

Both `clip_manager.py` and `corridorkey_cli.py`:

```python
parser.add_argument(
    "--roi-method",
    choices=["none", "yolo", "alpha_hint"],
    default="yolo",
    help="ROI method: yolo (YOLO detection), alpha_hint (mask bbox), none (full frame)",
)
```

- `run_inference()` signature: `roi_enabled=True` → `roi_method="yolo"`
- `roi_method="none"` → skip ROI entirely (same as old `--no-roi`)
- Backward compat: keep `--no-roi` as a hidden alias that sets `roi_method="none"`

#### 1d. Acceptance Criteria

- [x] `bbox_from_alpha_hint()` returns correct bbox for non-empty masks
- [x] Returns `None` for fully-black masks
- [x] 20% padding + lock-and-refine applied identically to YOLO path
- [x] `--roi-method yolo` produces identical output to current `--no-roi=False`
- [x] `--roi-method none` produces identical output to current `--no-roi=True`
- [x] `--roi-method alpha_hint` runs end-to-end without errors
- [x] YOLO model is NOT loaded when `--roi-method alpha_hint` (saves ~50ms startup + RAM)
- [x] Tests cover: empty mask, single pixel, full-frame mask, typical subject mask

---

### Phase 2: Alpha Hint Sparse Tiled Inference (Compute Skipping)

**Goal:** When `--roi-method alpha_hint`, skip CNN Refiner on tiles that contain only empty green screen, using bicubic-upsampled backbone output instead.

**Files modified:**
- `CorridorKeyModule/core/model_transformer.py` — modify `GreenFormer.forward()` to accept optional tile mask
- `CorridorKeyModule/roi_manager.py` — add tile mask generation from alpha hint
- `CorridorKeyModule/inference_engine.py` — thread `tile_skip_mask` through to model
- `tests/test_sparse_refiner.py` — new test file for tiling logic

#### 2a. Generate "Interesting Pixel" Mask

```python
# CorridorKeyModule/roi_manager.py

TILE_DILATION_KERNEL = 65
TILE_DILATION_PADDING = 32
REFINER_TILE_SIZE = 512

def build_refiner_tile_mask(
    alpha_hint: np.ndarray,
    model_input_size: int,
    tile_size: int = REFINER_TILE_SIZE,
) -> torch.Tensor:
    """Build a binary mask indicating which refiner tiles need CNN processing.

    1. Binarize alpha hint (active=1, empty=0)
    2. Dilate to cover refiner's ~65px receptive field
    3. Downsample to tile grid

    Args:
        alpha_hint: [H, W] float32 mask (0-1) at model input resolution.
        model_input_size: Model's processing resolution (e.g. 2048).
        tile_size: Refiner tile size (default 512).

    Returns:
        [grid_h, grid_w] bool tensor — True = run refiner, False = skip.
    """
    # 1. Binarize
    binary = (alpha_hint > 0.01).astype(np.float32)
    hint_tensor = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    # 2. Dilate (max_pool preserves spatial context for refiner RF)
    dilated = F.max_pool2d(
        hint_tensor,
        kernel_size=TILE_DILATION_KERNEL,
        stride=1,
        padding=TILE_DILATION_PADDING,
    )

    # 3. Downsample to tile grid via adaptive avg pool
    grid_size = model_input_size // tile_size
    grid = F.adaptive_avg_pool2d(dilated, (grid_size, grid_size))

    # Any non-zero cell → tile is active
    return (grid.squeeze() > 0)
```

#### 2b. Modify GreenFormer Forward Pass

```python
# In GreenFormer.forward(), after computing coarse logits:

def forward(self, x, tile_skip_mask=None):
    # ... existing encoder + decoder logic ...

    if self.use_refiner and self.refiner is not None:
        if tile_skip_mask is not None:
            # Sparse tiled refinement
            delta_logits = self._sparse_refine(rgb, coarse_pred, tile_skip_mask)
        else:
            # Full refinement (default)
            delta_logits = self.refiner(rgb, coarse_pred)
    else:
        delta_logits = torch.zeros_like(coarse_pred)

    # ... rest unchanged ...


def _sparse_refine(self, rgb, coarse_pred, tile_mask):
    """Run refiner only on tiles marked True in tile_mask."""
    B, C, H, W = coarse_pred.shape
    grid_h, grid_w = tile_mask.shape
    tile_h = H // grid_h
    tile_w = W // grid_w

    # Initialize delta logits to zero (= no refinement for skipped tiles)
    delta_logits = torch.zeros_like(coarse_pred)

    for ty in range(grid_h):
        for tw in range(grid_w):
            if not tile_mask[ty, tw]:
                continue  # Skip — backbone output used directly

            y0 = ty * tile_h
            y1 = y0 + tile_h
            x0 = tw * tile_w
            x1 = x0 + tile_w

            rgb_tile = rgb[:, :, y0:y1, x0:x1]
            coarse_tile = coarse_pred[:, :, y0:y1, x0:x1]
            delta_logits[:, :, y0:y1, x0:x1] = self.refiner(rgb_tile, coarse_tile)

    return delta_logits
```

**Design decisions:**
- Skipped tiles get `delta_logits = 0` → final output = `sigmoid(coarse_logits + 0)` = `sigmoid(coarse_logits)` = backbone-only output with bilinear upsampling (from decoder). No separate bicubic needed — the decoder already bilinear-upsamples from H/4.
- Per-tile refiner call preserves the CNN's receptive field within each tile. The dilation mask ensures border context is covered.
- For 2048 input / 512 tiles → 4×4 = 16 tiles. If subject occupies 25% of frame, ~4 tiles run refiner instead of all 16 → ~4x refiner speedup.

#### 2c. Thread Through Engine

```python
# inference_engine.py process_frame():
# Add tile_skip_mask param, pass to model forward

def process_frame(self, ..., tile_skip_mask=None):
    # ...
    with torch.autocast(...):
        out = self.model(inp_t, tile_skip_mask=tile_skip_mask)
```

ROIManager passes the tile mask when `roi_method == "alpha_hint"`:

```python
# roi_manager.py, in process_with_roi():
if self._roi_method == "alpha_hint":
    tile_mask = build_refiner_tile_mask(mask_padded, bucket)
    engine_kwargs["tile_skip_mask"] = tile_mask
```

#### 2d. Acceptance Criteria

- [x] `build_refiner_tile_mask()` produces correct grid for various input sizes
- [x] Dilation covers full refiner RF (~65px) — verified with test
- [x] Skipped tiles produce identical output to backbone-only (delta_logits=0)
- [x] Active tiles produce identical output to full-refiner path
- [x] `--roi-method yolo` is UNAFFECTED (no tile skipping)
- [x] `--roi-method none` is UNAFFECTED
- [ ] Memory savings measured on tiles with large green screen regions

---

### Phase 3: ROI Benchmarking Harness

**Goal:** Replace existing `benchmarks/bench_roi.py` with a 3-way comparison: `none` vs `yolo` vs `alpha_hint`. Add MAE and "% pixels > 1e-2" metrics.

**Files modified:**
- `benchmarks/bench_roi.py` — rewrite to support 3 methods

#### 3a. Benchmark Flow

```
1. Load frames + masks
2. Warmup (1 frame per method)
3. Run "none" (baseline) → save outputs as ground truth
4. Run "yolo" → compare to ground truth
5. Run "alpha_hint" → compare to ground truth
6. Report: timing, memory, quality per method
```

#### 3b. CLI

```python
parser.add_argument("--clip", required=True)
parser.add_argument("--alpha", required=True)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--device", default=None)
parser.add_argument("--backend", choices=["auto", "torch", "mlx"], default="auto")
parser.add_argument("--max-frames", type=int, default=10)
parser.add_argument(
    "--methods",
    nargs="+",
    choices=["none", "yolo", "alpha_hint"],
    default=["none", "yolo", "alpha_hint"],
    help="Which methods to benchmark (default: all three)",
)
```

#### 3c. Metrics

For each competitor method vs baseline:

| Metric | How | Notes |
|--------|-----|-------|
| **Per-frame time** | `time.perf_counter()` | Exclude frame 0 (warmup) |
| **Peak memory** | `torch.mps.driver_allocated_memory()` / `torch.cuda.max_memory_allocated()` | Reset between methods |
| **MAE** | `np.abs(baseline - competitor).mean()` | Per output channel |
| **% pixels > 1e-2** | `(np.abs(diff) > 1e-2).mean() * 100` | Key quality gate |

#### 3d. Output Format

```
======================================================================
ROI Benchmark — 10 frames, 1920x1080, device=mps
======================================================================

TIMING
  none (baseline): 44.0s median
  yolo           : 19.5s median (2.25x faster)
  alpha_hint     : 18.2s median (2.42x faster)

MEMORY
  none (baseline): peak=31.2 GB
  yolo           : peak=30.1 GB
  alpha_hint     : peak=28.5 GB

QUALITY vs BASELINE (none)
  yolo:
    alpha : MAE=0.0230, >1e-2: 4.2%
    fg    : MAE=0.0150, >1e-2: 2.8%
  alpha_hint:
    alpha : MAE=0.0180, >1e-2: 3.1%
    fg    : MAE=0.0120, >1e-2: 2.2%
======================================================================
```

#### 3e. Ground Truth Storage

Baseline outputs stored in RAM as list of numpy dicts — no .npy files needed for typical 10-frame benchmarks at 1080p (~1.2 GB).

#### 3f. Acceptance Criteria

- [x] Runs all three methods end-to-end
- [x] Baseline always runs first as ground truth
- [x] Timing excludes warmup frame
- [x] Memory reset between methods
- [x] MAE and % pixels > 1e-2 reported per output channel per method
- [x] `--methods` flag allows benchmarking a subset
- [x] Device sync before timing (MPS/CUDA)

---

## Key Design Decisions

1. **`--roi-method` replaces `--no-roi`** — cleaner API, extensible. Hidden `--no-roi` alias kept for backward compat.
2. **Alpha hint bbox uses same stabilizer** — 1-Euro filter + lock-and-refine applies identically. Alpha hint bbox is generally more stable than YOLO, so the filter has less work.
3. **YOLO not loaded for alpha_hint method** — saves startup time and ~100MB RAM.
4. **Sparse tiling only for alpha_hint path** — YOLO path can't build tile masks since it doesn't have per-pixel coverage info. YOLO users get standard ROI crop benefit only.
5. **Tile delta_logits=0 for skipped regions** — mathematically equivalent to backbone-only output. No separate bicubic path needed.
6. **Refiner processes tiles independently** — slight quality loss at tile boundaries possible but mitigated by dilation padding covering the 65px RF.

## Open Questions

1. Tile boundary artifacts? Dilation should cover RF but worth validating visually.
2. Is per-tile refiner call overhead significant for many active tiles? May want batched tiles.
3. Should `alpha_hint` method auto-detect when hint covers >80% of frame and skip ROI entirely?
4. Does tile_skip_mask break `torch.compile`? Dynamic control flow per tile likely causes graph breaks.
5. Keep `--no-roi` forever or deprecate after a release?

## References

- Existing plan: `docs/plans/2026-03-08-feat-dynamic-roi-crop-paste-plan.md`
- ROI Manager: `CorridorKeyModule/roi_manager.py`
- ROI Stabilizer: `CorridorKeyModule/roi_stabilizer.py`
- ROI Detector: `CorridorKeyModule/roi_detector.py`
- Inference Engine: `CorridorKeyModule/inference_engine.py`
- Model Architecture: `CorridorKeyModule/core/model_transformer.py`
- Existing Benchmark: `benchmarks/bench_roi.py`
- CLI Entry Points: `clip_manager.py`, `corridorkey_cli.py`
