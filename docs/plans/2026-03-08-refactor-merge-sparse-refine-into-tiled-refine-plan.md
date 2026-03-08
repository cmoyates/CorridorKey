---
title: "refactor: Merge _sparse_refine into _tiled_refine"
type: refactor
date: 2026-03-08
---

# refactor: Merge _sparse_refine into _tiled_refine

## Overview

`_sparse_refine` (GPU-only, no blending) causes 8x perf regression on MPS due to memory pressure (30.6 GB / 36 GB). Merge its tile-skipping capability into `_tiled_refine` (CPU offload, tent blending), then delete `_sparse_refine`. One code path for all tiling.

## Problem Statement

- `_sparse_refine` keeps all tiles on GPU — no CPU offloading like `_tiled_refine`
- On 36 GB MPS machine: 30.6 GB peak → paging → 76s/frame vs 10s baseline
- `_sparse_refine` also lacks overlap blending → hard seams at tile boundaries
- Two separate tiling code paths to maintain

## Proposed Solution

Add optional `tile_skip_mask` param to `_tiled_refine`. Before running refiner on each tile, check if the tile overlaps any active cell in the skip mask. If not, skip it (contributing nothing to accumulators — tent normalization naturally produces delta=0 in skipped regions).

### Grid Mapping Strategy

Use **Option C (conservative/any-overlap)**: a tile runs if its footprint overlaps ANY True cell in the 512-aligned skip mask. At most ~20% more tiles than optimal, but safe for quality. Implementation: 2-3 lines of overlap checking.

```
Skip mask grid (512-aligned):  [0-511, 512-1023, 1024-1535, 1536-2047]
Tile starts (stride=416):      [0, 416, 832, 1248, 1536]

Tile at x=416 (covers 416-927) overlaps cells 0 and 1 → run if either is True
```

### Small Image Edge Case

When `input_size <= tile_size` and `tile_skip_mask` is provided:
- All-False mask → return `torch.zeros_like(coarse_pred)` (early exit)
- Otherwise → single-pass `self.refiner(rgb, coarse_pred)` (one tile, skip mask is moot)

## Acceptance Criteria

- [ ] `_tiled_refine` accepts optional `tile_skip_mask` param
- [ ] Tiles not overlapping any True skip-mask cell are skipped (delta=0)
- [ ] `tile_skip_mask=None` → identical behavior to current `_tiled_refine`
- [ ] CPU offloading preserved for all tiles (processed and skipped)
- [ ] `_sparse_refine` method deleted
- [ ] `forward()` dispatch simplified: `tile_skip_mask` passed to `_tiled_refine`
- [ ] `bench_roi.py --shared-engine` alpha_hint ≤ 15s/frame on MPS (was 76s)
- [ ] Peak memory ≤ 27 GB on alpha_hint path (was 30.6 GB)
- [ ] All existing tests pass
- [ ] `test_sparse_refiner.py` integration tests rewritten for new path

## Implementation Steps

### Step 1: Add `tile_skip_mask` to `_tiled_refine`

**File:** `CorridorKeyModule/core/model_transformer.py`

Add `tile_skip_mask: torch.Tensor | None = None` param. Inside the tile loop, before calling `self.refiner()`:

```python
# model_transformer.py:_tiled_refine, inside the tile loop
def _should_skip_tile(self, x: int, y: int, tile_size: int, tile_skip_mask):
    """Check if tile overlaps any True cell in skip mask (conservative)."""
    if tile_skip_mask is None:
        return False
    grid_h, grid_w = tile_skip_mask.shape
    # Map pixel coords to 512-aligned grid cells
    cell_size_h = (tile_size * grid_h) // grid_h  # always 512 for current usage
    cell_size_w = (tile_size * grid_w) // grid_w
    # Actually: the mask grid is img_size // REFINER_TILE_SIZE, so cell size = img_size / grid
    _, _, h, w = ...  # need image dims
    cell_h = h // grid_h
    cell_w = w // grid_w
    cy0 = y // cell_h
    cy1 = min((y + tile_size - 1) // cell_h, grid_h - 1)
    cx0 = x // cell_w
    cx1 = min((x + tile_size - 1) // cell_w, grid_w - 1)
    return not tile_skip_mask[cy0:cy1+1, cx0:cx1+1].any()
```

If `_should_skip_tile` returns True → `continue` (skip tile, no accumulation).

### Step 2: Handle small-image dispatch

**File:** `CorridorKeyModule/core/model_transformer.py`, `forward()` (line 339)

```python
# Replace current 3-way dispatch:
if self.use_refiner and self.refiner is not None:
    if tile_skip_mask is not None and not tile_skip_mask.any():
        # All tiles skipped → backbone-only
        delta_logits = torch.zeros_like(coarse_pred)
    elif self.refiner_tile_size is not None and (
        input_size[0] > self.refiner_tile_size or input_size[1] > self.refiner_tile_size
        or tile_skip_mask is not None
    ):
        delta_logits = self._tiled_refine(rgb, coarse_pred, tile_skip_mask)
    else:
        delta_logits = self.refiner(rgb, coarse_pred)
```

### Step 3: Delete `_sparse_refine`

**File:** `CorridorKeyModule/core/model_transformer.py`

Remove lines 364-403 (`_sparse_refine` method).

### Step 4: Update tests

**File:** `tests/test_sparse_refiner.py`

- Keep `TestBuildRefinerTileMask` (10 tests) — unchanged
- Keep `TestROIManagerTilePassthrough` (2 tests) — unchanged
- Rewrite `TestSparseRefineIntegration` (2 tests) to test `_tiled_refine` with skip mask:
  - All-False mask → output is zeros
  - Partial mask → active tiles have non-zero delta, skipped tiles have ~zero delta

### Step 5: Benchmark verification

Run `bench_roi.py --shared-engine --methods none alpha_hint --max-frames 3` and verify:
- alpha_hint ≤ 15s/frame
- Peak memory ≤ 27 GB

## References

- Brainstorm: `docs/brainstorms/2026-03-08-roi-pipeline-perf-regression-brainstorm.md`
- `_tiled_refine`: `CorridorKeyModule/core/model_transformer.py:261`
- `_sparse_refine`: `CorridorKeyModule/core/model_transformer.py:364`
- `forward()` dispatch: `CorridorKeyModule/core/model_transformer.py:339`
- `build_refiner_tile_mask`: `CorridorKeyModule/roi_manager.py:225`
- Sparse refiner tests: `tests/test_sparse_refiner.py`
- Tiled refiner tests: `tests/test_quality_gate.py:310`
