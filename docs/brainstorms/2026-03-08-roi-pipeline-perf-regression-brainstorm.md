# ROI Pipeline Performance Regression

**Date:** 2026-03-08
**Status:** Ready for planning

## What We're Building

Merge `_sparse_refine` tile-skipping into the existing `_tiled_refine` path, then delete `_sparse_refine`. Single code path that supports both normal tiling and ROI skip masks, with CPU offloading in both cases.

## Why This Approach

### Root Cause Analysis

The ROI `alpha_hint` method is **8x slower** than full-frame baseline (76s vs 10s per frame on MPS).

**Cause chain:**
1. Medium subject (25-50% of 1920x1080) + 20% stabilizer padding exceeds 1024px in one dimension
2. Bucket selection jumps to 2048 (only options: 512/1024/2048) — zero inference savings
3. ROI path passes `tile_skip_mask` to model, triggering `_sparse_refine` instead of `_tiled_refine`
4. `_sparse_refine` keeps all tiles on GPU (no CPU offloading), unlike `_tiled_refine` which offloads each tile
5. Peak memory: 30.6 GB on a 36 GB Mac — MPS memory pressure causes paging/thrash

**Evidence:**
- Shared engine test ruled out MPS recompilation (still 76s with warm engine)
- Peak memory delta: 26.3 GB (none) vs 30.6 GB (alpha_hint) — extra 4.3 GB from no offloading
- 36 GB unified memory - ~5 GB macOS overhead = ~31 GB available, cutting it very close

### Why merge into `_tiled_refine` (not just fix `_sparse_refine`)

- Single code path = less maintenance
- Gets tent-weight blending for free (better quality at tile boundaries)
- CPU offloading already battle-tested in `_tiled_refine`
- `_sparse_refine` has no overlap handling — tile boundary artifacts possible

## Key Decisions

1. **Merge skip-mask into `_tiled_refine`, delete `_sparse_refine`** — one path for all tiling
2. **Skip mask check per overlap tile** — before running refiner on a tile, check if it overlaps any True cell in the skip mask grid; if not, skip (delta=0)
3. **Keep CPU offloading** — critical for MPS memory pressure on 36 GB machines
4. **Better buckets deferred** — separate work item; current fix eliminates the perf regression regardless of bucket size

## Open Questions

- Tile grid alignment: `_tiled_refine` uses overlap-based stride (512-96=416), skip mask uses exact 512 grid. Need mapping logic. Simplest: check if tile center falls in a True skip-mask cell?
- Should skip mask be built at `_tiled_refine`'s stride grid instead of a separate grid? Would eliminate alignment issue.
- Does tent-weight blending on skipped tiles (weight=0) cause normalization artifacts at skip boundaries?
- Future: rectangular/finer buckets would make ROI actually useful for speed. Track as separate item?
