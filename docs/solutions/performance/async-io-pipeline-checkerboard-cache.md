---
title: "Async I/O pipeline with checkerboard cache and postprocess timing"
category: performance
tags:
  - threading
  - async-io
  - gpu-utilization
  - caching
  - instrumentation
component: clip_manager, backend
symptoms:
  - GPU idle ~1s per frame during sequential I/O
  - Identical checkerboard array allocated every frame
  - postprocess phase timing silently missing from benchmarks
root_cause: >
  Sequential read-infer-write loop blocked GPU during disk I/O;
  no memoization on deterministic checkerboard generation;
  phase_times["postprocess"] declared but never written to
resolved_by: d98ad12
issues: [2, 4, 9]
date: 2026-03-13
---

# Async I/O Pipeline + Checkerboard Cache + Postprocess Timing

## Problem

`clip_manager.py::run_inference()` processed frames strictly sequentially:

```
read frame → infer (~1.3s) → write 4 files (~800ms) → repeat
```

The GPU sat idle during all I/O (~1s/frame wasted). Additionally:
- `create_checkerboard(w, h)` allocated an identical numpy array every frame
- `phase_times["postprocess"]` was declared but never populated, hiding postprocessing cost in benchmarks

## Root Cause

The frame loop was written as a simple `for` loop with blocking calls at each step. Python's GIL is released during cv2 I/O and MLX Metal compute, so threading can provide genuine overlap — but no threading was used. The checkerboard is deterministic (depends only on resolution + colors) but was never cached. The adapter-level postprocess timing was logged at DEBUG but never surfaced to the pipeline summary.

## Solution

### 1. Async I/O Pipeline

3-thread architecture with bounded queues:

```
[Reader Thread] --read_q(2)--> [Main Thread / Infer] --write_q(2)--> [Writer Thread]
```

- **Reader** (`_reader_worker`): owns video captures, decodes input + alpha, enqueues `(i, stem, img, mask, t_read)`
- **Main thread**: dequeues, runs `engine.process_frame()`, enqueues results
- **Writer** (`_writer_worker`): dequeues, writes FG/Matte/Comp/Processed, fires `on_frame_complete`

Key patterns:
- `Queue(maxsize=2)` for backpressure (prevents OOM)
- `threading.Event` for cross-thread error propagation
- Sentinel `None` for end-of-stream
- No locks needed: engine is stateless, each key appended by one thread, `list.append` is GIL-atomic

### 2. Checkerboard Cache

```python
_checkerboard_cache: dict[tuple, np.ndarray] = {}

def _get_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55):
    key = (w, h, checker_size, color1, color2)
    if key not in _checkerboard_cache:
        _checkerboard_cache[key] = cu.create_checkerboard(...)
    return _checkerboard_cache[key]
```

### 3. Postprocess Timing

Adapter attaches timing to result dict:
```python
result["_timing"] = {"mlx_inference": t_mlx, "postprocess": t_post}
```

Writer thread pops and records:
```python
timing = res.pop("_timing", None)
if timing and "postprocess" in timing:
    phase_times["postprocess"].append(timing["postprocess"])
```

## Files Changed

| File | Change |
|------|--------|
| `clip_manager.py` | Frame loop rewrite, `_reader_worker()`, `_writer_worker()` |
| `CorridorKeyModule/backend.py` | `_checkerboard_cache`, `_get_checkerboard()`, `_timing` dict |

## Verification

- 195/195 tests pass (no new tests needed — public API unchanged)
- Fidelity: Tier 3 (byte-identical outputs by construction)
- Expected speedup: ~30-40% (read + write fully overlap with inference)

## Prevention

1. **Profile I/O vs compute separately from the start.** Add per-phase timing in initial implementation.
2. **Default to async I/O for any frame loop touching GPU + disk.** Producer-consumer is the baseline pattern.
3. **Cache any allocation that depends only on frame dimensions.** Hoist or memoize resolution-keyed buffers.
4. **Instrument all pipeline phases at build time.** Invisible phases silently accumulate cost.
5. **Use bounded queues by default.** Always set `maxsize` to prevent latent OOM.

## Related

- [HANDOFF_TO_CORRIDORKEY.md](../../HANDOFF_TO_CORRIDORKEY.md) — Priority 1 optimization target
- [MLX_OPTIMIZATION_LOG.md](../../MLX_OPTIMIZATION_LOG.md) — Full experiment log
- Issues #3, #5 depend on this work (configurable outputs, conditional postprocess skip)
