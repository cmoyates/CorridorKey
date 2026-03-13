# MLX Optimization Log

Tracks all optimization experiments on the `feature/mlx-optimization` branch. Each entry follows the protocol from `HANDOFF_TO_CORRIDORKEY.md`.

---

## Baseline

| Configuration | 37-frame clip @ 1920x1080 |
|---|---|
| PyTorch (MPS) | 3:34 |
| MLX (pre-optimization) | 2:04 |

---

## Completed Optimizations

### 1. Async I/O Pipeline (Issue #2)

**Commit**: `d98ad12`
**Hypothesis**: Overlap frame read/write with inference using threads. Read (~200ms) and write (~800ms) can run while GPU does inference (~1300ms).
**Implementation**: 3-thread pipeline — reader thread decodes frames into `Queue(maxsize=2)`, main thread runs inference, writer thread writes outputs. `threading.Event` for error propagation.
**Files changed**: `clip_manager.py` (frame loop rewrite), `CorridorKeyModule/backend.py` (timing exposure)
**Fidelity**: Tier 3 (byte-identical outputs by construction)
**Test result**: 195/195 tests pass
**Expected speedup**: ~30-40% (read + write fully overlap with inference)
**Status**: Merged. Awaiting real-pipeline benchmark.

### 2. Checkerboard Cache (Issue #4)

**Commit**: `d98ad12`
**Hypothesis**: `create_checkerboard(w, h)` allocates identical array every frame. Cache by `(w, h, checker_size, color1, color2)` key.
**Implementation**: Module-level `_checkerboard_cache` dict in `backend.py`, `_get_checkerboard()` wrapper.
**Files changed**: `CorridorKeyModule/backend.py`
**Fidelity**: Tier 3 (identical output)
**Test result**: 195/195 tests pass
**Expected speedup**: Small per-frame savings (~1-2ms), but eliminates unnecessary allocation pressure.
**Status**: Merged.

### 3. Postprocess Timing Fix (Issue #9)

**Commit**: `d98ad12`
**Hypothesis**: `phase_times["postprocess"]` was declared but never populated, hiding postprocess cost inside "infer" timing.
**Implementation**: Adapter returns `_timing` dict with `mlx_inference` and `postprocess` keys. Writer thread extracts and appends to `phase_times["postprocess"]`.
**Files changed**: `CorridorKeyModule/backend.py`, `clip_manager.py`
**Fidelity**: N/A (instrumentation only)
**Status**: Merged.

---

## Remaining Issues

| # | Title | Priority | Status |
|---|---|---|---|
| 3 | Configurable output selection | P2 | Open |
| 5 | Skip postprocessing when outputs disabled | P3b | Open (depends on #3) |
| 6 | sRGB/linear conversions to MLX GPU | P3c | Open |
| 7 | Larger tile sizes (768/1024px) | P4 | Open |
| 8 | PyAV VideoToolbox hw decode | P5 | Open |
| 10 | Tile overlap 64 vs 128 discrepancy | Investigation | Open |
| 11 | mx.async_eval benefit when GPU saturated | Investigation | Open |
| 12 | PyAV min version for hwaccel | Investigation | Open |
| 13 | MLX thread safety with Metal | Investigation | Open |

---

## Dead Ends (from prior research)

See `HANDOFF_TO_CORRIDORKEY.md` Section 5 for full list. Key items:
- Temporal blending/caching: edge artifacts at all blend ratios
- Int8 quantization: 11% slower on Apple Silicon
- Backbone resolution decoupling: edge degradation at even 12% downscale
- GPU stream parallelism: single GPU on Apple Silicon
