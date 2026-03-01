---
title: MPS Inference Optimization Audit & Full Pipeline Benchmarking
type: feat
date: 2026-03-01
---

# MPS Inference Optimization Audit & Full Pipeline Benchmarking

## Overview

Comprehensive inference performance audit for CorridorKey's GreenFormer model on Apple Silicon MPS. Extends the existing 10-phase optimization work with full-pipeline benchmarking at production resolution (2048x2048), MPS environment knob testing, preprocessing/postprocessing profiling, and a consolidated optimization report.

## Existing Work (Already Completed)

A 10-phase MPS audit has already been done. **Do not duplicate this work.**

| Phase | Finding | Status |
|-------|---------|--------|
| Device selection | Clean, backend-agnostic (`device_utils.py`) | Done |
| `torch.inference_mode()` | Adopted in `inference_engine.py` | Done |
| `non_blocking=True` | Adopted on `.to(device)` | Done |
| MPS `empty_cache` | Adopted in `device_utils.py` | Done |
| fp16 autocast | Confirmed best dtype, parity within 1e-3 | Done |
| bfloat16 | Rejected — marginal parity failure | Done |
| channels_last | Rejected — Hiera attention conversion overhead | Done |
| torch.compile | Rejected — 4x regression (refiner), crash (full model) | Done |
| MPS fallback detection | All 13 op categories native, zero fallbacks | Done |
| Transfer overhead | Negligible (<0.2% of total) | Done |

**Baseline at 1024x1024**: 0.60s/frame, 1.65 FPS, ~3.3 GB peak memory.
**Bottleneck**: 100% Hiera ViT attention (quadratic in sequence length).

## Problem Statement

Despite thorough model-forward profiling, several gaps remain:

1. **No 2048x2048 benchmark** — production resolution untested
2. **No full-pipeline benchmark** — preprocessing + model + postprocessing not measured separately
3. **MPS fast math untested** — `PYTORCH_MPS_FAST_MATH=1` is an open question in existing report
4. **No preprocessing/postprocessing optimization** — all NumPy/CV2, potentially movable to MPS
5. **No `OPTIMIZATION_REPORT.md`** — existing results are in `docs/mps_optimization_results.md`, not the requested deliverable format
6. **Benchmark script lacks full-pipeline mode** — only measures model-forward

## Proposed Solution

Extend existing infrastructure to close gaps, then consolidate everything into the requested deliverables.

---

## Implementation Phases

### Phase 1: Full-Pipeline Benchmark at 2048x2048

**Files**: `scripts/benchmark_inference.py` (new), extends patterns from `scripts/benchmark_mps.py`

Create a new **full-pipeline** benchmark that measures each stage independently:

```
Stage                    Measurement
─────────────────────────────────────
1. Input generation      time to create synthetic frame + mask (NumPy)
2. Preprocessing         cv2.resize + normalize + concat + tensor creation
3. CPU→MPS transfer      .to(device, non_blocking=True)
4. Model forward         autocast + model(inp_t)
5. MPS→CPU transfer      .cpu().numpy()
6. Postprocessing        cv2.resize + clean_matte + despill + srgb/linear + premul + checkerboard
7. Total pipeline        end-to-end process_frame()
```

Must support:
- `--device mps|cpu|auto`
- `--img-size` (default 2048)
- `--num-warmup` / `--num-iters`
- `--skip-postprocess` (isolate model-only)
- `--output-json <path>`
- `--checkpoint <path>` (required) OR `--random-weights` (skip checkpoint, valid for timing only)
- `--alpha-type gradient|flat|binary` (default `gradient` — radial gradient simulating real matte topology; `flat` = all zeros; `binary` = step function)
- Proper `torch.mps.synchronize()` before all timers
- MPS memory snapshots via `device_utils.memory_snapshot()`

**Per-stage sync strategy:**
| Stage | Sync before timer start? | Sync before timer end? | Notes |
|-------|-------------------------|----------------------|-------|
| 1. Input gen | No | No | Pure CPU |
| 2. Preprocess | No | No | Pure CPU (NumPy/cv2) |
| 3. CPU→MPS transfer | Yes (drain prior MPS work) | Yes (`mps.synchronize()`) | Use `non_blocking=False` for accurate stage timing |
| 4. Model forward | Yes | Yes (`mps.synchronize()`) | The dominant stage |
| 5. MPS→CPU transfer | Yes | No | `.cpu().numpy()` is an implicit sync |
| 6. Postprocess | No | No | Pure CPU |
| 7. Total | Yes (before start) | Yes (after all stages) | End-to-end with `non_blocking=True` (real-world mode) |

**OOM handling:** Catch `RuntimeError` containing "out of memory" or "MPS". Write `{"oom": true, "img_size": 2048, "device": "mps"}` to JSON if `--output-json` specified. Print message suggesting `--img-size 1024`. Exit 0.

**Acceptance criteria:**
- [ ] Runnable with `uv run python scripts/benchmark_inference.py --device mps --random-weights`
- [ ] Also works with `--checkpoint <path>` for real-weights timing + parity
- [ ] Reports per-stage latency breakdown with correct sync
- [ ] Reports total pipeline latency + throughput
- [ ] Reports MPS memory (allocated + driver)
- [ ] Graceful OOM handling at 2048x2048
- [ ] Structured JSON output option
- [ ] Runs both `auto_despeckle=True` and `auto_despeckle=False` and reports separately

### Phase 2: MPS Fast Math Testing

**Files**: `scripts/benchmark_fast_math.py` (new, separate script)

**Critical implementation detail**: `PYTORCH_MPS_FAST_MATH=1` must be set **before** `import torch`. The env var is read during MPS backend initialization, not at runtime. Therefore:
- This **cannot** be a flag on `benchmark_inference.py` (which imports torch at module level)
- Use a **separate script** that sets the env var at line 1, before any imports
- Or use a wrapper: `PYTORCH_MPS_FAST_MATH=1 uv run python scripts/benchmark_inference.py ...`

Test procedure:
1. Run `benchmark_inference.py` with fast math OFF (control) — save raw model outputs
2. Run with `PYTORCH_MPS_FAST_MATH=1` prefix — save raw model outputs
3. Compare: latency delta, memory delta, output parity

**Parity check target**: Compare raw `out["alpha"]` and `out["fg"]` tensors **before** `.cpu().numpy()` postprocessing. Do NOT compare post-`clean_matte` outputs — `clean_matte` hard-thresholds at 0.5 and applies morphology, introducing large deterministic diffs unrelated to fast math precision.

**Quality threshold**: `max_abs_diff > 1e-3` is a soft flag requiring visual inspection, not automatic reject. Document both numeric diff and recommendation. For 8-bit delivery (PNG comp), diffs < 1/255 (~0.004) are invisible.

**Acceptance criteria:**
- [ ] Quantified speed impact of MPS fast math
- [ ] Parity check on raw model outputs (not post-processed)
- [ ] Document whether quality loss is perceptible for matting use case
- [ ] Clear recommendation: adopt / reject / gate behind flag

### Phase 3: Preprocessing/Postprocessing Profiling

**Files**: `scripts/benchmark_inference.py`

Profile the full `process_frame()` pipeline at 2048x2048 to determine actual time breakdown:

| Stage | Current impl | Potential optimization |
|-------|-------------|----------------------|
| cv2.resize (2x: img + mask) | CPU NumPy | Could use torch.nn.functional.interpolate on MPS |
| linear_to_srgb | CPU NumPy | color_utils already supports torch tensors |
| ImageNet normalize | CPU NumPy | Trivial on MPS |
| Concat + transpose | CPU NumPy | torch ops |
| cv2.resize back (Lanczos) | CPU NumPy | torch interpolate (bicubic, not Lanczos) |
| clean_matte | CPU NumPy (connected components) | Must stay CPU (cv2 morphology) |
| despill | CPU NumPy | Could move to torch |
| srgb_to_linear | CPU NumPy | color_utils supports torch |
| premultiply | CPU NumPy | Trivial either way |
| create_checkerboard | CPU NumPy per-frame | Cache or precompute |

**Key question**: At 2048x2048, if model forward dominates (likely >95%), pre/post optimization is low-impact. Measure first, optimize only if >5% of total.

**Acceptance criteria:**
- [ ] Per-stage timing at 2048x2048 production resolution
- [ ] Clear determination: is pre/post >5% of total pipeline?
- [ ] If yes: implement top pre/post optimizations
- [ ] If no: document as future work, skip implementation

### Phase 4: Implement Safe Optimizations (Conditional)

Only implement changes justified by Phase 1-3 measurements.

**Likely candidates** (if pre/post is meaningful):
- Cache checkerboard per resolution instead of regenerating
- Move ImageNet normalize to MPS (eliminate one NumPy allocation)
- Skip `.float()` in postprocess if already float32

**Unlikely to help** (based on existing findings):
- channels_last, torch.compile, bf16 — already rejected
- Transfer optimization — already negligible

**Gate behind flags if numerics-changing:**
- MPS fast math → `--mps-fast-math` CLI flag
- Any precision change → explicit flag

**Acceptance criteria:**
- [ ] Every change backed by benchmark evidence
- [ ] No regressions to model output quality
- [ ] Focused diffs, no broad refactors

### Phase 5: Consolidate into OPTIMIZATION_REPORT.md

**File**: `OPTIMIZATION_REPORT.md` (repo root)

Consolidate all findings (existing 10-phase audit + new pipeline benchmarks) into the requested format:

1. Executive summary
2. Prioritized optimization table (Priority / File / Problem / Fix / Speed impact / Memory impact / Risk / Effort / Implemented?)
3. Benchmark methodology
4. Benchmark results (before/after, per-stage, memory, 2048x2048 findings)
5. Code changes made
6. Remaining opportunities
7. Final recommendation (ship now / ship behind flag / needs more validation)

**Handling existing docs:** Add a deprecation notice at top of `docs/mps_optimization_results.md`:
```
> **Superseded by [OPTIMIZATION_REPORT.md](../OPTIMIZATION_REPORT.md).** This file is a historical artifact from the initial 10-phase audit.
```
Keep as historical reference; do not delete (commit history references it).

**Acceptance criteria:**
- [ ] All existing findings migrated
- [ ] `docs/mps_optimization_results.md` marked as superseded
- [ ] New pipeline benchmark results included
- [ ] MPS fast math results included
- [ ] Clear next-steps for CUDA when hardware available
- [ ] Single source of truth for optimization status

### Phase 6: Final Summary

Concise response with:
- What was found
- What was changed
- Benchmark deltas (model-forward + full pipeline at 2048x2048)
- What still remains

---

## Technical Considerations

- **OOM at 2048x2048**: Hiera at 2048x2048 produces massive attention matrices. May OOM on 16GB/32GB Macs. Document memory requirements clearly.
- **MPS async timing**: All benchmarks must `torch.mps.synchronize()` before reading wall clocks.
- **Existing benchmark preserved**: `scripts/benchmark_mps.py` stays as-is. New script `scripts/benchmark_inference.py` covers full pipeline.
- **color_utils dual-mode**: Already supports both numpy and torch tensors — easy to move pre/post to MPS if warranted.
- **MPS fast math env var**: Must be set before `import torch`. Cannot be toggled at runtime. Requires separate process invocation or dedicated script.
- **Synthetic alpha topology**: `clean_matte` uses `cv2.connectedComponentsWithStats` — cost scales with # of components. Random noise alpha = pathologically slow (millions of tiny components). Use radial gradient (single foreground blob) as default to match real matte topology.
- **CPU baseline at 2048x2048**: Run if feasible (may take minutes per frame). If impractical, document and skip.

## Dependencies & Risks

| Risk | Mitigation |
|------|-----------|
| 2048x2048 OOM on MPS | Catch RuntimeError, write JSON with `"oom": true`, suggest --img-size 1024 |
| MPS fast math quality degradation | Parity check on raw model outputs with atol=1e-3, gate behind env var |
| Pre/post optimization negligible | Measure first; if <5% total, skip and document |
| Fast math env var set too late | Separate script / subprocess invocation, never set after import torch |
| Synthetic alpha misrepresents clean_matte cost | Default to radial gradient; also run with auto_despeckle=False |

## Success Metrics

- Full pipeline benchmark at 2048x2048 with per-stage breakdown
- MPS fast math tested and recommendation documented
- `OPTIMIZATION_REPORT.md` in requested format with all findings
- `scripts/benchmark_inference.py` runnable and documented

## Unresolved Questions

1. **Target latency for 2048x2048?** Without a target, "ship now" vs "needs architectural change" is subjective.
2. **CPU baseline at 2048x2048 required?** May take 5-10 min/frame — run once for the report or skip?
3. **refiner_scale != 1.0 path worth benchmarking?** Forward hook adds overhead; rarely used in production.
4. **input_is_linear=True path worth benchmarking?** Adds linear_to_srgb step; could be MPS-accelerated.

## References

### Internal
- `docs/mps_optimization_results.md` — existing 10-phase summary (to be superseded)
- `docs/mps_optimization_plan.md` — original optimization plan
- `scripts/benchmark_mps.py` — existing model-forward benchmark
- `CorridorKeyModule/inference_engine.py:82-220` — process_frame() hot path
- `CorridorKeyModule/core/color_utils.py` — dual numpy/torch color ops
- `device_utils.py` — device selection + memory snapshot

### Open Questions from Existing Report (addressed by this plan)
- Does `PYTORCH_MPS_FAST_MATH=1` affect matting quality perceptibly? → Phase 2
- What is the actual MPS memory ceiling before OOM at 2048×2048? → Phase 1
