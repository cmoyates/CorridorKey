# MPS Optimization Plan — CorridorKey

*Date: 2026-03-01*
*Target: Apple Silicon (M-series) via PyTorch MPS backend*

---

## Phase 0 — Repository & Environment Audit

### Environment

| Item | Value |
|---|---|
| macOS | 26.3 |
| Python | 3.12.8 |
| PyTorch | 2.9.1 |
| MPS built | Yes |
| MPS available | Yes |
| Default dtype | float32 |
| Package manager | uv |

### Repo Profile

- **Domain:** Vision / image matting (green screen chroma keying)
- **Model:** GreenFormer — Hiera ViT backbone (timm) + dual DecoderHead + dilated CNN refiner
- **Input:** Single frame `[1, 4, H, W]` (RGB + coarse alpha), default 2048x2048
- **Output:** Alpha matte + foreground color
- **Inference only** — no training loop in this repo
- **Single-frame processing** — no DataLoader in the core path

### Device Selection (already centralized)

- `device_utils.py`: `resolve_device()` — CUDA > MPS > CPU, env var override
- All modules accept `device=` arg, default `"cpu"`
- `clear_device_cache()` handles CUDA only — **MPS gap**

### Current MPS Readiness

| Area | Status | Notes |
|---|---|---|
| Device routing | Done | `device_utils.py` |
| Model `.to(device)` | Done | `inference_engine.py:31` |
| Checkpoint `map_location` | Done | `inference_engine.py:38` |
| autocast | Done | `torch.autocast(device_type=..., dtype=float16)` at line 160 |
| `@torch.no_grad` | Used | Could switch to `inference_mode` |
| `clear_device_cache` | Partial | No MPS branch — `torch.mps.empty_cache()` missing |
| Post-processing | CPU (numpy) | By design — cv2 ops after `.cpu().numpy()` |

### Code Smells Found

1. **`@torch.no_grad` instead of `torch.inference_mode()`** — `inference_engine.py:82`. `inference_mode` is faster, disables version tracking.
2. **`clear_device_cache` ignores MPS** — `device_utils.py:69-73`. Should call `torch.mps.empty_cache()`.
3. **No `torch.mps.synchronize()` anywhere** — timing and memory measurements will be inaccurate without explicit sync.
4. **Post-process is all numpy/cv2 on CPU** — expected for this workload, but the `.cpu().numpy()` transfer at lines 171-172 is a sync point worth measuring.

### No Issues Found

- No hardcoded `.cuda()` calls
- No hardcoded `"cuda"` device strings (outside guarded checks)
- No accidental float64 in the core CorridorKey path
- ImageNet mean/std correctly typed as float32

---

## Phase 1 — Establish Trustworthy Baseline

### Deliverable: `scripts/benchmark_mps.py`

**Hypothesis:** We need sync-aware timing before any optimization work is meaningful.

**Harness requirements:**
- Devices: `cpu`, `mps` (skip `cuda` on Apple Silicon)
- Warmup: 3 iterations (discard)
- Measured iterations: 10+
- Timing: `torch.mps.synchronize()` before start/stop for MPS
- Metrics: median latency, p95 latency, throughput (frames/sec), peak MPS memory
- Configurable: `--device`, `--dtype` (float32/float16), `--img-size`, `--iterations`, `--compile`
- Output: JSON + markdown summary
- Input: synthetic random `[1, 4, img_size, img_size]` tensor (no real images needed for timing)
- Separate model-load time from inference time

**Parity harness:**
- Run same input on CPU float32 (reference) and MPS
- Compare outputs with `torch.allclose(atol=1e-3, rtol=1e-3)`
- Detect NaN/Inf
- Fail loudly on divergence

### Deliverable: `scripts/mps_env_report.py`

Already created. Prints full system/PyTorch/MPS/env-var diagnostics.

---

## Phase 2 — Device-Agnostic Fixes & inference_mode

**Hypothesis:** `inference_mode` is strictly faster than `no_grad` for inference-only paths. MPS cache clearing prevents memory buildup across frames.

### Changes

| Change | File | Risk |
|---|---|---|
| `@torch.no_grad` → `@torch.inference_mode()` | `inference_engine.py:82` | Low — inference only, no autograd needed |
| Add MPS branch to `clear_device_cache` | `device_utils.py:69-73` | None — additive |

### Benchmark Matrix

| Config | Before | After | Keep? |
|---|---|---|---|
| MPS fp16 autocast, inference_mode | baseline | measure | ? |
| MPS fp16 autocast, no_grad (control) | baseline | — | — |

### Accept Criteria

- inference_mode: keep if latency improves or is neutral
- clear_device_cache: keep unconditionally (correctness fix)

---

## Phase 3 — MPS Memory Instrumentation

**Hypothesis:** Memory visibility prevents OOMs and reveals leaks during multi-frame processing.

### Changes

| Change | Notes |
|---|---|
| Add `mps_memory_snapshot()` helper | Returns dict: current_alloc, driver_alloc, recommended_max |
| Log memory after model load | In benchmark harness |
| Log memory after warmup | In benchmark harness |
| Log memory after each measured iteration | Optional verbose mode |
| Detect growing memory across iterations | Flag >10% growth as potential leak |
| Add `torch.mps.empty_cache()` at coarse boundaries | After full frame batch, not per-frame |

### Environment Variable Experiments

Test each independently, document result:

| Env Var | Test | Expected |
|---|---|---|
| `PYTORCH_ENABLE_MPS_FALLBACK=1` | Run with unsupported op detection | Identifies fallback ops |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` | Unlimited caching | May improve throughput |
| `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0` | Aggressive reclaim | May reduce peak memory |
| `PYTORCH_MPS_FAST_MATH=1` | Relaxed precision | May speed up, check parity |
| `PYTORCH_DEBUG_MPS_ALLOCATOR=1` | Allocator tracing | Diagnostic only |

---

## Phase 4 — dtype & Precision Audit (COMPLETE)

**Hypothesis:** The core path already uses float16 autocast correctly. Main risk is accidental float64 from numpy or upstream code.

**Result:** Audit confirms all dtype usage is correct. fp16 autocast remains best default. bf16 supported but marginal parity failure. No float64 accumulation found. See `docs/phase4_results.md` for full results.

### Audit Checklist

| Location | Current | Action |
|---|---|---|
| `inference_engine.py:148` | `.float()` (fp32 input) | Correct — autocast handles downcast |
| `inference_engine.py:160` | `autocast(dtype=float16)` | Correct |
| `inference_engine.py:20-21` | np.float32 mean/std | Correct |
| Post-process (lines 171-174) | `.cpu().numpy()` → fp32 | Correct — CPU-side cv2 |

### Benchmark Matrix

| Config | Notes |
|---|---|
| fp32 no autocast | Control baseline |
| fp16 autocast (current default) | Expected best for MPS |
| bfloat16 autocast | Test if MPS supports it in PyTorch 2.9 |

### Accept Criteria

- Keep current fp16 autocast if parity holds
- Test bfloat16 only if PyTorch 2.9 MPS supports it — revert if NaN/regression

---

## Phase 5 — Memory-Traffic Reductions

**Hypothesis:** Reducing unnecessary sync points and tensor copies in the hot path improves throughput.

### Audit Targets

| Pattern | Location | Action |
|---|---|---|
| `.cpu().numpy()` in hot path | `inference_engine.py:171-172` | Necessary — cv2 post-process. Measure transfer cost. |
| No `.item()` calls in inference | Clean | No action |
| No repeated `.to(device)` | Clean | No action |
| `non_blocking=True` on input transfer | `inference_engine.py:148` | Test — `.to(device, non_blocking=True)` |
| Refiner hook registration per frame | `inference_engine.py:152-158` | Only when `refiner_scale != 1.0` — OK |

### Benchmark

| Change | Expected |
|---|---|
| `non_blocking=True` on `.to(device)` | Small win if MPS command queue isn't empty |
| Pre-allocate output buffer | Minimal — single frame, small tensors |

---

## Phase 6 — Vision-Specific: channels_last

**Hypothesis:** `channels_last` memory format may improve CNN refiner and decoder conv performance on MPS.

### Experiment

1. Convert model to channels_last: `model = model.to(memory_format=torch.channels_last)`
2. Convert input tensor: `inp_t = inp_t.to(memory_format=torch.channels_last)`
3. Verify output correctness
4. Benchmark latency and memory

### Risk

- Hiera ViT backbone may not propagate channels_last cleanly through attention ops
- Mixed format conversions could add overhead
- Test at model default size (2048) and smaller (512, 1024)

### Accept Criteria

- Keep only if >5% latency improvement at 2048x2048
- Revert if parity fails or memory increases significantly

---

## Phase 7 — torch.compile (Strictly Gated)

**Hypothesis:** `torch.compile` on MPS may help the CNN refiner (regular conv pattern) but likely struggles with the Hiera backbone (dynamic shapes, attention).

### Experiment Plan

1. Baseline: eager mode (current)
2. `torch.compile(model.refiner)` only — smallest meaningful unit
3. `torch.compile(model)` full — if refiner works
4. Test with `mode="reduce-overhead"` and `mode="default"`
5. Check for graph breaks, correctness, compilation time

### Risk Matrix

| Target | Risk | Notes |
|---|---|---|
| Full model | High | Hiera + timm + attention = many graph breaks likely |
| Refiner only | Medium | Pure CNN, static shapes — best candidate |
| Decoder heads | Medium | MLP + interpolate — may break on interpolate |

### Accept Criteria

- Keep compiled refiner only if: steady-state latency improves >10%, correctness within atol=1e-3, no NaN
- Reject full-model compile if any graph breaks or regressions

---

## Phase 8 — Unsupported Ops & Fallback Detection

**Hypothesis:** Some ops may silently fall back to CPU via `PYTORCH_ENABLE_MPS_FALLBACK`. Need to identify and substitute.

### Method

1. Run with `PYTORCH_ENABLE_MPS_FALLBACK=1` and capture warnings
2. Profile with `torch.profiler` to identify CPU-dispatched ops
3. Check known MPS-unsupported ops against ops used:
   - `F.interpolate(mode='bilinear')` — should be supported
   - `F.interpolate(mode='bicubic')` — used in pos_embed resize (load-time only)
   - `torch.sigmoid` — supported
   - `torch.cat`, `torch.where` — supported
   - GroupNorm, BatchNorm — check MPS support
   - `max_pool2d` with large kernels — check limits

### Accept Criteria

- Substitute any hot-path op that falls back to CPU
- Leave load-time-only ops (pos_embed resize) on CPU — no perf impact

---

## Phase 9 — Advanced Profiling (Conditional)

Only if earlier phases reveal an unexplained bottleneck.

### Tools

- `torch.profiler.profile()` with `activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]` (if available)
- Operator-level timing breakdown
- Memory snapshots across forward pass

### Targets

- Hiera encoder (likely dominant cost — ViT attention at 2048x2048)
- Bilinear upsampling in decoders
- CNN refiner (4 dilated res blocks at full resolution)

---

## Phase 10 — Strategic Assessment

### Question: Is PyTorch+MPS the right long-term path?

**Factors favoring PyTorch+MPS:**
- Existing codebase is PyTorch
- timm backbone (Hiera) only available in PyTorch
- MPS support in PyTorch 2.9 is mature for standard vision ops

**Factors favoring future MLX exploration:**
- Pure Apple Silicon inference is the primary use case
- MLX has lower overhead for Metal dispatch
- No CUDA compatibility needed for this deployment target

**Recommendation:** Stay on PyTorch+MPS for now. Revisit MLX only if MPS benchmarks reveal fundamental bottlenecks that can't be worked around.

---

## Summary: Priority-Ordered Optimization Targets

| Priority | Phase | Change | Expected Impact | Risk |
|---|---|---|---|---|
| 1 | 2 | `inference_mode()` | +2-5% latency | Very low |
| 2 | 2 | Fix `clear_device_cache` for MPS | Correctness | None |
| 3 | 3 | MPS memory instrumentation | Visibility | None |
| 4 | 6 | channels_last experiment | +5-15% if it works | Medium |
| 5 | 5 | `non_blocking=True` input transfer | +1-3% | Low |
| 6 | 7 | torch.compile on refiner | +10-20% if stable | High |
| 7 | 4 | bfloat16 experiment | Unknown | Medium |
| 8 | 8 | Unsupported op substitution | Varies | Low |

---

## Unresolved Questions

- Does `torch.autocast` with `dtype=float16` on MPS behave identically to CUDA in PyTorch 2.9?
- Does channels_last propagate cleanly through timm's Hiera?
- Is `torch.compile` stable on MPS for CNN-only subgraphs in 2.9?
- Does `PYTORCH_MPS_FAST_MATH` affect matting quality perceptibly?
- What's the actual MPS memory ceiling before OOM at 2048x2048?
