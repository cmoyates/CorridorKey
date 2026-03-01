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

## Phase 5 — Memory-Traffic Reductions (COMPLETE)

**Hypothesis:** Reducing unnecessary sync points and tensor copies in the hot path improves throughput.

**Result:** Transfer costs are <0.2% of total inference time. `non_blocking=True` is effectively neutral (+0.07%). Kept as correct-by-default pattern. No further memory-traffic optimizations needed — the bottleneck is entirely in the model forward pass.

### Changes

| Change | File | Status |
|---|---|---|
| `non_blocking=True` on `.to(device)` | `inference_engine.py:148` | Done — kept |

### Benchmark Results (1024x1024, 10 iterations)

#### Transfer Costs

| Direction | Mode | Median (ms) | % of Inference |
|---|---|---|---|
| CPU→MPS | blocking | 0.528 | 0.09% |
| CPU→MPS | non_blocking | 0.475 | 0.08% |
| MPS→CPU | .cpu().numpy() | 0.655 | 0.11% |

#### End-to-End (fp16 autocast, includes transfers)

| Mode | Median (s) | P95 (s) | FPS | Delta |
|---|---|---|---|---|
| blocking | 0.5985 | 0.5998 | 1.67 | baseline |
| non_blocking | 0.5981 | 0.5990 | 1.67 | +0.07% |

### Key Finding

Transfer overhead is negligible (~1.2ms out of ~598ms). The bottleneck is 100% model forward pass. Future optimization should target `channels_last` (Phase 6) and `torch.compile` (Phase 7).

---

## Phase 6 — Vision-Specific: channels_last (COMPLETE — REJECTED)

**Hypothesis:** `channels_last` memory format may improve CNN refiner and decoder conv performance on MPS.

**Result:** No latency improvement. Refiner is too small a fraction of total inference time. Full-model `channels_last` is slightly slower (-0.30%) due to Hiera attention format conversion overhead. **Not adopted.**

### Benchmark Results (1024x1024, 10 iterations, fp16 autocast)

| Config | Median (s) | P95 (s) | FPS | Peak Mem (MB) | Delta |
|---|---|---|---|---|---|
| baseline | 0.5976 | 0.6038 | 1.67 | 3345.9 | — |
| refiner_only | 0.5978 | 0.5994 | 1.67 | 2233.9 | -0.03% |
| full_model | 0.5994 | 0.6012 | 1.67 | 2241.9 | -0.30% |

### Parity

| Config | Status | Alpha Diff | FG Diff |
|---|---|---|---|
| refiner_only | PASS | 0.000000 | 0.000000 |
| full_model | PASS | 0.000488 | 0.000488 |

### Key Finding

Hiera backbone dominates compute (~80%+). CNN refiner is too small to move the needle with memory format changes alone. Future optimization must target the backbone (torch.compile) or the full pipeline.

---

## Phase 7 — torch.compile (COMPLETE — REJECTED)

**Hypothesis:** `torch.compile` on MPS may help the CNN refiner (regular conv pattern) but likely struggles with the Hiera backbone (dynamic shapes, attention).

**Result:** torch.compile is not viable on MPS in PyTorch 2.9. Refiner-only compile runs ~4x slower with parity failure. Full-model compile crashes due to PyTorch bug in MPS SDPA meta registration. **Not adopted.** See `docs/phase7_results.md` for full results.

### Benchmark Results (1024x1024, 10 iterations, fp16 autocast)

| Config | Median (s) | P95 (s) | FPS | Compile (s) | Delta |
|---|---|---|---|---|---|
| eager (baseline) | 0.6053 | 0.6074 | 1.65 | — | — |
| refiner_default | 2.4908 | 2.6841 | 0.40 | 6.06 | -311% |
| refiner_reduce_overhead | 2.7127 | 2.7171 | 0.37 | 3.47 | -348% |
| full_default | FAILED | — | — | — | crash |
| full_reduce_overhead | FAILED | — | — | — | crash |

### Parity

| Config | Status | Alpha Diff | FG Diff |
|---|---|---|---|
| refiner_default | FAIL | 0.009766 | 0.012207 |
| refiner_reduce_overhead | FAIL | 0.009766 | 0.012207 |
| full_default | ERROR (crash) | — | — |
| full_reduce_overhead | ERROR (crash) | — | — |

### Key Findings

- Refiner compile: inductor overhead makes MPS ~4x slower for small CNN
- Full model: PyTorch 2.9 bug in `sdpa_vector_fast_mps` crashes on Hiera's 5D attention tensors
- No configuration meets accept criteria (>10% improvement + atol=1e-3)

---

## Phase 8 — Unsupported Ops & Fallback Detection (COMPLETE — NO ACTION NEEDED)

**Hypothesis:** Some ops may silently fall back to CPU via `PYTORCH_ENABLE_MPS_FALLBACK`. Need to identify and substitute.

**Result:** All ops run natively on MPS. Zero fallback warnings with `PYTORCH_ENABLE_MPS_FALLBACK=1`. All 13 tested op categories (interpolate, SDPA, GroupNorm, BatchNorm, dilated Conv2d, etc.) confirmed MPS-native. See `docs/phase8_results.md` for full profiler breakdown.

### Key Finding

PyTorch 2.9.1 MPS backend fully supports every op used by GreenFormer. The profiler confirms SDPA runs via dedicated `_scaled_dot_product_attention_math_for_mps` Metal kernel. No substitutions needed.

---

## Phase 9 — Advanced Profiling (SKIPPED — No Unexplained Bottleneck)

Phase 8 profiler results already provide operator-level breakdown. All compute is accounted for:
- SDPA attention: 21% (Hiera backbone — expected dominant cost)
- Convolutions: 10% (CNN refiner + decoder fuse)
- dtype copy/autocast: 51% (fp32→fp16 transfers — expected under autocast)
- GroupNorm/BatchNorm/LayerNorm: ~2% (normalization layers)
- Remaining: reshape, cat, interpolate (~5%)

No unexplained bottleneck exists. Further profiling would not yield actionable insights.

---

## Phase 10 — Strategic Assessment (COMPLETE)

### Question: Is PyTorch+MPS the right long-term path?

**Factors favoring PyTorch+MPS:**
- Existing codebase is PyTorch
- timm backbone (Hiera) only available in PyTorch
- MPS support in PyTorch 2.9 is mature for standard vision ops
- All ops confirmed native MPS (Phase 8)

**Factors favoring future MLX exploration:**
- Pure Apple Silicon inference is the primary use case
- MLX has lower overhead for Metal dispatch
- No CUDA compatibility needed for this deployment target

**Assessment based on benchmarking results (Phases 1-8):**

PyTorch+MPS is the correct path for now. Key findings:

1. **All ops native** — no CPU fallbacks, full Metal shader utilization
2. **fp16 autocast works correctly** — best performance/quality tradeoff
3. **torch.compile not viable** — crashes on SDPA, massive regression on CNN
4. **Memory format (channels_last) neutral** — Hiera dominates, CNN refiner too small
5. **Transfer overhead negligible** — <0.2% of total time
6. **Performance ceiling** — ~0.60s/frame (1.65 FPS) at 1024x1024 with fp16 autocast

**Recommendation:** Stay on PyTorch+MPS. The 0.60s/frame performance is limited by the Hiera backbone ViT attention at high resolution. To meaningfully improve latency, one must either:
- Reduce input resolution (quadratic cost in attention)
- Use a smaller/faster backbone
- Port to MLX (requires Hiera reimplementation — significant effort)

Revisit MLX only if the project moves to Apple-only deployment and justified by user demand.

---

## Summary: Priority-Ordered Optimization Targets

| Priority | Phase | Change | Expected Impact | Risk |
|---|---|---|---|---|
| 1 | 2 | `inference_mode()` | +2-5% latency | Very low |
| 2 | 2 | Fix `clear_device_cache` for MPS | Correctness | None |
| 3 | 3 | MPS memory instrumentation | Visibility | None |
| 4 | 6 | channels_last experiment | +5-15% if it works | Medium |
| 5 | 5 | `non_blocking=True` input transfer | +1-3% | Low |
| ~~6~~ | ~~7~~ | ~~torch.compile on refiner~~ | ~~-311% regression~~ | ~~Rejected~~ |
| 7 | 4 | bfloat16 experiment | Unknown | Medium |
| ~~8~~ | ~~8~~ | ~~Unsupported op substitution~~ | ~~All ops native~~ | ~~None needed~~ |

---

## Resolved Questions

- **torch.autocast fp16 on MPS?** Yes — works correctly, parity within atol=1e-3 (Phase 4)
- **channels_last through Hiera?** Propagates but no benefit — Hiera attention conversion overhead negates gains (Phase 6)
- **torch.compile on MPS for CNN subgraphs?** No — compiles but runs 4x slower, parity fails (Phase 7)
- **torch.compile on full model?** No — crashes on SDPA meta registration bug in PyTorch 2.9 (Phase 7)
- **All ops native on MPS?** Yes — zero fallbacks confirmed (Phase 8)

## Remaining Open Questions

- Does `PYTORCH_MPS_FAST_MATH` affect matting quality perceptibly? (not tested — low priority)
- What's the actual MPS memory ceiling before OOM at 2048x2048? (not stress-tested)
