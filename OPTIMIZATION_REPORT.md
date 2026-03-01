# CorridorKey MPS Inference Optimization Report

*Environment: macOS 26.3 · Python 3.12.8 · PyTorch 2.9.1 · Apple Silicon MPS (M-series)*
*Completed: 2026-03-01*

---

## 1. Executive Summary

Comprehensive inference performance audit of CorridorKey's GreenFormer model (Hiera ViT backbone + CNN refiner) on Apple Silicon MPS. Covers a 10-phase model-level optimization audit plus full-pipeline benchmarking at production resolution (2048x2048), MPS fast math testing, and preprocessing/postprocessing profiling.

**Key findings:**
- **1024x1024**: 0.60s/frame, 1.65 FPS, ~3.3 GB peak memory
- **2048x2048**: 14.6s/frame, 0.07 FPS, ~18.9 GB peak memory
- **Bottleneck**: 93% of pipeline time is Hiera ViT attention (quadratic in sequence length)
- **MPS fast math**: Rejected — 12.5% slower with large parity diffs
- **Pre/postprocessing**: ~5.1% of total at 2048x2048 — marginal optimization target
- **All MPS ops native**: Zero CPU fallbacks across 13 tested op categories
- **3 changes adopted**, 5 experiments rejected, 2 new benchmark scripts delivered

**Recommendation: Ship as-is.** The model is at the MPS hardware-limited ceiling for this architecture. Further gains require architectural changes (smaller backbone, lower resolution, or MLX port), not PyTorch-level tuning.

---

## 2. Prioritized Optimization Table

| # | Priority | File | Problem | Fix | Speed Impact | Memory Impact | Risk | Effort | Implemented? |
|---|----------|------|---------|-----|-------------|---------------|------|--------|-------------|
| 1 | **Done** | `inference_engine.py` | `no_grad` suboptimal for inference | `torch.inference_mode()` | Marginal | Marginal | None | Low | Yes |
| 2 | **Done** | `device_utils.py` | No MPS cache clearing | `torch.mps.empty_cache()` branch | None (correctness) | Prevents buildup | None | Low | Yes |
| 3 | **Done** | `inference_engine.py` | Blocking `.to(device)` | `non_blocking=True` | +0.07% | None | None | Low | Yes |
| 4 | **Rejected** | model | channels_last memory format | Hiera attention conversion overhead | -0.3% (regression) | None | None | Low | No |
| 5 | **Rejected** | model | torch.compile | Inductor MPS overhead | -311% (refiner), crash (full) | N/A | High | Medium | No |
| 6 | **Rejected** | model | bfloat16 autocast | Marginal parity failure | Neutral | None | Medium | Low | No |
| 7 | **Rejected** | env | `PYTORCH_MPS_FAST_MATH=1` | 12.5% slower, large parity diffs | -12.5% (regression) | None | High | Low | No |
| 8 | **Future** | `inference_engine.py` | Checkerboard regenerated per-frame | Cache per resolution | <0.1s saving | None | None | Low | No |
| 9 | **Future** | `inference_engine.py` | Pre/post on CPU NumPy | Move to MPS tensors | <0.8s on 15s frame | None | Low | Medium | No |
| 10 | **Future** | architecture | Hiera ViT quadratic attention | Smaller backbone or lower resolution | 4x+ at half res | Proportional | High | High | No |

---

## 3. Benchmark Methodology

### Model-Forward Benchmark (`scripts/benchmark_mps.py`)

Existing harness measuring sync-aware model-forward latency. Supports device/dtype matrix, parity checking, torch.compile modes.

- `torch.mps.synchronize()` before all timer reads
- `torch.inference_mode()` + `torch.autocast(dtype=torch.float16)`
- Warmup: 3 iterations (excluded from measurements)
- Synthetic input: `torch.randn(1, 4, H, W)` on device

### Full-Pipeline Benchmark (`scripts/benchmark_inference.py`)

New harness measuring 7 pipeline stages independently:

| Stage | What it measures | Sync strategy |
|-------|-----------------|---------------|
| 1. Input gen | Synthetic frame + mask creation | None (CPU) |
| 2. Preprocess | cv2.resize + ImageNet normalize + concat + transpose | None (CPU) |
| 3. CPU→MPS transfer | `.to(device, non_blocking=False)` | Sync before + after |
| 4. Model forward | autocast fp16 + model(inp_t) | Sync before + after |
| 5. MPS→CPU transfer | `.cpu().numpy()` (implicit sync) | Sync before |
| 6. Postprocess | resize + clean_matte + despill + color + premul + checkerboard | None (CPU) |
| 7. Total | End-to-end with sync at boundaries | Sync before + after |

- Runs both `auto_despeckle=True` and `False` separately
- Default synthetic alpha: radial gradient (realistic single-blob matte topology)
- OOM handling: catches RuntimeError, writes `{"oom": true}` JSON, exits 0

### MPS Fast Math Benchmark (`scripts/benchmark_fast_math.py`)

Orchestrator/worker pattern to ensure `PYTORCH_MPS_FAST_MATH=1` is set before `import torch`:

- Spawns two subprocess workers: control (no fast math) vs fast math
- Deterministic inputs (seed=42) for valid parity comparison
- Compares raw `out["alpha"]` and `out["fg"]` tensors **before** postprocessing
- Quality threshold: max_abs_diff > 1/255 (~0.004) = visible in 8-bit delivery

---

## 4. Benchmark Results

### 4.1 Model-Forward at 1024x1024 (Phase 1-10 Audit)

```
Device:          MPS (Apple Silicon)
Dtype:           fp16 autocast
Resolution:      1024x1024
Median latency:  0.60s/frame
P95 latency:     0.60s/frame
Throughput:      1.65 FPS
Peak memory:     ~3.3 GB
```

### 4.2 Full Pipeline at 2048x2048

#### With `auto_despeckle=True`

```
Stage                    Median      % of Total
─────────────────────────────────────────────────
input_gen                0.0000s       0.0%
preprocess               0.0797s       0.5%
transfer_to_device       0.0023s       0.0%
model_forward           13.6068s      93.5%  ██████████████████████████████████████████████
transfer_to_cpu          0.1891s       1.3%
postprocess              0.7400s       5.1%  ██
─────────────────────────────────────────────────
total                   14.5559s     100.0%

Throughput:          0.07 FPS
Peak driver memory:  18,898 MB (~18.5 GB)
Recommended max:     28,754 MB (~28.1 GB)
```

#### With `auto_despeckle=False`

```
Stage                    Median      % of Total
─────────────────────────────────────────────────
input_gen                0.0000s       0.0%
preprocess               0.0926s       0.6%
transfer_to_device       0.0076s       0.0%
model_forward           15.2185s      93.1%  ██████████████████████████████████████████████
transfer_to_cpu          0.3131s       1.9%
postprocess              0.8258s       5.1%  ██
─────────────────────────────────────────────────
total                   16.3469s     100.0%

Throughput:          0.06 FPS
Peak driver memory:  19,922 MB (~19.5 GB)
```

### 4.3 Full Pipeline at 512x512

```
Stage                    Median      % of Total
─────────────────────────────────────────────────
preprocess               0.0038s       2.1%
transfer_to_device       0.0009s       0.5%
model_forward            0.1536s      83.6%  █████████████████████████████████████████
transfer_to_cpu          0.0008s       0.4%
postprocess              0.0245s      13.3%  ██████
─────────────────────────────────────────────────
total                    0.1837s     100.0%

Throughput:          5.44 FPS
Peak driver memory:  1,314 MB (~1.3 GB)
```

### 4.4 Resolution Scaling Summary

| Resolution | Model Forward | Total Pipeline | FPS | Peak Memory |
|-----------|--------------|---------------|-----|-------------|
| 512x512 | 0.15s | 0.18s | 5.44 | 1.3 GB |
| 1024x1024 | 0.60s | ~0.65s* | 1.65 | 3.3 GB |
| 2048x2048 | 13.6s | 14.6s | 0.07 | 18.5 GB |

*\*1024 total estimated from model-forward + typical 8% overhead.*

Scaling is approximately quadratic in sequence length, as expected for ViT attention.

### 4.5 MPS Fast Math Results

```
Resolution:  512x512 (random weights)

Latency (median):
  Control:    0.1280s  (7.81 FPS)
  Fast Math:  0.1441s  (6.94 FPS)
  Delta:      -12.52% (SLOWER)

Output Parity (raw model tensors):
  alpha:  max_abs_diff = 0.1326   mean = 0.0240   89.6% pixels > 1/255
  fg:     max_abs_diff = 0.1931   mean = 0.0403   93.5% pixels > 1/255

8-bit invisible: NO
```

**Recommendation: REJECT.** MPS fast math is both slower and introduces large parity deviations. Not viable for matting.

### 4.6 Compute Breakdown (1024x1024, from Phase 8 profiler)

| Component | CPU Time % | Notes |
|---|---|---|
| dtype copy (autocast fp32→fp16) | 51% | Expected under torch.autocast |
| SDPA attention (Hiera) | 21% | Dominant model compute — native MPS Metal kernel |
| ReLU activations | 12% | Native MPS |
| MPS convolution | 10% | CNN refiner + decoder fuse |
| Normalization (Group/Batch/Layer) | 2% | Native MPS |
| Other (reshape, cat, interpolate) | 4% | Native MPS |

### 4.7 MPS Op Compatibility

All 13 tested op categories confirmed MPS-native — zero CPU fallbacks:

| Op | Status |
|---|---|
| F.interpolate (bilinear, bicubic) | Native |
| scaled_dot_product_attention | Native (dedicated MPS kernel) |
| Conv2d (standard, dilated) | Native |
| GroupNorm, BatchNorm, LayerNorm | Native |
| Linear, ReLU, sigmoid, cat | Native |
| max_pool2d (3x3, 7x7) | Native |

### 4.8 Transfer Overhead

| Direction | Median | % of Inference |
|---|---|---|
| CPU→MPS input transfer | 0.5ms (1024) / 2.3ms (2048) | <0.1% |
| MPS→CPU output (.cpu().numpy()) | 0.7ms (1024) / 189ms (2048) | 0.1% / 1.3% |

Transfer is negligible at all resolutions.

---

## 5. Code Changes Made

### Adopted in Production Code

| Change | File | Phase |
|---|---|---|
| `torch.inference_mode()` decorator | `inference_engine.py:82` | 2 |
| `torch.mps.empty_cache()` in clear_device_cache | `device_utils.py:102` | 2 |
| `non_blocking=True` on `.to(device)` | `inference_engine.py:148` | 5 |

### Benchmark Infrastructure Added

| Script | Purpose |
|---|---|
| `scripts/benchmark_mps.py` | Model-forward benchmark (device/dtype matrix, parity, compile modes) |
| `scripts/benchmark_inference.py` | Full-pipeline benchmark (7-stage timing, OOM handling, JSON output) |
| `scripts/benchmark_fast_math.py` | MPS fast math parity + speed comparison (subprocess isolation) |
| `scripts/mps_env_report.py` | System/PyTorch/MPS diagnostic report |
| `scripts/detect_mps_fallback.py` | Unsupported op and CPU fallback detection |

---

## 6. Remaining Opportunities

### Low-Impact (< 5% improvement potential)

| Opportunity | Current Cost | Potential Saving | Notes |
|---|---|---|---|
| Cache checkerboard per resolution | ~50ms at 2048 | ~50ms | Trivial to implement, low priority |
| Move ImageNet normalize to MPS | ~10ms at 2048 | ~10ms | Eliminates one NumPy allocation |
| Move despill/color_utils to MPS tensors | ~200ms at 2048 | ~100ms | color_utils already supports torch |
| Skip `.float()` if already float32 | ~5ms | ~5ms | Guard check |

### High-Impact (requires architectural changes)

| Opportunity | Expected Impact | Effort | Risk |
|---|---|---|---|
| **Reduce input resolution** (e.g. 1024 for preview, 2048 for final) | ~4x faster at half res | Low | Quality tradeoff |
| **Smaller backbone** (Hiera Small vs Base Plus) | Significant speedup | High (retrain) | Accuracy tradeoff |
| **MLX port** | Lower Metal dispatch overhead | Very High | Requires Hiera reimplementation |
| **Wait for PyTorch torch.compile MPS** | Unknown | None | Revisit on PyTorch 2.10+ |
| **CUDA deployment** | 10-50x faster on A100/H100 | Medium | Requires CUDA hardware |

---

## 7. Final Recommendation

**Ship now.** The model is at the MPS hardware-limited performance ceiling for this architecture:

- All ops dispatch natively to MPS Metal shaders
- `inference_mode()`, `non_blocking`, and `empty_cache` are adopted
- fp16 autocast confirmed as best dtype
- channels_last, torch.compile, bfloat16, and MPS fast math all rejected with data
- No pre/post optimization can meaningfully close the gap when model forward is 93% of the pipeline

**For production at 2048x2048**: Expect ~15s/frame and ~19 GB peak memory on Apple Silicon. This is architecture-bound (Hiera ViT attention, quadratic in sequence length). Meaningful speedup requires one of:
1. Resolution reduction (1024 for interactive preview, 2048 for final render)
2. Backbone change (Hiera Small, or non-ViT architecture)
3. CUDA hardware (when available)

**CUDA next-steps**: When CUDA hardware is available, run `scripts/benchmark_inference.py --device cuda` and `scripts/benchmark_fast_math.py` on GPU. The benchmark infrastructure is device-agnostic and ready. Expect 10-50x improvement on datacenter GPUs due to higher memory bandwidth, tensor cores, and mature CUDA kernel ecosystem.

---

## Appendix: Resolved Questions

| Question | Answer | Source |
|---|---|---|
| Does fp16 autocast work on MPS? | Yes, parity within atol=1e-3 | Phase 4 |
| Does channels_last help through Hiera? | No — conversion overhead negates gains | Phase 6 |
| Is torch.compile viable on MPS? | No — 4x regression (CNN), crash (full model) | Phase 7 |
| Are any ops falling back to CPU? | No — all 13 categories native MPS | Phase 8 |
| Is there a memory leak? | No — stable across iterations | Phase 3 |
| Does `PYTORCH_MPS_FAST_MATH=1` help? | No — 12.5% slower with large parity diffs | Phase 2 (new) |
| What is MPS memory ceiling at 2048x2048? | ~19 GB peak, no OOM on 32 GB machine | Phase 1 (new) |
| Is pre/post worth optimizing? | Borderline — 5.1% of total, <1s on 15s frame | Phase 3 (new) |
