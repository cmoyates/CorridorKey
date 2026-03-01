# Phase 4 — dtype & Precision Audit Results

*Date: 2026-03-01*
*Device: Apple Silicon MPS (PyTorch 2.9.1)*
*Test size: 512x512 (quick validation), 5 iterations*

---

## Audit Checklist

All locations verified correct — no changes needed:

| Location | Current | Status |
|---|---|---|
| `inference_engine.py:148` | `.float()` (fp32 input) | Correct — autocast handles downcast |
| `inference_engine.py:160` | `autocast(dtype=float16)` | Correct |
| `inference_engine.py:20-21` | np.float32 mean/std | Correct |
| Post-process (lines 171-174) | `.cpu().numpy()` -> fp32 | Correct — CPU-side cv2 |
| GVM UNet timesteps (`unet_spatio_temporal_condition.py:495-500`) | fp32 for MPS, fp64 for CUDA | Already handled |
| VideoMaMa pipeline (`pipeline.py:874-877`) | fp32 CLIP/VAE, fp16 UNet | Correct mixed-precision |
| GVM pipeline (`pipeline_gvm.py:67,83,91,151`) | VAE fp16, output cast to fp32 | Correct |

No float64 accumulation found anywhere in the MPS inference path.

---

## Benchmark Matrix (MPS, 512x512, 5 iters)

| Config | Autocast | Median (s) | P95 (s) | FPS | Peak Mem (MB) |
|---|---|---|---|---|---|
| fp32 no-autocast | Off | 0.3037 | 0.3062 | 3.29 | 1315.9 |
| fp16 autocast | On | 0.2983 | 0.3016 | 3.35 | 1316.0 |
| bfloat16 autocast | On | 0.2648 | 0.3955 | 3.78 | 1316.1 |

Working memory (current_alloc): fp16/bf16 ~296 MB vs fp32 ~301-308 MB.

At 512x512 all three configs are close. Larger resolution (2048) expected to show bigger gaps.

---

## Parity Checks (CPU fp32 reference vs MPS)

| Target dtype | Alpha | FG | Max Diff | Status |
|---|---|---|---|---|
| float32 | PASS | PASS | 0.000000 | Bit-exact |
| float16 | PASS | PASS | 0.000392 | Within atol=1e-3 |
| bfloat16 | FAIL | FAIL | 0.003361 | Above atol=1e-3 (3.4x) |

### bfloat16 Parity Analysis

bf16 has 8-bit mantissa vs fp16's 11-bit, so larger numerical differences are expected. The max_diff of 0.003361 is marginal — only 3.4x above the 1e-3 threshold. For matting, this level of error is likely imperceptible. If bf16 is adopted, the parity tolerance should be widened to atol=5e-3 for bf16 specifically.

---

## Key Findings

1. **bfloat16 IS supported on MPS in PyTorch 2.9** — works without errors, competitive performance.
2. **fp16 autocast is the recommended default** — best parity (PASS) with slight latency improvement over fp32.
3. **bf16 shows higher variance** — P95 latency (0.3955s) much higher than median (0.2648s), suggesting less stable dispatch.
4. **fp32 no-autocast works as expected** — bit-exact parity with CPU, suitable as control baseline.
5. **No float64 accumulation** in any MPS code path — audit clean.
6. **Memory is nearly identical** across dtypes at 512x512 — peak allocation dominated by model weights + MPS allocator overhead.

---

## Decisions

| Config | Decision | Reason |
|---|---|---|
| fp16 autocast (current default) | **Keep** | Best parity + good performance |
| bfloat16 autocast | **Do not adopt** | Marginal parity failure, higher variance |
| fp32 no-autocast | Available as `--no-autocast` flag | Control baseline for future benchmarks |

---

## Benchmark Harness Changes

- Added `--no-autocast` flag for true fp32 control baseline
- Added `--phase4` convenience flag for dtype audit matrix
- Added bfloat16 to `--all` matrix with graceful RuntimeError handling
- Fixed parity check: now shares weights between CPU and target (was building separate random-weight models)
- Parity output tensors cast to float32 before comparison (fixes dtype mismatch crash)

---

## Command to Reproduce

```bash
python scripts/benchmark_mps.py --phase4 --img-size 512 --iterations 5 --verbose
```

For full-resolution validation (slow, ~15-20 min):
```bash
python scripts/benchmark_mps.py --phase4 --img-size 2048 --iterations 10
```
