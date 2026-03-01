# Phase 7: torch.compile Results

**Date:** 2026-03-01
**Device:** MPS (Apple Silicon) | **Size:** 1024x1024 | **Iterations:** 10 | **PyTorch:** 2.9.1

## Verdict: REJECTED — torch.compile not viable on MPS

## Latency

| Config | Median (s) | P95 (s) | FPS | Compile (s) | Graph Breaks | Delta |
|---|---|---|---|---|---|---|
| eager (baseline) | 0.6053 | 0.6074 | 1.65 | — | — | — |
| refiner_default | 2.4908 | 2.6841 | 0.40 | 6.06 | 0 | -311% |
| refiner_reduce_overhead | 2.7127 | 2.7171 | 0.37 | 3.47 | 0 | -348% |
| full_default | FAILED | — | — | — | — | — |
| full_reduce_overhead | FAILED | — | — | — | — | — |

## Parity (vs eager)

| Config | Status | Alpha Max Diff | FG Max Diff | NaN |
|---|---|---|---|---|
| refiner_default | FAIL | 0.009766 | 0.012207 | No |
| refiner_reduce_overhead | FAIL | 0.009766 | 0.012207 | No |
| full_default | ERROR (crash) | — | — | — |
| full_reduce_overhead | ERROR (crash) | — | — | — |

## Analysis

### Refiner-only compile: massive regression

Both `mode="default"` and `mode="reduce-overhead"` produce ~4x slower inference. The inductor backend on MPS adds substantial overhead for the small CNN refiner (4 dilated res blocks, 64 channels). The compilation cost (~3-6s) is never recovered since steady-state latency is far worse than eager.

Parity fails: max diffs (~0.01) are ~10x the acceptance threshold (atol=1e-3). No NaN produced, but numerical divergence exceeds tolerance.

### Full model compile: PyTorch bug

Compilation crashes with `TypeError: can only concatenate list (not "torch.Size") to list` in `torch._meta_registrations.meta__scaled_dot_product_attention_math_for_mps`. This is a bug in PyTorch 2.9's MPS SDPA meta registration — the `sdpa_vector_fast_mps()` function incorrectly handles shape concatenation for Hiera's 5D attention tensors `(1, 2, 1024, 64, 56)`.

This confirms the plan's risk assessment: Hiera + timm + attention = incompatible with torch.compile on MPS.

## Conclusion

torch.compile on MPS is not ready for this workload in PyTorch 2.9:
- CNN-only subgraphs (refiner) compile but run ~4x slower with parity failure
- Full model with SDPA attention crashes during compilation
- No configuration meets the >10% improvement + atol=1e-3 acceptance criteria

**Recommendation:** Skip torch.compile entirely. The MPS eager backend at 0.60s/frame (1.65 FPS) at 1024x1024 with fp16 autocast represents the practical performance ceiling for this model on Apple Silicon via PyTorch+MPS. Future improvement requires either:
1. PyTorch MPS backend maturation (compile support, Metal shader optimization)
2. MLX port (Phase 10 consideration)
3. Model architecture changes (smaller backbone, reduced resolution)
