# Further Performance Optimization Research Avenues

Beyond Phases 0-5 (FP16, GPU postprocess, backbone 1024, tiled refiner, CLI flags). Organized by likely impact.

---

## High-Potential

### 1. `torch.compile()` Graph Fusion

Refiner is a straightforward dilated CNN — ideal for `torch.compile(mode="reduce-overhead")`. Fuses GroupNorm+ReLU+Conv chains, eliminates per-op kernel launches. On CUDA expect 20-40% refiner speedup. MPS backend support is maturing — worth benchmarking both. Zero quality impact.

### 2. CoreML / Apple Neural Engine Export

MPS inference leaves the ANE (Neural Engine) completely idle. CoreML export via `coremltools` could route backbone or refiner to ANE, which does ~15 TOPS at a fraction of GPU power. Hiera's windowed attention and the CNN refiner are both ANE-friendly op patterns. Potentially the single largest win on Apple Silicon — some models see 3-5x speedup.

**Research:** Which ops fall back to GPU/CPU? Is ANE's precision (FP16/INT16) acceptable for our color math?

### 3. Sparse / Masked Refiner Inference

Refiner corrects edge regions — pure foreground and pure background need zero refinement. Use coarse alpha to build a binary mask of "interesting" pixels (e.g., `0.01 < alpha < 0.99` dilated by receptive field), only run refiner on those tiles/regions. For typical green screen shots, edges are ~10-20% of pixels. Could skip 80% of refiner compute with zero quality loss on skipped regions.

### 4. Temporal Coherence for Video Sequences

Consecutive video frames share ~95% of content. Directions:

- **Feature caching**: Cache encoder features from frame N, warp to frame N+1 via optical flow, only re-encode regions with large motion
- **Lightweight temporal propagation**: Small ConvLSTM/ConvGRU on top of frame features — warm-starts from previous frame's alpha/FG
- **Adaptive keyframe strategy**: Full inference every Nth frame, lightweight propagation between keyframes

### 5. INT8 Post-Training Quantization of Refiner

CNN refiner is 4 residual blocks of 3x3 convs — extremely quantization-friendly topology. PyTorch's `torch.ao.quantization` or ONNX-based INT8 PTQ could halve refiner compute beyond FP16. Backbone (attention + softmax) is riskier for INT8, but refiner alone is low-risk.

**Research:** Calibration dataset, per-channel vs per-tensor, quality degradation.

### 6. Flash / Memory-Efficient Attention in Hiera

Check whether `timm`'s Hiera uses Flash Attention or naive attention. Hiera uses windowed (local) attention — ideal for Flash Attention's tiling scheme. If not already enabled, `torch.nn.functional.scaled_dot_product_attention` with `enable_flash=True` could reduce backbone memory by 50%+ and improve throughput. Especially impactful at 2048x2048 where sequence length is massive.

### 7. ONNX Runtime with Execution Providers

Export full model to ONNX, run via `onnxruntime` with:

- `CoreMLExecutionProvider` (ANE on macOS)
- `CUDAExecutionProvider` with TensorRT subgraph optimization (CUDA)
- `DmlExecutionProvider` (Windows DirectML — opens AMD/Intel GPU support)

Bypasses PyTorch overhead entirely, enables cross-platform hardware-specific optimization.

**Research:** Which ops need custom implementations? Dynamic shapes support?

---

## Medium-Potential / More Speculative

### 8. Knowledge Distillation to Lighter Backbone

Train a smaller backbone (EfficientViT, MobileViT, or smaller Hiera variant) to match current model outputs. Target: 80% quality at 20% compute. Refiner already recovers fine detail — weaker backbone + strong refiner might be viable. Requires training infrastructure but could be biggest long-term win.

### 9. Batched Multi-Frame Inference

Model supports batch dim > 1. With Phase 1+4 VRAM savings, processing 2-4 frames simultaneously improves GPU utilization. Research: at what batch size does VRAM bottleneck again, and throughput gain per frame.

### 10. Token Merging (ToMe) for Adaptive Resolution

Instead of uniform downsampling to 1024, use content-aware token/patch merging — keep full resolution at edge regions, merge patches in flat areas. Hiera's hierarchical structure may support this natively.

**Research:** ToMe applied to Hiera specifically — any prior work?

### 11. Asynchronous Pipeline (Overlap I/O and Compute)

While frame N is in inference, frame N+1 reads from disk and preprocesses, frame N-1 writes to EXR. Use Python threading or `torch.cuda.Stream` to overlap:

- Disk read + resize + normalize (CPU)
- Model inference (GPU)
- Post-process + EXR write (CPU)

Doesn't reduce per-frame cost but improves throughput by hiding I/O latency. ~15-30% wall-clock improvement depending on I/O speed.

### 12. CUDA Graphs (CUDA Only)

Record entire inference path as a CUDA graph — eliminates all kernel launch overhead and Python-side dispatch. Very effective for models with many small ops (refiner's 8 conv layers). Requires static shapes (already fixed at 2048x2048). Zero quality impact, ~10-20% throughput gain.

### 13. Decoder Weight Sharing / Pruning

Alpha and FG decoders are structurally identical (4 MLPs + fuse + classifier each). Directions:

- **Shared decoder**: Fuse into one decoder with 4 output channels instead of separate 1+3
- **Structured pruning**: Sensitivity analysis on `embedding_dim` (currently 256) — would 128 or 192 suffice?

Both reduce parameters and compute with potential minor quality tradeoff.

### 14. FP8 Inference (Hopper/Ada GPUs)

NVIDIA H100/4090 support native FP8 (E4M3/E5M2). If targeting modern CUDA hardware, FP8 could halve compute again beyond FP16. PyTorch supports via `torch.float8_e4m3fn`.

**Research:** Which layers tolerate FP8? Mixed FP8/FP16 strategies.

---

## Key Questions for Research Agents

- Does `timm`'s Hiera implementation use `scaled_dot_product_attention`? If not, what's needed to enable it?
- Current state of `torch.compile` on MPS backend? Which ops fall back?
- CoreML export of Hiera — any known blockers with windowed attention ops?
- Token Merging (ToMe) applied to Hiera specifically — any prior work?
- Typical sparsity ratio of alpha edges in green screen footage (% of pixels in 0.01-0.99 range)?
