---
title: "feat: Throughput & Temporal Coherence Optimizations (Phases 6-10)"
type: feat
date: 2026-03-07
depends_on: 2026-03-07-feat-vram-performance-optimizations-plan.md
branch: feature/misc-optimizations
---

# Throughput & Temporal Coherence Optimizations (Phases 6-10)

## Overview

Five-phase optimization plan targeting sub-1s frame times and temporal video coherence. Builds on Phases 0-5 (FP16, GPU postprocess, 1024 backbone, tiled refiner, CLI flags). Phases 6-8 maximize single-frame throughput; Phase 9 exploits temporal redundancy in video; Phase 10 targets Apple Neural Engine.

**Current baseline (Phase 4, MPS M3 Max):** 6.35s/frame median, 15.36 GB peak memory.

## Implementation Phases

---

### Phase 6: Sparse / Masked Tiled Inference (Compute Skipping)

**Goal:** Skip CNN refiner forward pass for tiles that are entirely solid BG or FG. Typical green screen: edges are ~10-20% of pixels, so 70-80% of refiner tiles can be skipped.

**Files:** `model_transformer.py` (`GreenFormer._tiled_refine`, `GreenFormer.forward`)

#### 6a. Coarse Alpha Interest Mask

After `alpha_coarse = torch.sigmoid(alpha_logits_up)` at `model_transformer.py:333`:

```python
# Binary mask: 1 where alpha is "interesting" (edge region)
interest_mask = ((alpha_coarse > 0.01) & (alpha_coarse < 0.99)).float()  # [B, 1, H, W]
```

Pixels with alpha in `(0.01, 0.99)` need refinement. Pure BG (alpha < 0.01) and pure FG (alpha > 0.99) get zero refiner delta anyway -- skip them.

#### 6b. Morphological Dilation (Receptive Field Guard)

The CNN refiner has a ~65px receptive field (dilated convs d=1,2,4,8). Edge pixels need surrounding context to produce correct outputs. Dilate the interest mask to ensure the refiner sees its full context window:

```python
# Dilate interest mask by refiner receptive field
# max_pool2d with kernel_size=65, stride=1, padding=32 acts as binary dilation
interest_dilated = F.max_pool2d(interest_mask, kernel_size=65, stride=1, padding=32)
```

**Why max_pool2d:** Binary dilation on a {0,1} tensor -- max over a neighborhood. Runs on GPU, no CPU roundtrip. Kernel size 65 matches the refiner's receptive field exactly.

#### 6c. Tile Activity Classification

Downsample the dilated mask to a per-tile activity grid:

```python
# For each tile position, check if ANY pixel is active
# Use avg_pool over tile_stride-sized blocks; >0 means at least one active pixel
tile_stride = tile_size - overlap
grid_h = (H - tile_size) // tile_stride + 1
grid_w = (W - tile_size) // tile_stride + 1

# Sample tile centers -- if dilated mask has any nonzero pixel in tile region, mark active
```

Implementation: iterate tile starts (reuse existing `_starts()` helper), check `interest_dilated[:, :, y:y+ts, x:x+ts].any()` per tile. This check is O(1) GPU time per tile.

#### 6d. Execution: Skip vs Refine

Modify `_tiled_refine()` to accept the dilated interest mask:

- **Empty tile** (`mask.any() == False`): Skip refiner entirely. The delta for this tile is zero -- coarse predictions pass through unchanged. No need to even upsample; `alpha_logits_up + 0 = alpha_logits_up`. Simply skip the tile in the accumulator (tent-weighted zero delta = no contribution needed since coarse already covers it).

- **Active tile**: Run refiner as before (existing path). Accumulate delta into CPU accumulator with tent blending.

**Key insight:** We don't need to "fill in" skipped tiles with bicubic-upsampled coarse alpha. The coarse predictions (`alpha_coarse`, `fg_coarse`) are already at full resolution from `F.interpolate` at line 323-324. The refiner only adds delta corrections. For skipped tiles, delta = 0, so `final = coarse + 0 = coarse`. The existing code path at lines 365-370 handles this automatically -- we just need to ensure `delta_logits` is zero in skipped regions.

**Simplest implementation:** Keep the current accumulator logic but `continue` past skipped tiles. The accumulator starts at zero. After the loop, replace regions where `weight_acc == 0` (no tile contributed) with zero delta:

```python
mask_no_contribution = (weight_acc == 0)
output_acc[mask_no_contribution.expand_as(output_acc)] = 0.0
weight_acc[mask_no_contribution] = 1.0  # avoid div-by-zero
```

#### 6e. Performance Expectations

| Metric | Phase 4 (current) | Phase 6 (est.) | Notes |
|--------|-------------------|----------------|-------|
| Refiner tiles processed | 25 (5x5 grid) | ~5-8 | Depends on edge coverage |
| Refiner time | ~4.5s | ~1.0-1.5s | ~70% skip rate typical |
| Quality impact | -- | Zero | Skipped tiles get exact coarse output |

**Quality guarantee:** Skipped tiles receive delta=0, meaning `final_logits = coarse_logits + 0`. This is mathematically identical to running the refiner and getting a zero delta output. For pure BG/FG tiles, the refiner *would* output near-zero delta anyway -- we're just avoiding the computation.

#### 6f. CLI Integration

Add `--sparse-refiner / --no-sparse-refiner` flag (default: on). Wire through `CorridorKeyEngine` -> `GreenFormer`.

### Acceptance Criteria -- Phase 6

- [x] Interest mask generated from coarse alpha (0.01 < alpha < 0.99)
- [x] Mask dilated by 65px (max_pool2d) to guard refiner receptive field
- [x] `_tiled_refine()` skips empty tiles
- [x] Zero-delta regions handled correctly (no NaN, no artifacts)
- [ ] Benchmark run -- timing improvement documented
- [x] Quality gate tests pass (should be lossless for skipped tiles)
- [x] CLI flag `--sparse-refiner` exposed
- [ ] Visual comparison: no artifacts at skip/refine boundaries

### Phase 6 Risks

- **Dilation too conservative:** kernel_size=65 may flag too many tiles as active, reducing skip rate. Mitigate: benchmark actual skip rates on reference clip; consider smaller kernel if quality permits.
- **Dilation too aggressive (kernel too small):** Edge tiles missing context could produce discontinuities. The 65px kernel exactly matches receptive field -- should be safe.
- **MPS compatibility:** `F.max_pool2d` with large kernel -- verify no MPS bugs. Fallback: CPU-side dilation if needed.

---

### Phase 7: Asynchronous I/O Pipeline (Triple Buffering)

**Goal:** Overlap disk I/O with GPU inference. Currently sequential: read -> infer -> write -> read -> ... Triple buffering pipelines all three concurrently.

**Files:** `clip_manager.py` (frame loop at lines 628-749), new `CorridorKeyModule/pipeline.py`

#### 7a. Ring Buffer Allocation

Pre-allocate 3 slots of pinned CPU memory and (optionally) GPU memory at engine init:

```python
class InferenceRingBuffer:
    """Triple-buffer for pipelined frame processing."""

    def __init__(self, img_size: int, device: torch.device):
        self.size = 3
        # Pinned CPU buffers for fast H2D transfer
        self.cpu_input = [
            torch.zeros(1, 4, img_size, img_size, dtype=torch.float32).pin_memory()
            for _ in range(3)
        ]
        # GPU inference buffers (reused, never reallocated)
        self.gpu_input = [
            torch.zeros(1, 4, img_size, img_size, dtype=torch.float32, device=device)
            for _ in range(3)
        ]
```

**Why pinned memory:** `pin_memory()` enables DMA transfers via `non_blocking=True`, allowing CPU work to overlap with H2D copy. Without pinning, `to(device)` blocks the CPU thread.

**Why static allocation:** Dynamic `torch.from_numpy().to(device)` allocates new tensors every frame. Static buffers eliminate allocator overhead and prevent memory fragmentation.

#### 7b. Thread Pool Architecture

```python
from concurrent.futures import ThreadPoolExecutor, Future

class AsyncFramePipeline:
    def __init__(self, engine: CorridorKeyEngine, ring: InferenceRingBuffer):
        self.engine = engine
        self.ring = ring
        self.read_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="io-read")
        self.write_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="io-write")
```

**Thread A (Read/Preprocess):** Reads frame from disk (EXR/PNG/video), normalizes, writes into `ring.cpu_input[F % 3]`. Runs in `read_pool`.

**Main Thread (Inference):** Copies `ring.cpu_input[(F-1) % 3]` to GPU via `non_blocking=True`, runs model, copies result back. Uses dedicated CUDA stream if available.

**Thread B (Write/Postprocess):** Takes completed result from slot `(F-2) % 3`, runs CPU-side postprocess (clean_matte), writes EXR/PNG to disk. Runs in `write_pool`.

#### 7c. Stream Isolation (CUDA)

```python
if device.type == "cuda":
    self.compute_stream = torch.cuda.Stream()
    self.copy_stream = torch.cuda.Stream()
```

- **copy_stream:** H2D transfer (`non_blocking=True`)
- **compute_stream:** Model forward pass
- Record events between streams for synchronization

**MPS note:** MPS has a single command queue -- no multi-stream support. Triple buffering still helps by overlapping CPU I/O with GPU compute (GIL released during MPS kernel execution), but no stream-level parallelism. The `non_blocking=True` path still works for MPS (returns immediately, GPU executes asynchronously).

#### 7d. Pipeline Warmup & Drain

- **Warmup:** First 2 frames fill the pipeline (frame 0: read only, frame 1: read + infer frame 0)
- **Drain:** After last frame, wait for final write to complete
- **Error handling:** If read fails (corrupt frame), inject a sentinel to skip inference + write for that slot

#### 7e. Integration with clip_manager.py

Replace the sequential `for i in range(num_frames)` loop (line 628) with:

```python
pipeline = AsyncFramePipeline(engine, ring_buffer)
pipeline.process_clip(input_source, alpha_source, output_dirs, num_frames)
```

The `ClipManager` passes frame sources (video cap or file list) and output dirs. Pipeline handles the rest.

#### 7f. Performance Expectations

| Metric | Sequential (current) | Pipelined (est.) | Notes |
|--------|---------------------|------------------|-------|
| Wall time per frame | infer + read + write | max(infer, read, write) | I/O fully hidden |
| Throughput gain | -- | +15-30% | Depends on I/O speed vs inference time |
| Memory overhead | 1 frame | 3 frames pinned CPU | ~150MB extra for 2048x2048 |

### Acceptance Criteria -- Phase 7

- [ ] `InferenceRingBuffer` with 3 pinned CPU slots
- [ ] `AsyncFramePipeline` with read/write thread pools
- [ ] CUDA stream isolation (compute + copy streams)
- [ ] MPS graceful fallback (single stream, still async I/O)
- [ ] Pipeline warmup (2-frame fill) and drain (wait for final write)
- [ ] Error handling for corrupt/missing frames
- [ ] Benchmark: wall-clock improvement over sequential loop
- [ ] No race conditions (frame ordering preserved)
- [ ] CLI flag `--async-pipeline / --no-async-pipeline` (default: on)

### Phase 7 Risks

- **GIL contention:** Python GIL limits true parallelism. Mitigated by: I/O threads release GIL (cv2.imread, file writes), GPU kernels release GIL. Only NumPy preprocessing holds GIL briefly.
- **Frame ordering:** Must ensure frame N's write completes before frame N+3's read reuses the same buffer slot. Enforced by `Future.result()` before reusing slot.
- **MPS async behavior:** MPS `non_blocking=True` behavior is less mature than CUDA. Test thoroughly; fallback to synchronous if issues arise.

---

### Phase 8: TorchDynamo Compilation (Graph Fusion)

**Goal:** Compile the model with `torch.compile` to fuse ops, eliminate kernel launch overhead, and enable hardware-specific optimizations. Target: 20-40% speedup on refiner, 10-20% on backbone.

**Files:** `inference_engine.py` (`_load_model`), `model_transformer.py` (potential monkey-patches)

#### 8a. Identify Graph Breaks in Hiera

`timm`'s Hiera uses dynamic shape extraction in window partitioning:

```python
B, H, W, C = x.shape  # Dynamic -- triggers TorchDynamo graph break
```

With `fullgraph=True`, this causes compilation failure. Since our backbone is locked to a fixed resolution (1024x1024 or 2048x2048), we can replace dynamic shapes with static constants.

**Discovery step:** Run `torch._dynamo.explain(model, sample_input)` to enumerate all graph breaks. Document each break and its source.

#### 8b. Monkey-Patch Static Shapes

For each Hiera block that extracts dynamic spatial dims, create a patched version:

```python
def _patch_hiera_static_shapes(model: nn.Module, backbone_size: int):
    """Replace dynamic H/W extraction with static constants for torch.compile."""
    # Calculate expected spatial dims at each Hiera stage
    # Hiera base_plus: patch_size=7, stride=4 -> stage 0: backbone_size/4
    # Each subsequent stage halves spatial dims

    for name, module in model.named_modules():
        if hasattr(module, 'window_partition') or 'HieraBlock' in type(module).__name__:
            # Patch the forward method to use static shapes
            _patch_block_static(module, expected_h, expected_w)
```

**Key constraint:** This only works when backbone_size is fixed. If backbone_size changes between runs, patches must be recomputed. In practice, backbone_size is set once at engine init.

**What to patch:**
1. Window partitioning: `x.reshape(B, H // win_h, win_h, W // win_w, win_w, C)` -- replace `H`, `W` with constants
2. Window unpartitioning: reverse reshape
3. Any `x.shape` extraction used for spatial dimension calculations

#### 8c. Compile the Model

```python
# In _load_model(), after model.half():
if compile_model:
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
```

**`mode="reduce-overhead"`:** Uses CUDA graphs under the hood (CUDA only). On MPS, falls back to default Inductor optimizations.

**`fullgraph=True`:** Strict mode -- fails if any graph break exists. Ensures maximum optimization. If this fails even after patching, fall back to `fullgraph=False` (still beneficial, just with some Python fallback sections).

#### 8d. Warmup Compilation

First inference call triggers JIT compilation (~30-60s). Subsequent calls use cached compiled graph.

```python
# After model creation, run one warmup inference
if compile_model:
    dummy = torch.randn(1, 4, backbone_size, backbone_size, device=device, dtype=torch.float16)
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        _ = model(dummy)
    print("Compilation warmup complete.")
```

Cache the compiled graph to disk via `torch._dynamo.config.cache_size_limit` and environment-based caching.

#### 8e. MPS Considerations

- `torch.compile` MPS backend support is improving but not as mature as CUDA Inductor
- MPS does NOT support CUDA graphs (the `reduce-overhead` mode's main trick)
- Use `mode="default"` on MPS -- still fuses ops via Inductor but without CUDA graphs
- Test for correctness; fall back to eager mode if compile produces wrong results on MPS

#### 8f. Refiner-Only Compilation (Fallback)

If full-model compilation is too fragile (Hiera graph breaks too complex to patch), compile only the CNN refiner:

```python
# Refiner is a straightforward dilated CNN -- no dynamic shapes
self.refiner = torch.compile(self.refiner, mode="reduce-overhead", fullgraph=True)
```

The refiner is 4 residual blocks of 3x3 dilated convs + GroupNorm + ReLU -- the ideal topology for `torch.compile`. No attention, no dynamic shapes, no data-dependent control flow.

### Acceptance Criteria -- Phase 8

- [ ] `torch._dynamo.explain()` run to identify all graph breaks
- [ ] Hiera static shape patches implemented and tested
- [ ] `torch.compile(fullgraph=True)` succeeds on full model (or refiner-only fallback)
- [ ] Warmup compilation integrated into `_load_model`
- [ ] CUDA: `reduce-overhead` mode verified
- [ ] MPS: `default` mode verified (or graceful fallback to eager)
- [ ] Benchmark: per-frame speedup documented
- [ ] Quality gate: compiled output matches eager output (bit-exact or within FP16 tolerance)
- [ ] CLI flag `--compile / --no-compile` (default: off until proven stable)

### Phase 8 Risks

- **Hiera graph breaks too numerous/complex:** Monkey-patching timm internals is fragile and breaks on timm version updates. Mitigate: pin timm version, document patches, implement refiner-only fallback.
- **MPS compile bugs:** MPS Inductor backend may produce incorrect results for some ops. Mitigate: quality gate test; auto-disable compile on MPS if quality drops.
- **Compilation time:** 30-60s warmup is acceptable for batch processing (amortized over hundreds of frames) but annoying for single-frame testing. Mitigate: cache compiled graphs to disk.

---

### Phase 9: Temporal Video Coherence (Feature Caching + Flow Warping)

**Goal:** For video sequences, bypass the heavy Hiera backbone on non-keyframes by warping cached encoder features with optical flow. Target: 5-20x throughput on smooth video with keyframe every 5-20 frames.

**Files:** New `CorridorKeyModule/temporal.py`, modifications to `model_transformer.py` and `clip_manager.py`

#### 9a. Keyframe Strategy

```python
class TemporalCache:
    def __init__(self, keyframe_interval: int = 10):
        self.keyframe_interval = keyframe_interval
        self.cached_features: list[torch.Tensor] | None = None  # Multi-scale encoder features
        self.cached_frame: torch.Tensor | None = None  # Previous frame (for flow)
        self.frame_index: int = 0

    def is_keyframe(self, frame_idx: int) -> bool:
        return frame_idx % self.keyframe_interval == 0
```

**Keyframe (every Nth frame):** Full Hiera backbone + cache multi-scale encoder feature maps in VRAM. These are the 4 feature tensors returned by `self.encoder(x_backbone)` at `model_transformer.py:315`.

**Non-keyframe:** Skip Hiera entirely. Warp cached features to current frame geometry using optical flow.

#### 9b. Optical Flow Estimation

Use `torchvision.models.optical_flow.raft_small` -- lightweight, pretrained, GPU-native:

```python
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

class FlowEstimator:
    def __init__(self, device: torch.device):
        weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=weights).to(device)
        self.model.requires_grad_(False)
        # RAFT small: ~1M params, ~5ms per frame pair at 1024x1024

    @torch.no_grad()
    def estimate(self, prev_frame: torch.Tensor, curr_frame: torch.Tensor) -> torch.Tensor:
        """Returns dense flow field [B, 2, H, W] in pixel units."""
        # RAFT expects [B, 3, H, W] in [0, 255] range
        flow = self.model(prev_frame * 255, curr_frame * 255)[-1]  # Last iteration
        return flow
```

**Why RAFT small:** ~5ms inference at 1024x1024 (negligible vs Hiera's ~1.5s). Pretrained on FlyingChairs/Sintel -- works well for studio footage. No training needed.

**Alternative:** `cv2.calcOpticalFlowFarneback` -- CPU-only, ~50ms, no model weights needed. Good fallback if RAFT dependencies are unwanted.

#### 9c. Feature Warping

Warp cached multi-scale features to align with the current frame:

```python
def warp_features(
    cached_features: list[torch.Tensor],
    flow: torch.Tensor,  # [B, 2, H_flow, W_flow] in pixel units
) -> list[torch.Tensor]:
    """Warp cached encoder features using optical flow."""
    warped = []
    for feat in cached_features:
        _, _, fh, fw = feat.shape
        # Downsample flow to feature resolution
        flow_ds = F.interpolate(flow, size=(fh, fw), mode="bilinear", align_corners=False)
        # Scale flow values proportionally
        flow_ds[:, 0] *= fw / flow.shape[3]
        flow_ds[:, 1] *= fh / flow.shape[2]

        # Convert to normalized grid coordinates [-1, 1]
        # grid_sample expects sampling locations, not displacements
        # backward warp: grid[y,x] = (x + flow_x, y + flow_y) normalized to [-1,1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, fh, device=flow.device),
            torch.linspace(-1, 1, fw, device=flow.device),
            indexing="ij"
        )
        # Add normalized flow displacement
        grid_x = grid_x + flow_ds[:, 0] * 2.0 / fw
        grid_y = grid_y + flow_ds[:, 1] * 2.0 / fh
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, fh, fw, 2]

        warped_feat = F.grid_sample(feat, grid, mode="bilinear", align_corners=False, padding_mode="border")
        warped.append(warped_feat)
    return warped
```

**Backward warping:** For each pixel in the target (current) frame, look up where it came from in the source (previous/keyframe). This avoids holes that forward warping creates.

#### 9d. Modified Forward Pass

```python
def forward(self, x, temporal_cache=None):
    input_size = x.shape[2:]
    rgb = x[:, :3]

    if temporal_cache is not None and not temporal_cache.is_keyframe(temporal_cache.frame_index):
        # NON-KEYFRAME: warp cached features
        flow = self.flow_estimator.estimate(temporal_cache.cached_frame[:, :3], rgb)
        features = warp_features(temporal_cache.cached_features, flow)
    else:
        # KEYFRAME: full backbone
        x_backbone = self._maybe_downsample(x)
        features = self.encoder(x_backbone)
        if temporal_cache is not None:
            temporal_cache.cached_features = [f.detach() for f in features]
            temporal_cache.cached_frame = x.detach()

    # Decode + refine as normal (unchanged)
    alpha_logits = self.alpha_decoder(features)
    ...
```

#### 9e. Adaptive Keyframe Triggers

Beyond fixed interval, force keyframes on:
- **Scene cuts:** High L1 difference between consecutive frames (> threshold)
- **Large motion:** Flow magnitude exceeds threshold (warping becomes unreliable)
- **Quality drift:** Optional quality metric comparing warped vs actual (expensive, off by default)

```python
def should_force_keyframe(self, flow: torch.Tensor, prev_frame: torch.Tensor, curr_frame: torch.Tensor) -> bool:
    # Scene cut detection
    frame_diff = (prev_frame - curr_frame).abs().mean()
    if frame_diff > 0.15:  # ~15% average pixel change
        return True
    # Large motion
    flow_mag = flow.norm(dim=1).mean()
    if flow_mag > 50.0:  # >50px average displacement
        return True
    return False
```

#### 9f. VRAM Budget for Feature Caching

Cached features (Hiera base_plus at 1024x1024 backbone):
- Stage 1: [1, 112, 256, 256] = 7.3 MB (FP16)
- Stage 2: [1, 224, 128, 128] = 7.3 MB
- Stage 3: [1, 448, 64, 64] = 3.7 MB
- Stage 4: [1, 896, 32, 32] = 1.8 MB
- **Total: ~20 MB** -- negligible VRAM cost

Plus RAFT small weights: ~5 MB. Plus previous frame: ~12 MB (1024x1024 FP16).

**Total overhead: ~37 MB** -- well within budget.

### Acceptance Criteria -- Phase 9

- [ ] `TemporalCache` class with keyframe interval + cached features
- [ ] `FlowEstimator` wrapping RAFT small (or Farneback fallback)
- [ ] `warp_features()` correctly warps multi-scale features with grid_sample
- [ ] `GreenFormer.forward()` supports temporal cache path
- [ ] `clip_manager.py` frame loop passes temporal cache between frames
- [ ] Adaptive keyframe triggers (scene cut, large motion)
- [ ] Benchmark: frames/second on video with various keyframe intervals
- [ ] Quality comparison: warped-frame output vs full-inference output (PSNR, SSIM)
- [ ] CLI flags: `--temporal / --no-temporal`, `--keyframe-interval <int>`
- [ ] Temporal flickering assessment (visual inspection of video output)

### Phase 9 Risks

- **Warping artifacts:** Occlusion/disocclusion regions have no valid source pixels. `padding_mode="border"` handles edges but occluded regions get wrong features. Mitigate: force keyframe when occlusion is detected (large flow divergence).
- **Quality drift over time:** Errors accumulate over multiple warped frames. Mitigate: reasonable keyframe interval (5-10), adaptive triggers.
- **RAFT dependency:** Adds `torchvision` dependency and ~5MB model weights. Mitigate: Farneback fallback (OpenCV, already a dependency).
- **Temporal flickering:** Even with correct warping, subtle feature-level differences between keyframe and warped frames can cause flickering in output. Mitigate: blend warped features with a small amount of actual features (if available) at transition points.

---

### Phase 10: Apple Neural Engine Export Script (Optional Utility)

**Goal:** Create `scripts/export_coreml.py` to export the model for Apple's Neural Engine (ANE). M3 Max ANE: 38 TOPS at ~5W. Currently unused during MPS inference.

**Files:** New `scripts/export_coreml.py`, new optional dependency `coremltools`

#### 10a. CoreML Conversion

```python
import coremltools as ct

def export_backbone_coreml(model: GreenFormer, backbone_size: int = 1024):
    # Trace the encoder only (decoder + refiner remain on MPS/CPU)
    encoder = model.encoder

    example_input = torch.randn(1, 4, backbone_size, backbone_size)
    traced = torch.jit.trace(encoder, example_input)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=example_input.shape, name="input")],
        compute_precision=ct.precision.FLOAT32,  # Preserve dynamic range for linear math
        compute_units=ct.ComputeUnit.ALL,  # Allow ANE + GPU + CPU scheduling
    )
    mlmodel.save("CorridorKey_backbone.mlpackage")
```

**Why FLOAT32 precision:** Our color math requires full dynamic range. FP16 on ANE may crush shadows/highlights. Use `FLOAT32` and let CoreML handle internal precision decisions per-op.

#### 10b. Hiera Window Partitioning Fix (5D Layout)

ANE crashes on 6D tensor reshapes. Hiera's window partitioning does:

```python
# Current (6D -- ANE incompatible):
x = x.reshape(B, H // win_h, win_h, W // win_w, win_w, C)
```

Rewrite for ANE:

```python
# Rewritten (5D max -- ANE compatible):
# Fuse batch and window dims
x = x.reshape(B * num_windows, win_h, win_w, C, 1)  # 5D
```

This must be done in the traced/exported graph, not the live model. Implement as a graph transformation pass in the export script.

#### 10c. GroupNorm -> BatchNorm Swap

ANE does not natively support `GroupNorm` -- it falls back to CPU, killing performance. The refiner uses `GroupNorm(8, 64)` in every `RefinerBlock`.

**Weight folding:** At export time, convert GroupNorm to equivalent BatchNorm2d by folding the group statistics:

```python
def groupnorm_to_batchnorm(gn: nn.GroupNorm) -> nn.BatchNorm2d:
    """Convert GroupNorm to BatchNorm2d with folded weights for ANE compatibility."""
    bn = nn.BatchNorm2d(gn.num_channels)
    # Copy affine parameters
    bn.weight.data = gn.weight.data
    bn.bias.data = gn.bias.data
    # Set running stats to identity (no learned stats -- GN is instance-level)
    bn.running_mean.fill_(0.0)
    bn.running_var.fill_(1.0)
    bn.eps = gn.eps
    return bn
```

**Caveat:** This is only valid at batch_size=1 (inference). GroupNorm(8, 64) computes stats over groups of 8 channels per-instance; BatchNorm uses per-channel running stats. At batch=1, they're equivalent if running stats are set to identity. This is an export-time transformation only.

#### 10d. Export Script CLI

```bash
uv run python scripts/export_coreml.py \
    --checkpoint path/to/model.pth \
    --backbone-size 1024 \
    --output CorridorKey_backbone.mlpackage
```

#### 10e. Runtime Integration (Future)

This phase only creates the export script. Runtime integration (loading `.mlpackage` and routing inference through CoreML) would be a separate follow-up. The export script validates:
1. Conversion succeeds without errors
2. CoreML output matches PyTorch output within tolerance
3. ANE utilization confirmed via Xcode Instruments

### Acceptance Criteria -- Phase 10

- [ ] `scripts/export_coreml.py` created
- [ ] Backbone exports to `.mlpackage` successfully
- [ ] `compute_precision=FLOAT32` enforced
- [ ] 6D -> 5D reshape transformation for Hiera windows
- [ ] GroupNorm -> BatchNorm weight folding for refiner
- [ ] Output validation: CoreML vs PyTorch within tolerance
- [ ] `coremltools` added as optional dependency (`uv sync --group coreml`)
- [ ] README/docs updated with export instructions

### Phase 10 Risks

- **CoreML conversion failures:** Hiera uses ops that may not have CoreML equivalents. Mitigate: export backbone and refiner separately; only convert what works.
- **ANE precision:** Even with `FLOAT32` flag, ANE may internally quantize. Mitigate: validate output quality rigorously.
- **Maintenance burden:** CoreML export is a snapshot -- model architecture changes require re-export. Mitigate: keep export script modular, document assumptions.
- **Limited to macOS:** This is Apple-only. No benefit for CUDA users. Mitigate: clearly mark as optional utility.

---

## Phase Dependency Graph

```
Phase 6 (Sparse Tiles) ---+
                           +--- Independent, can be done in any order
Phase 7 (Async I/O) ------+

Phase 8 (torch.compile) ------- After Phase 6 (compile must handle sparse logic)

Phase 9 (Temporal) ------------ After Phase 6 (feature caching interacts with sparse refiner)

Phase 10 (CoreML) ------------- Independent (export script, no runtime changes)
```

**Recommended order:** 6 -> 7 -> 8 -> 9 -> 10

Phases 6 and 7 are independent and could be parallelized. Phase 8 should come after 6 because `torch.compile` needs to handle the sparse tile logic. Phase 9 modifies the forward pass significantly and should be done last among the runtime phases.

## Cumulative Performance Targets

| Phase | Est. Frame Time | Est. VRAM | Key Mechanism |
|-------|----------------|-----------|---------------|
| 4 (current) | 6.35s | 15.36 GB | Baseline for this plan |
| 6 (sparse tiles) | ~3.0-4.0s | 15.36 GB | Skip 70% of refiner tiles |
| 7 (async I/O) | ~2.5-3.5s | +0.15 GB | Hide I/O latency |
| 8 (torch.compile) | ~1.5-2.5s | ~same | Fused kernels, CUDA graphs |
| 9 (temporal, non-KF) | ~0.3-0.8s | +0.04 GB | Skip backbone entirely |
| 10 (CoreML/ANE) | TBD | N/A | ANE offload (Apple only) |

**Sub-1s target achievable** with Phases 6+8+9 on non-keyframes. Keyframes still require full backbone (~1.5-2.5s with compile).

## Open Questions

1. Phase 6: What's the actual skip rate on the reference clip? Need to measure `interest_mask` coverage
2. Phase 7: MPS `non_blocking=True` reliability -- any known bugs with current PyTorch?
3. Phase 8: How many graph breaks does `torch._dynamo.explain()` report for Hiera? Is full-model compile feasible or refiner-only?
4. Phase 9: Acceptable keyframe interval before quality drift is visible? Need A/B test at 5/10/20
5. Phase 9: RAFT small vs Farneback -- latency/quality tradeoff on our footage type?
6. Phase 10: Which Hiera ops lack CoreML equivalents? Need `coremltools` conversion test
7. General: Should Phase 9's temporal mode be opt-in (off by default) given quality risk?
