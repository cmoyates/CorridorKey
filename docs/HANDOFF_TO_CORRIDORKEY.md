# Handoff: MLX Backend → CorridorKey Pipeline Optimization

This doc covers everything needed to pick up optimization work in the main CorridorKey repository. It includes methodology, fidelity standards, benchmarking protocols, timing data, and a prioritized list of what to try (and what not to try).

---

## 1. Current performance baseline

| Configuration | 37-frame clip @ 1920×1080 |
|---|---|
| **PyTorch (MPS)** | **3:34** |
| **MLX (current)** | **2:04** |
| **Speedup** | **1.72×** |

### Per-frame breakdown (~3.4s/frame)

| Phase | Time | % of total | Where |
|---|---|---|---|
| **Frame read** (decode + cvtColor + float conversion) | ~200ms est. | ~6% | `clip_manager.py:673-734` |
| **MLX engine.process_frame()** | ~1300ms | ~38% | `engine.py` → tiled inference |
| **Adapter postprocessing** (despill, despeckle, composite, colorspace) | ~500ms est. | ~15% | `backend.py:_wrap_mlx_output` |
| **Frame write** (4× cv2.imwrite — FG/Matte EXR + Comp PNG + Processed EXR) | ~800ms est. | ~24% | `clip_manager.py:757-782` |
| **Python overhead** (loop, GC, array copies) | ~600ms est. | ~17% | scattered |
| **Total** | **~3400ms** | **100%** | |

The "est." values are estimated from the gap between isolated MLX benchmarks and full pipeline timing. The exact split hasn't been instrumented yet — **adding per-phase timing to `run_inference()` should be the first step** in the main repo.

### Tile-level breakdown (15 tiles/frame, ~87ms/tile)

| Component | ms/tile | % |
|---|---|---|
| Backbone (24 blocks) | 37 | 43% |
| Decoders (alpha + fg) | 5 | 6% |
| Refiner (4 dilated ResBlocks, 9 GroupNorms) | 30 | 34% |
| Per-tile overhead (mx.eval, slice, concat) | 15 | 17% |

---

## 2. Optimization approach: go for highest ROI first

**Always try the highest-potential option first.** If it fails fidelity, fall back to the next-highest option. Don't start with "safe" low-impact changes — start with the change that could save the most time, and only retreat to smaller wins if the big ones don't work.

### Prioritized pipeline optimization targets

Listed in order of expected impact. Try #1 first. If it works, move to #2. If #1 fails fidelity, try #2 anyway (they're independent).

#### Priority 1: Async I/O pipeline (~30-40% potential)

**What**: Overlap frame read, inference, and frame write using threads/queues.

Currently `run_inference()` is strictly sequential: read frame → infer → postprocess → write 4 files → repeat. Each step blocks. The MLX inference takes ~1.3s — plenty of time to read the next frame and write previous outputs in parallel.

**Implementation sketch**:
```
Reader thread:  read frame N+2  →  read frame N+3  →  ...
Inference:      infer frame N    →  infer frame N+1 →  ...
Writer thread:  write frame N-1 →  write frame N    →  ...
```

**Where to change**: `clip_manager.py:run_inference()` (L596-825). The frame loop at L673 is the target. Use `queue.Queue` with a `threading.Thread` reader and writer.

**Risk**: Low. No change to model outputs. Fidelity is identical by construction.

#### Priority 2: Reduce output writes (~15-20% potential)

**What**: Skip unnecessary output files.

Currently every frame writes 4 files: FG (EXR), Matte (EXR), Comp (PNG), Processed (EXR). Each `cv2.imwrite` is synchronous and involves CPU compression.

- **Comp** is a preview composite over checkerboard — skip it if the video stitch will regenerate it, or generate it lazily
- **Processed** (premultiplied RGBA EXR) may not be needed for all workflows
- Make output selection configurable: "matte-only", "matte+fg", "all"

**Where to change**: `clip_manager.py:757-782`, add an `enabled_outputs` setting.

**Risk**: None to fidelity. Pure I/O reduction.

#### Priority 3: Streamline adapter postprocessing (~10-15% potential)

**What**: Reduce or defer numpy work in `_wrap_mlx_output`.

The MLX adapter (`backend.py:109-155`) runs 7+ numpy operations per frame:
1. uint8 → float32 conversion
2. `cu.clean_matte()` — morphological ops (dilation, blur)
3. `cu.despill()` — per-pixel green removal
4. `cu.create_checkerboard()` — allocates a full-res checkerboard every frame
5. `cu.composite_straight()` — alpha blending
6. `cu.srgb_to_linear()` / `cu.linear_to_srgb()` — gamma curves
7. `cu.premultiply()` — premultiply for processed output

**Quick wins**:
- Cache the checkerboard (it's the same every frame, just `(w, h)` dependent)
- Skip despill/despeckle/composite if outputs aren't being written
- Move colorspace conversions to MLX (GPU) instead of numpy (CPU)

**Where to change**: `backend.py:_wrap_mlx_output` and `CorridorKeyModule/core/color_utils.py`.

#### Priority 4: Larger tiles (~5-10% potential)

**What**: Use 768 or 1024px tiles instead of 512.

1920×1080 with 512px tiles = 15 tiles/frame. Each tile has ~15ms fixed overhead (mx.eval, slicing, concat). Fewer tiles = less overhead.

| Tile size | Tiles/frame | Fixed overhead/frame | Model compute/frame |
|---|---|---|---|
| 512 | 15 | 225ms | 1080ms |
| 768 | ~7 | 105ms | ~500ms |
| 1024 | ~4 | 60ms | ~290ms |

**Trade-off**: Higher peak memory per tile. At 1024px the refiner's dilated convolutions produce ~9× im2col inflation. On a 16GB Mac this might be tight. On 32GB+ it's fine.

**Where to change**: `backend.py:DEFAULT_MLX_TILE_SIZE` (currently 512). The overlap should stay 128px (2× refiner receptive field).

**Risk**: Medium — need to verify peak memory fits. Fidelity should be identical (same model, same overlap).

#### Priority 5: Video decode optimization (~3-5% potential)

**What**: Use hardware-accelerated video decode instead of `cv2.VideoCapture.read()`.

PyAV (`av` package) is already a dependency and can use VideoToolbox for hardware decode on macOS. Or pre-extract frames to an image sequence before inference starts.

**Where to change**: `clip_manager.py:660-668` (video capture setup) and the per-frame read at L681-687.

---

## 3. Testing methodology

### Experiment protocol

Every optimization follows this loop:

1. **Hypothesis**: What will change, why it should help, expected magnitude
2. **Implement**: Minimal change, one variable at a time
3. **Benchmark**: Time the full 37-frame clip with `time` or internal instrumentation
4. **Fidelity check**: Compare outputs against baseline (see thresholds below)
5. **Decide**: Keep if faster AND fidelity passes. Revert otherwise.
6. **Record**: Log results (timing, fidelity metrics, verdict)

### Benchmarking protocol

**For pipeline timing** (what matters most now):

```bash
# Run the full pipeline and note the time shown in the progress bar
python corridorkey_cli.py run-inference --backend mlx --linear --despill 5 --despeckle --refiner 1.0
```

The progress bar shows total wall time. This is the ground-truth metric.

**For per-phase profiling** (to identify bottlenecks):

Add timing instrumentation inside `run_inference()`:
```python
import time

# Before frame read
t_read_start = time.perf_counter()
# ... read frame ...
t_read = time.perf_counter() - t_read_start

# Before inference
t_infer_start = time.perf_counter()
res = engine.process_frame(...)
t_infer = time.perf_counter() - t_infer_start

# Before write
t_write_start = time.perf_counter()
# ... write outputs ...
t_write = time.perf_counter() - t_write_start
```

**For isolated MLX benchmarks** (if touching model/engine code):

```bash
uv run python scripts/run_research_experiment.py
uv run python scripts/score_experiment.py
```

### Measurement rules

- **Warmup**: Always discard first 1-3 frames (model JIT, cache cold)
- **Metric**: Use median timing, not mean (outlier-resistant)
- **Runs**: Minimum 3 full-pipeline runs for timing claims
- **Environment**: Close other GPU-heavy apps (browser, etc.) during benchmarks
- **Quantification**: Report absolute times AND percentages. "10% faster" is meaningless without "from X ms to Y ms"

---

## 4. Fidelity standards

Fidelity is a hard gate, not an optimization target. Any candidate failing fidelity is rejected regardless of speed gains.

### Tier 1 — Numerical precision changes

For changes that alter precision but not algorithm (dtype changes, compilation, operator fusion):

| Metric | Threshold | Comparison against |
|---|---|---|
| alpha_final max absolute error | < 5e-3 | golden reference (PyTorch) |
| fg_final max absolute error | < 5e-3 | golden reference (PyTorch) |
| No NaN/Inf in any output | — | — |

Golden references: `corridorkey-mlx/reference/fixtures/golden.npz` (512×512) and `golden_2048.npz` (2048×2048).

### Tier 2 — Algorithmic/temporal changes

For changes that alter the computation path (frame skipping, feature reuse, interpolation):

| Metric | Threshold |
|---|---|
| Alpha PSNR vs full pipeline | > 35 dB |
| FG PSNR vs full pipeline | > 33 dB |
| Alpha SSIM vs full pipeline | > 0.97 |
| dtSSD (temporal coherence) | < 1.5 |

### Tier 3 — Pipeline-only changes (I/O, threading, output selection)

For changes that don't touch model outputs at all (async I/O, output selection, decode optimization):

| Metric | Threshold |
|---|---|
| Binary identical outputs | Exact byte-match of output EXR/PNG files |

This is the strictest tier — if the outputs differ at all, the pipeline change introduced a bug. Use `diff` or `cmp` on output files.

### Practical fidelity testing

For quick validation during development:

```python
# Compare two output directories
import cv2, numpy as np, os

def compare_outputs(dir_a, dir_b, subdir="Matte"):
    """Compare output frames between two runs."""
    files_a = sorted(os.listdir(os.path.join(dir_a, subdir)))
    max_diff = 0
    for f in files_a:
        a = cv2.imread(os.path.join(dir_a, subdir, f), cv2.IMREAD_UNCHANGED)
        b = cv2.imread(os.path.join(dir_b, subdir, f), cv2.IMREAD_UNCHANGED)
        if a is None or b is None:
            print(f"Missing: {f}")
            continue
        diff = np.abs(a.astype(float) - b.astype(float)).max()
        max_diff = max(max_diff, diff)
    print(f"Max diff across all frames ({subdir}): {max_diff}")
```

---

## 5. What NOT to try (proven dead ends)

These were all tested and failed. Don't re-attempt without fundamentally new information.

### Temporal optimizations (all failed for matting)

| Approach | Why it fails |
|---|---|
| Output-space EMA blending | Temporal lag on edges at ALL blend values |
| Feature-space EMA blending | Even 90/10 blend → 70.5/255 edge error |
| Backbone feature caching (S2-S3) | S2 features change too much between frames (247/255 max error) |
| Backbone feature caching (S3 only) | Quality safe but only 1.6% speedup — not worth complexity |
| Frame skipping (every other frame) | Visible artifacts on fast motion |

**Root cause**: Matting edges are high-frequency, temporally volatile, and unrecoverable once lost. Any blending, caching, or interpolation smears them.

**The only viable temporal approach** would be optical flow feature warping (warp cached features to match new frame geometry). This was scoped but not implemented due to high complexity.

### Apple Silicon constraints

| Approach | Why it fails |
|---|---|
| Int8 backbone quantization | **11% SLOWER** — dequant overhead > bandwidth savings on unified memory |
| GPU stream parallelism | Apple Silicon has one GPU — no parallel execution across streams |
| Batch processing (B>1) | Linear scaling — GPU fully utilized at B=1 |
| Memory pinning (`mx.set_wired_limit`) | No latency benefit with unified memory |

### Spatial optimizations (all failed for matting)

| Approach | Why it fails |
|---|---|
| Backbone resolution decoupling | Even 12% downscale → 91/255 max edge error |
| Refiner spatial subsampling | Loses edge detail |
| Token deduplication (RLT) | Hiera's windowed attention + small global-attn token counts = nothing to deduplicate |

---

## 6. MLX engine configuration reference

These settings are baked into the MLX engine and should not be overridden by the pipeline:

| Setting | Value | Why |
|---|---|---|
| `MLX_MAX_MB_PER_BUFFER` | 2 | 17% faster — forces frequent eval in tiled workloads |
| `MLX_MAX_OPS_PER_BUFFER` | 2 | Same reason |
| `tile_size` | 512 | Validated with 128px overlap for refiner RF |
| `overlap` | 128 | 2× safety margin over refiner receptive field (65px) |
| `quantize_backbone_stages` | False | Int8 is 11% slower on Apple Silicon |
| `backbone_size` | None | Any downscaling degrades edges |
| `decoder_dtype` | bfloat16 | Halves decoder bandwidth, negligible precision impact |
| `slim` | True | Only returns final outputs (alpha_final, fg_final, alpha_coarse, fg_coarse) |

---

## 7. Repository structure

### corridorkey-mlx (this repo)

The MLX inference backend. Installed as a pip package in the main repo.

| Path | Purpose |
|---|---|
| `src/corridorkey_mlx/engine.py` | Public API: `CorridorKeyMLXEngine` |
| `src/corridorkey_mlx/model/corridorkey.py` | GreenFormer model (backbone + decoders + refiner) |
| `src/corridorkey_mlx/inference/tiling.py` | Tiled inference orchestration |
| `research/OPTIMIZATION_SUMMARY.md` | Full experiment details (57 experiments) |
| `research/experiments.jsonl` | Structured experiment data |
| `research/compound/` | Detailed analysis notes (27 documents) |

### CorridorKey (main repo)

| Path | Purpose | Optimization target? |
|---|---|---|
| `clip_manager.py:run_inference()` | Frame loop — read/infer/write | **YES — primary target** |
| `CorridorKeyModule/backend.py` | Backend factory + MLX adapter | **YES — postprocessing** |
| `CorridorKeyModule/inference_engine.py` | Torch engine | No (MLX path) |
| `CorridorKeyModule/core/color_utils.py` | Despill, composite, colorspace | YES — numpy bottleneck |
| `backend/ffmpeg_tools.py` | Video decode/encode | YES — I/O |
| `corridorkey_cli.py` | CLI entry point | No |

---

## 8. Quick start checklist

1. **Instrument**: Add per-phase timing to `run_inference()` to get exact read/infer/write split
2. **Profile**: Run the 37-frame test clip, collect per-phase timings
3. **Target**: Pick the highest-time phase from profiling
4. **Implement**: One change at a time, following the experiment protocol
5. **Validate**: Compare outputs byte-for-byte (Tier 3 fidelity for pipeline changes)
6. **Benchmark**: Full pipeline timing on the test clip (3 runs, report median)
7. **Record**: Log what you tried, what happened, and why
