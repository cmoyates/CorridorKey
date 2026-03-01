#!/usr/bin/env python3
"""MPS benchmark harness for CorridorKey GreenFormer inference.

Measures sync-aware latency, throughput, peak memory, and output parity
across devices (cpu, mps) and dtypes (float32, float16, bfloat16).

Usage:
    python scripts/benchmark_mps.py --device mps --dtype float16 --img-size 2048 --iterations 10
    python scripts/benchmark_mps.py --parity          # CPU vs MPS output comparison
    python scripts/benchmark_mps.py --all              # full matrix: cpu fp32, mps fp32/fp16/bf16
    python scripts/benchmark_mps.py --phase4           # Phase 4 dtype audit: fp32 no-autocast, fp16, bf16 + parity
    python scripts/benchmark_mps.py --no-autocast      # disable autocast for true fp32 baseline
    python scripts/benchmark_mps.py --phase7           # Phase 7 torch.compile: eager vs refiner-only vs full-model
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Resolve project root so we can import CorridorKeyModule
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from CorridorKeyModule.core.model_transformer import GreenFormer  # noqa: E402
from device_utils import clear_device_cache, memory_snapshot  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_IMG_SIZE = 2048
DEFAULT_ITERATIONS = 10
WARMUP_ITERATIONS = 3
PARITY_ATOL = 1e-3
PARITY_RTOL = 1e-3
ENCODER_NAME = "hiera_base_plus_224.mae_in1k_ft_in1k"
INPUT_CHANNELS = 4  # RGB + coarse alpha


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync(device: torch.device) -> None:
    """Synchronize device before timing."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


MEMORY_LEAK_THRESHOLD = 0.10  # 10% growth flags potential leak


def _build_model(device: torch.device, img_size: int) -> tuple[GreenFormer, float]:
    """Instantiate GreenFormer and return (model, load_time_sec)."""
    _sync(device)
    t0 = time.perf_counter()
    model = GreenFormer(
        encoder_name=ENCODER_NAME,
        in_channels=INPUT_CHANNELS,
        img_size=img_size,
        use_refiner=True,
    )
    model = model.to(device)
    model.eval()
    _sync(device)
    load_time = time.perf_counter() - t0
    return model, load_time


def _make_input(device: torch.device, img_size: int, dtype: torch.dtype) -> torch.Tensor:
    """Create synthetic [1, 4, H, W] input on device."""
    return torch.randn(1, INPUT_CHANNELS, img_size, img_size, device=device, dtype=dtype)


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unknown dtype '{name}'. Options: {list(mapping.keys())}")
    return mapping[name]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _autocast_context(device_type: str, dtype: torch.dtype, enabled: bool):
    """Return autocast context manager, or a no-op if disabled."""
    if enabled:
        return torch.autocast(device_type=device_type, dtype=dtype)
    return contextlib.nullcontext()


def run_benchmark(
    device_str: str,
    dtype_str: str,
    img_size: int,
    iterations: int,
    use_compile: bool,
    verbose: bool = False,
    no_autocast: bool = False,
) -> dict:
    """Run benchmark and return results dict."""
    device = torch.device(device_str)
    autocast_dtype = _resolve_dtype(dtype_str)
    autocast_enabled = not no_autocast

    print(f"\n{'=' * 60}")
    print(f"Benchmark: device={device_str}  dtype={dtype_str}  "
          f"img_size={img_size}  iters={iterations}  compile={use_compile}  "
          f"autocast={autocast_enabled}")
    print(f"{'=' * 60}")

    # --- Model load ---
    model, load_time = _build_model(device, img_size)
    snap_after_load = memory_snapshot(device)
    mem_after_load = snap_after_load["current_alloc_mb"]
    print(f"Model load: {load_time:.2f}s  |  Memory after load: {mem_after_load or 'N/A'} MB")
    if verbose:
        print(f"  Full snapshot: {snap_after_load}")

    if use_compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # --- Prepare input (fp32 — autocast handles downcast) ---
    inp = _make_input(device, img_size, torch.float32)

    # --- Warmup ---
    print(f"Warmup ({WARMUP_ITERATIONS} iterations)...")
    for _ in range(WARMUP_ITERATIONS):
        with torch.inference_mode(), _autocast_context(device.type, autocast_dtype, autocast_enabled):
            _ = model(inp)
        _sync(device)

    snap_after_warmup = memory_snapshot(device)
    mem_after_warmup = snap_after_warmup["current_alloc_mb"]
    print(f"Memory after warmup: {mem_after_warmup or 'N/A'} MB")
    if verbose:
        print(f"  Full snapshot: {snap_after_warmup}")

    # --- Measured iterations ---
    latencies: list[float] = []
    per_iter_memory: list[float | None] = []
    print(f"Measuring ({iterations} iterations)...")
    for i in range(iterations):
        _sync(device)
        t0 = time.perf_counter()
        with torch.inference_mode(), _autocast_context(device.type, autocast_dtype, autocast_enabled):
            _ = model(inp)
        _sync(device)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        iter_mem = memory_snapshot(device)["current_alloc_mb"]
        per_iter_memory.append(iter_mem)
        if verbose and iter_mem is not None:
            print(f"  iter {i}: {elapsed:.4f}s  mem={iter_mem:.1f} MB")

    # Coarse-boundary cache clear after all measured iterations
    clear_device_cache(device)

    peak_snap = memory_snapshot(device)
    peak_mem = peak_snap["driver_alloc_mb"]

    # --- Leak detection ---
    leak_warning = None
    valid_mems = [m for m in per_iter_memory if m is not None]
    if len(valid_mems) >= 2:
        first_mem = valid_mems[0]
        last_mem = valid_mems[-1]
        if first_mem > 0:
            growth_ratio = (last_mem - first_mem) / first_mem
            if growth_ratio > MEMORY_LEAK_THRESHOLD:
                leak_warning = (
                    f"Potential memory leak: {first_mem:.1f} MB -> {last_mem:.1f} MB "
                    f"({growth_ratio:.1%} growth over {iterations} iterations)"
                )
                print(f"  WARNING: {leak_warning}")

    # --- Compute stats ---
    latencies_sorted = sorted(latencies)
    median_lat = statistics.median(latencies)
    mean_lat = statistics.mean(latencies)
    p95_idx = max(0, int(len(latencies_sorted) * 0.95) - 1)
    p95_lat = latencies_sorted[p95_idx]
    min_lat = latencies_sorted[0]
    max_lat = latencies_sorted[-1]
    throughput = 1.0 / median_lat if median_lat > 0 else 0.0

    results = {
        "device": device_str,
        "dtype": dtype_str,
        "autocast": autocast_enabled,
        "img_size": img_size,
        "iterations": iterations,
        "compile": use_compile,
        "model_load_sec": round(load_time, 3),
        "warmup_iterations": WARMUP_ITERATIONS,
        "latency_median_sec": round(median_lat, 4),
        "latency_mean_sec": round(mean_lat, 4),
        "latency_p95_sec": round(p95_lat, 4),
        "latency_min_sec": round(min_lat, 4),
        "latency_max_sec": round(max_lat, 4),
        "throughput_fps": round(throughput, 2),
        "memory_after_load_mb": round(mem_after_load, 1) if mem_after_load else None,
        "memory_after_warmup_mb": round(mem_after_warmup, 1) if mem_after_warmup else None,
        "peak_memory_mb": round(peak_mem, 1) if peak_mem else None,
        "per_iter_memory_mb": [round(m, 1) if m is not None else None for m in per_iter_memory],
        "leak_warning": leak_warning,
        "all_latencies_sec": [round(lat, 4) for lat in latencies],
    }

    # --- Print summary ---
    print("\n--- Results ---")
    print(f"  Median latency:   {median_lat:.4f}s")
    print(f"  Mean latency:     {mean_lat:.4f}s")
    print(f"  P95 latency:      {p95_lat:.4f}s")
    print(f"  Min/Max:          {min_lat:.4f}s / {max_lat:.4f}s")
    print(f"  Throughput:       {throughput:.2f} fps")
    print(f"  Peak memory:      {peak_mem or 'N/A'} MB")
    if leak_warning:
        print(f"  LEAK WARNING:     {leak_warning}")

    return results


# ---------------------------------------------------------------------------
# Parity check
# ---------------------------------------------------------------------------

def run_parity(
    target_device: str,
    dtype_str: str,
    img_size: int,
) -> dict:
    """Compare model outputs between CPU fp32 (reference) and target device."""
    print(f"\n{'=' * 60}")
    print(f"Parity check: CPU fp32 vs {target_device} {dtype_str}  img_size={img_size}")
    print(f"{'=' * 60}")

    autocast_dtype = _resolve_dtype(dtype_str)

    # --- Reference: CPU fp32 (build once, share weights) ---
    cpu_device = torch.device("cpu")
    model_cpu, _ = _build_model(cpu_device, img_size)

    # Use fixed seed for reproducibility
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    inp_cpu = torch.randn(1, INPUT_CHANNELS, img_size, img_size, generator=gen)

    with torch.inference_mode():
        ref_out = model_cpu(inp_cpu)
    ref_alpha = ref_out["alpha"].clone()
    ref_fg = ref_out["fg"].clone()

    # --- Target device (copy same weights to target) ---
    target = torch.device(target_device)
    import copy
    model_target = copy.deepcopy(model_cpu).to(target)
    model_target.eval()
    _sync(target)
    inp_target = inp_cpu.to(target)

    use_autocast = autocast_dtype != torch.float32
    with torch.inference_mode(), _autocast_context(target.type, autocast_dtype, use_autocast):
        target_out = model_target(inp_target)
    _sync(target)
    target_alpha = target_out["alpha"].float().cpu()
    target_fg = target_out["fg"].float().cpu()

    # --- Compare ---
    def _check(name: str, ref: torch.Tensor, tgt: torch.Tensor) -> dict:
        has_nan = torch.isnan(tgt).any().item()
        has_inf = torch.isinf(tgt).any().item()
        close = torch.allclose(ref, tgt, atol=PARITY_ATOL, rtol=PARITY_RTOL)
        max_diff = (ref - tgt).abs().max().item()
        mean_diff = (ref - tgt).abs().mean().item()

        status = "PASS" if (close and not has_nan and not has_inf) else "FAIL"
        print(f"  {name}: {status}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  "
              f"NaN={has_nan}  Inf={has_inf}  allclose={close}")

        return {
            "name": name,
            "status": status,
            "max_abs_diff": round(max_diff, 6),
            "mean_abs_diff": round(mean_diff, 6),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "allclose": close,
        }

    alpha_result = _check("alpha", ref_alpha, target_alpha)
    fg_result = _check("fg", ref_fg, target_fg)

    overall_pass = alpha_result["status"] == "PASS" and fg_result["status"] == "PASS"
    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'}")

    return {
        "reference": "cpu_fp32",
        "target_device": target_device,
        "target_dtype": dtype_str,
        "img_size": img_size,
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "overall_pass": overall_pass,
        "checks": [alpha_result, fg_result],
    }


# ---------------------------------------------------------------------------
# Phase 5: Memory-traffic benchmark
# ---------------------------------------------------------------------------

PHASE5_TRANSFER_ITERS = 20


def run_transfer_benchmark(
    device_str: str,
    img_size: int,
    iterations: int = PHASE5_TRANSFER_ITERS,
) -> dict:
    """Measure CPU→device and device→CPU transfer costs, blocking vs non_blocking."""
    device = torch.device(device_str)

    print(f"\n{'=' * 60}")
    print(f"Phase 5 Transfer Benchmark: {device_str}  img_size={img_size}  iters={iterations}")
    print(f"{'=' * 60}")

    # Simulate real inference_engine input: [1, 4, H, W] fp32 on CPU
    inp_cpu = torch.randn(1, INPUT_CHANNELS, img_size, img_size)

    results = {}

    # --- CPU→device: blocking vs non_blocking ---
    for non_blocking in (False, True):
        label = "non_blocking" if non_blocking else "blocking"
        latencies = []
        for _ in range(iterations):
            _sync(device)
            t0 = time.perf_counter()
            _ = inp_cpu.to(device, non_blocking=non_blocking)
            _sync(device)
            latencies.append(time.perf_counter() - t0)

        median_lat = statistics.median(latencies)
        results[f"to_device_{label}_median_ms"] = round(median_lat * 1000, 3)
        results[f"to_device_{label}_p95_ms"] = round(
            sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] * 1000, 3
        )
        print(f"  CPU→{device_str} ({label}): median={median_lat*1000:.3f}ms")

    # --- device→CPU: .cpu().numpy() (the real post-process path) ---
    # Create output-sized tensors on device (model output shape)
    out_alpha = torch.randn(1, 1, img_size, img_size, device=device)
    out_fg = torch.randn(1, 3, img_size, img_size, device=device)

    latencies_transfer_back = []
    for _ in range(iterations):
        _sync(device)
        t0 = time.perf_counter()
        _ = out_alpha[0].permute(1, 2, 0).cpu().numpy()
        _ = out_fg[0].permute(1, 2, 0).cpu().numpy()
        elapsed = time.perf_counter() - t0
        latencies_transfer_back.append(elapsed)

    median_back = statistics.median(latencies_transfer_back)
    results["to_cpu_numpy_median_ms"] = round(median_back * 1000, 3)
    results["to_cpu_numpy_p95_ms"] = round(
        sorted(latencies_transfer_back)[max(0, int(len(latencies_transfer_back) * 0.95) - 1)] * 1000, 3
    )
    print(f"  {device_str}→CPU .cpu().numpy(): median={median_back*1000:.3f}ms")

    # --- End-to-end: full inference with CPU input (blocking vs non_blocking) ---
    model, _ = _build_model(device, img_size)

    for non_blocking in (False, True):
        label = "non_blocking" if non_blocking else "blocking"

        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
                inp_dev = inp_cpu.to(device, non_blocking=non_blocking)
                out = model(inp_dev)
            _sync(device)

        # Measured
        latencies = []
        for _ in range(iterations):
            _sync(device)
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
                inp_dev = inp_cpu.to(device, non_blocking=non_blocking)
                out = model(inp_dev)
                # Include the cpu().numpy() transfer in end-to-end
                _ = out["alpha"][0].permute(1, 2, 0).cpu().numpy()
                _ = out["fg"][0].permute(1, 2, 0).cpu().numpy()
            _sync(device)
            latencies.append(time.perf_counter() - t0)

        median_e2e = statistics.median(latencies)
        p95_e2e = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
        fps = 1.0 / median_e2e if median_e2e > 0 else 0.0

        results[f"e2e_{label}_median_sec"] = round(median_e2e, 4)
        results[f"e2e_{label}_p95_sec"] = round(p95_e2e, 4)
        results[f"e2e_{label}_fps"] = round(fps, 2)
        print(f"  E2E ({label}): median={median_e2e:.4f}s  p95={p95_e2e:.4f}s  fps={fps:.2f}")

    results["device"] = device_str
    results["img_size"] = img_size
    results["iterations"] = iterations

    # Non-blocking delta
    blocking_e2e = results["e2e_blocking_median_sec"]
    nb_e2e = results["e2e_non_blocking_median_sec"]
    if blocking_e2e > 0:
        delta_pct = ((blocking_e2e - nb_e2e) / blocking_e2e) * 100
        results["non_blocking_delta_pct"] = round(delta_pct, 2)
        print(f"\n  non_blocking delta: {delta_pct:+.2f}% ({'faster' if delta_pct > 0 else 'slower/neutral'})")

    return results


# ---------------------------------------------------------------------------
# Phase 6: channels_last benchmark
# ---------------------------------------------------------------------------

CHANNELS_LAST_CONFIGS = ["baseline", "refiner_only", "full_model"]


def _apply_channels_last(model: GreenFormer, config: str) -> None:
    """Apply channels_last memory format to model components."""
    if config == "refiner_only":
        if model.refiner is not None:
            model.refiner = model.refiner.to(memory_format=torch.channels_last)
    elif config == "full_model":
        model = model.to(memory_format=torch.channels_last)
    # "baseline" — no changes


def run_channels_last_benchmark(
    device_str: str,
    img_size: int,
    iterations: int,
    verbose: bool = False,
) -> list[dict]:
    """Benchmark channels_last: baseline vs refiner-only vs full model."""
    device = torch.device(device_str)
    all_results = []

    for config in CHANNELS_LAST_CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"Phase 6 channels_last: config={config}  device={device_str}  "
              f"img_size={img_size}  iters={iterations}")
        print(f"{'=' * 60}")

        # Fresh model each config to avoid format contamination
        model, load_time = _build_model(device, img_size)

        # Apply channels_last
        _apply_channels_last(model, config)

        # Prepare input
        inp = _make_input(device, img_size, torch.float32)
        if config == "full_model":
            inp = inp.to(memory_format=torch.channels_last)

        # Warmup
        print(f"Warmup ({WARMUP_ITERATIONS} iterations)...")
        for _ in range(WARMUP_ITERATIONS):
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
                _ = model(inp)
            _sync(device)

        snap_after_warmup = memory_snapshot(device)

        # Measured iterations
        latencies: list[float] = []
        print(f"Measuring ({iterations} iterations)...")
        for i in range(iterations):
            _sync(device)
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
                _ = model(inp)
            _sync(device)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            if verbose:
                print(f"  iter {i}: {elapsed:.4f}s")

        clear_device_cache(device)
        peak_snap = memory_snapshot(device)

        median_lat = statistics.median(latencies)
        p95_idx = max(0, int(len(latencies) * 0.95) - 1)
        p95_lat = sorted(latencies)[p95_idx]
        throughput = 1.0 / median_lat if median_lat > 0 else 0.0

        result = {
            "config": config,
            "device": device_str,
            "img_size": img_size,
            "iterations": iterations,
            "latency_median_sec": round(median_lat, 4),
            "latency_p95_sec": round(p95_lat, 4),
            "throughput_fps": round(throughput, 2),
            "peak_memory_mb": round(peak_snap["driver_alloc_mb"], 1) if peak_snap["driver_alloc_mb"] else None,
            "memory_after_warmup_mb": round(snap_after_warmup["current_alloc_mb"], 1) if snap_after_warmup["current_alloc_mb"] else None,
            "all_latencies_sec": [round(lat, 4) for lat in latencies],
        }

        print(f"  Median: {median_lat:.4f}s  P95: {p95_lat:.4f}s  FPS: {throughput:.2f}  "
              f"Peak: {result['peak_memory_mb'] or 'N/A'} MB")

        all_results.append(result)

    # Compute deltas vs baseline
    baseline_median = all_results[0]["latency_median_sec"]
    for r in all_results:
        if baseline_median > 0:
            delta = ((baseline_median - r["latency_median_sec"]) / baseline_median) * 100
            r["delta_vs_baseline_pct"] = round(delta, 2)
        else:
            r["delta_vs_baseline_pct"] = 0.0

    return all_results


def run_channels_last_parity(
    device_str: str,
    img_size: int,
) -> list[dict]:
    """Verify channels_last produces identical outputs to baseline."""
    device = torch.device(device_str)
    all_parity = []

    # Build reference model (baseline, no channels_last)
    model_ref, _ = _build_model(device, img_size)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    inp_cpu = torch.randn(1, INPUT_CHANNELS, img_size, img_size, generator=gen)
    inp = inp_cpu.to(device)

    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
        ref_out = model_ref(inp)
    _sync(device)
    ref_alpha = ref_out["alpha"].float().cpu()
    ref_fg = ref_out["fg"].float().cpu()

    for config in CHANNELS_LAST_CONFIGS[1:]:  # skip baseline
        print(f"\nParity check: {config} vs baseline")
        import copy
        model_test = copy.deepcopy(model_ref)
        _apply_channels_last(model_test, config)

        test_inp = inp.clone()
        if config == "full_model":
            test_inp = test_inp.to(memory_format=torch.channels_last)

        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
            test_out = model_test(test_inp)
        _sync(device)
        test_alpha = test_out["alpha"].float().cpu()
        test_fg = test_out["fg"].float().cpu()

        alpha_max_diff = (ref_alpha - test_alpha).abs().max().item()
        fg_max_diff = (ref_fg - test_fg).abs().max().item()
        alpha_close = torch.allclose(ref_alpha, test_alpha, atol=PARITY_ATOL, rtol=PARITY_RTOL)
        fg_close = torch.allclose(ref_fg, test_fg, atol=PARITY_ATOL, rtol=PARITY_RTOL)

        status = "PASS" if (alpha_close and fg_close) else "FAIL"
        print(f"  {config}: {status}  alpha_diff={alpha_max_diff:.6f}  fg_diff={fg_max_diff:.6f}")

        all_parity.append({
            "config": config,
            "status": status,
            "alpha_max_diff": round(alpha_max_diff, 6),
            "fg_max_diff": round(fg_max_diff, 6),
            "alpha_close": alpha_close,
            "fg_close": fg_close,
        })

    return all_parity


# ---------------------------------------------------------------------------
# Phase 7: torch.compile benchmark
# ---------------------------------------------------------------------------

COMPILE_CONFIGS = [
    # (label, compile_target, compile_mode)
    ("eager", None, None),
    ("refiner_default", "refiner", "default"),
    ("refiner_reduce_overhead", "refiner", "reduce-overhead"),
    ("full_default", "full", "default"),
    ("full_reduce_overhead", "full", "reduce-overhead"),
]


def _apply_compile(model: GreenFormer, target: str | None, mode: str | None) -> GreenFormer:
    """Apply torch.compile to model component. Returns model (may be replaced)."""
    if target is None:
        return model  # eager baseline
    if target == "refiner":
        if model.refiner is not None:
            model.refiner = torch.compile(model.refiner, mode=mode)
        return model
    if target == "full":
        return torch.compile(model, mode=mode)
    raise ValueError(f"Unknown compile target: {target}")


def _count_graph_breaks(
    model: GreenFormer,
    inp: torch.Tensor,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> int:
    """Run one forward pass with dynamo logging to count graph breaks."""
    import logging

    graph_break_count = 0
    original_level = logging.getLogger("torch._dynamo").level

    class GraphBreakCounter(logging.Handler):
        def emit(self, record):
            nonlocal graph_break_count
            if "graph break" in record.getMessage().lower():
                graph_break_count += 1

    handler = GraphBreakCounter()
    dynamo_logger = logging.getLogger("torch._dynamo")
    dynamo_logger.addHandler(handler)
    dynamo_logger.setLevel(logging.DEBUG)

    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
            _ = model(inp)
        _sync(device)
    finally:
        dynamo_logger.removeHandler(handler)
        dynamo_logger.setLevel(original_level)

    return graph_break_count


def run_compile_benchmark(
    device_str: str,
    img_size: int,
    iterations: int,
    verbose: bool = False,
) -> list[dict]:
    """Benchmark torch.compile: eager vs refiner-only vs full model."""
    device = torch.device(device_str)
    autocast_dtype = torch.float16
    all_results = []

    for label, compile_target, compile_mode in COMPILE_CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"Phase 7 torch.compile: config={label}  device={device_str}  "
              f"img_size={img_size}  iters={iterations}")
        print(f"{'=' * 60}")

        # Fresh model each config
        model, load_time = _build_model(device, img_size)

        # Compilation
        compile_time = 0.0
        graph_breaks = 0
        if compile_target is not None:
            print(f"Compiling ({compile_target}, mode={compile_mode})...")
            t0 = time.perf_counter()
            try:
                model = _apply_compile(model, compile_target, compile_mode)
            except Exception as exc:
                print(f"  COMPILE FAILED: {exc}")
                all_results.append({
                    "config": label,
                    "compile_target": compile_target,
                    "compile_mode": compile_mode,
                    "status": "compile_failed",
                    "error": str(exc),
                })
                continue

            # First forward triggers actual compilation
            inp_warmup = _make_input(device, img_size, torch.float32)
            try:
                with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    _ = model(inp_warmup)
                _sync(device)
            except Exception as exc:
                print(f"  FIRST FORWARD FAILED: {exc}")
                all_results.append({
                    "config": label,
                    "compile_target": compile_target,
                    "compile_mode": compile_mode,
                    "status": "forward_failed",
                    "error": str(exc),
                })
                continue

            compile_time = time.perf_counter() - t0
            print(f"  Compilation time: {compile_time:.2f}s")

            # Count graph breaks (rebuild model for clean detection)
            try:
                model_for_breaks, _ = _build_model(device, img_size)
                model_for_breaks = _apply_compile(model_for_breaks, compile_target, compile_mode)
                torch._dynamo.reset()
                graph_breaks = _count_graph_breaks(
                    model_for_breaks, inp_warmup, device, autocast_dtype
                )
                print(f"  Graph breaks: {graph_breaks}")
                del model_for_breaks
                clear_device_cache(device)
            except Exception as exc:
                print(f"  Graph break detection failed: {exc}")
                graph_breaks = -1

        # Prepare input
        inp = _make_input(device, img_size, torch.float32)

        # Warmup
        print(f"Warmup ({WARMUP_ITERATIONS} iterations)...")
        for _ in range(WARMUP_ITERATIONS):
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
                _ = model(inp)
            _sync(device)

        snap_after_warmup = memory_snapshot(device)

        # Measured iterations
        latencies: list[float] = []
        print(f"Measuring ({iterations} iterations)...")
        for i in range(iterations):
            _sync(device)
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
                _ = model(inp)
            _sync(device)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            if verbose:
                print(f"  iter {i}: {elapsed:.4f}s")

        clear_device_cache(device)
        peak_snap = memory_snapshot(device)

        median_lat = statistics.median(latencies)
        p95_idx = max(0, int(len(latencies) * 0.95) - 1)
        p95_lat = sorted(latencies)[p95_idx]
        throughput = 1.0 / median_lat if median_lat > 0 else 0.0

        result = {
            "config": label,
            "compile_target": compile_target,
            "compile_mode": compile_mode,
            "status": "ok",
            "device": device_str,
            "img_size": img_size,
            "iterations": iterations,
            "compile_time_sec": round(compile_time, 2),
            "graph_breaks": graph_breaks,
            "latency_median_sec": round(median_lat, 4),
            "latency_p95_sec": round(p95_lat, 4),
            "throughput_fps": round(throughput, 2),
            "peak_memory_mb": round(peak_snap["driver_alloc_mb"], 1) if peak_snap["driver_alloc_mb"] else None,
            "memory_after_warmup_mb": round(snap_after_warmup["current_alloc_mb"], 1) if snap_after_warmup["current_alloc_mb"] else None,
            "all_latencies_sec": [round(lat, 4) for lat in latencies],
        }

        print(f"  Median: {median_lat:.4f}s  P95: {p95_lat:.4f}s  FPS: {throughput:.2f}  "
              f"Compile: {compile_time:.2f}s  Breaks: {graph_breaks}  "
              f"Peak: {result['peak_memory_mb'] or 'N/A'} MB")

        all_results.append(result)

    # Compute deltas vs eager baseline
    eager_results = [r for r in all_results if r["config"] == "eager" and r.get("status") == "ok"]
    eager_median = eager_results[0]["latency_median_sec"] if eager_results else 0
    for r in all_results:
        if r.get("status") == "ok" and eager_median > 0:
            delta = ((eager_median - r["latency_median_sec"]) / eager_median) * 100
            r["delta_vs_eager_pct"] = round(delta, 2)
        else:
            r["delta_vs_eager_pct"] = 0.0

    return all_results


def run_compile_parity(
    device_str: str,
    img_size: int,
) -> list[dict]:
    """Verify compiled models produce same outputs as eager baseline."""
    device = torch.device(device_str)
    autocast_dtype = torch.float16
    all_parity = []

    # Build eager reference
    model_ref, _ = _build_model(device, img_size)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    inp_cpu = torch.randn(1, INPUT_CHANNELS, img_size, img_size, generator=gen)
    inp = inp_cpu.to(device)

    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
        ref_out = model_ref(inp)
    _sync(device)
    ref_alpha = ref_out["alpha"].float().cpu()
    ref_fg = ref_out["fg"].float().cpu()

    # Test each compile config (skip eager)
    for label, compile_target, compile_mode in COMPILE_CONFIGS[1:]:
        print(f"\nParity check: {label} vs eager")
        import copy
        model_test = copy.deepcopy(model_ref)
        try:
            model_test = _apply_compile(model_test, compile_target, compile_mode)

            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
                test_out = model_test(inp)
            _sync(device)
            test_alpha = test_out["alpha"].float().cpu()
            test_fg = test_out["fg"].float().cpu()

            alpha_max_diff = (ref_alpha - test_alpha).abs().max().item()
            fg_max_diff = (ref_fg - test_fg).abs().max().item()
            alpha_close = torch.allclose(ref_alpha, test_alpha, atol=PARITY_ATOL, rtol=PARITY_RTOL)
            fg_close = torch.allclose(ref_fg, test_fg, atol=PARITY_ATOL, rtol=PARITY_RTOL)
            has_nan = torch.isnan(test_alpha).any().item() or torch.isnan(test_fg).any().item()

            status = "PASS" if (alpha_close and fg_close and not has_nan) else "FAIL"
            print(f"  {label}: {status}  alpha_diff={alpha_max_diff:.6f}  "
                  f"fg_diff={fg_max_diff:.6f}  NaN={has_nan}")

            all_parity.append({
                "config": label,
                "status": status,
                "alpha_max_diff": round(alpha_max_diff, 6),
                "fg_max_diff": round(fg_max_diff, 6),
                "alpha_close": alpha_close,
                "fg_close": fg_close,
                "has_nan": has_nan,
            })
        except Exception as exc:
            print(f"  {label}: ERROR — {exc}")
            all_parity.append({
                "config": label,
                "status": "ERROR",
                "error": str(exc),
            })

        del model_test
        clear_device_cache(device)

    return all_parity


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def format_markdown(results: list[dict], parity: list[dict] | None = None) -> str:
    """Format results as a markdown summary table."""
    lines = ["# MPS Benchmark Results\n"]

    if results:
        lines.append("## Latency & Throughput\n")
        lines.append("| Device | dtype | Autocast | Size | Median (s) | P95 (s) | FPS | Peak Mem (MB) | Compile |")
        lines.append("|--------|-------|----------|------|-----------|---------|-----|---------------|---------|")
        for r in results:
            peak = r.get("peak_memory_mb") or "N/A"
            autocast = r.get("autocast", True)
            lines.append(
                f"| {r['device']} | {r['dtype']} | {autocast} | {r['img_size']} | "
                f"{r['latency_median_sec']:.4f} | {r['latency_p95_sec']:.4f} | "
                f"{r['throughput_fps']:.2f} | {peak} | {r['compile']} |"
            )
        lines.append("")

    if parity:
        lines.append("## Parity Checks\n")
        lines.append("| Target | dtype | Size | Alpha | FG | Max Diff |")
        lines.append("|--------|-------|------|-------|-----|----------|")
        for p in parity:
            alpha_check = p["checks"][0]
            fg_check = p["checks"][1]
            max_diff = max(alpha_check["max_abs_diff"], fg_check["max_abs_diff"])
            lines.append(
                f"| {p['target_device']} | {p['target_dtype']} | {p['img_size']} | "
                f"{alpha_check['status']} | {fg_check['status']} | {max_diff:.6f} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPS benchmark for CorridorKey GreenFormer")
    parser.add_argument("--device", type=str, default="mps", help="Device: cpu, mps, cuda")
    parser.add_argument("--dtype", type=str, default="float16", help="Autocast dtype: float32, float16, bfloat16")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Input image size (square)")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Measured iterations")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
    parser.add_argument("--no-autocast", action="store_true", help="Disable autocast (true fp32 baseline)")
    parser.add_argument("--parity", action="store_true", help="Run parity check (CPU fp32 vs target)")
    parser.add_argument("--all", action="store_true", help="Run full benchmark matrix")
    parser.add_argument("--phase4", action="store_true", help="Phase 4 dtype audit: fp32 no-autocast, fp16, bf16 + parity")
    parser.add_argument("--phase5", action="store_true", help="Phase 5 memory-traffic: non_blocking transfer + cpu().numpy() cost")
    parser.add_argument("--phase6", action="store_true", help="Phase 6 channels_last: refiner-only vs full model vs baseline")
    parser.add_argument("--phase7", action="store_true", help="Phase 7 torch.compile: eager vs refiner-only vs full-model")
    parser.add_argument("--verbose", action="store_true", help="Per-iteration memory logging")
    parser.add_argument("--output", type=str, default=None, help="Write JSON results to file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_results: list[dict] = []
    all_parity: list[dict] = []

    def _mps_available() -> bool:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    def _run_config(dev: str, dt: str, comp: bool, no_ac: bool = False) -> dict | None:
        if dev == "mps" and not _mps_available():
            print(f"Skipping {dev} — not available")
            return None
        if dev == "cuda" and not torch.cuda.is_available():
            print(f"Skipping {dev} — not available")
            return None
        try:
            return run_benchmark(
                dev, dt, args.img_size, args.iterations, comp,
                verbose=args.verbose, no_autocast=no_ac,
            )
        except RuntimeError as exc:
            print(f"SKIPPED {dev}/{dt}: {exc}")
            return None

    if args.phase7:
        # Phase 7: torch.compile experiment
        if not _mps_available():
            print("ERROR: Phase 7 requires MPS device")
            sys.exit(1)

        compile_results = run_compile_benchmark(
            "mps", args.img_size, args.iterations, verbose=args.verbose
        )
        compile_parity = run_compile_parity("mps", args.img_size)

        # Markdown report
        report_lines = ["# Phase 7: torch.compile Results\n"]
        report_lines.append(f"**Device:** mps  **Size:** {args.img_size}  **Iters:** {args.iterations}\n")
        report_lines.append("## Latency\n")
        report_lines.append("| Config | Median (s) | P95 (s) | FPS | Peak Mem (MB) | Compile (s) | Graph Breaks | Delta |")
        report_lines.append("|--------|-----------|---------|-----|---------------|-------------|-------------|-------|")
        for r in compile_results:
            if r.get("status") != "ok":
                report_lines.append(
                    f"| {r['config']} | — | — | — | — | — | — | {r.get('status', 'error')}: {r.get('error', '')[:50]} |"
                )
                continue
            delta = r.get("delta_vs_eager_pct", 0)
            peak = r.get("peak_memory_mb") or "N/A"
            report_lines.append(
                f"| {r['config']} | {r['latency_median_sec']} | {r['latency_p95_sec']} | "
                f"{r['throughput_fps']} | {peak} | {r['compile_time_sec']} | "
                f"{r['graph_breaks']} | {delta:+.2f}% |"
            )
        report_lines.append("")
        report_lines.append("## Parity (vs eager)\n")
        report_lines.append("| Config | Status | Alpha Diff | FG Diff | NaN |")
        report_lines.append("|--------|--------|-----------|---------|-----|")
        for p in compile_parity:
            if p.get("status") == "ERROR":
                report_lines.append(f"| {p['config']} | ERROR | — | — | — |")
                continue
            report_lines.append(
                f"| {p['config']} | {p['status']} | {p.get('alpha_max_diff', 'N/A')} | "
                f"{p.get('fg_max_diff', 'N/A')} | {p.get('has_nan', 'N/A')} |"
            )
        report_lines.append("")

        report = "\n".join(report_lines)
        print(f"\n{report}")

        output_data = {"phase7_compile": compile_results, "phase7_parity": compile_parity}
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(json.dumps(output_data, indent=2))
            print(f"Results written to {output_path}")
        else:
            default_path = PROJECT_ROOT / "scripts" / "phase7_results.json"
            default_path.write_text(json.dumps(output_data, indent=2))
            print(f"Results written to {default_path}")
        return

    elif args.phase6:
        # Phase 6: channels_last experiment
        if not _mps_available():
            print("ERROR: Phase 6 requires MPS device")
            sys.exit(1)

        cl_results = run_channels_last_benchmark(
            "mps", args.img_size, args.iterations, verbose=args.verbose
        )
        cl_parity = run_channels_last_parity("mps", args.img_size)

        # Markdown report
        report_lines = ["# Phase 6: channels_last Results\n"]
        report_lines.append(f"**Device:** mps  **Size:** {args.img_size}  **Iters:** {args.iterations}\n")
        report_lines.append("## Latency\n")
        report_lines.append("| Config | Median (s) | P95 (s) | FPS | Peak Mem (MB) | Delta |")
        report_lines.append("|--------|-----------|---------|-----|---------------|-------|")
        for r in cl_results:
            delta = r.get("delta_vs_baseline_pct", 0)
            peak = r.get("peak_memory_mb") or "N/A"
            report_lines.append(
                f"| {r['config']} | {r['latency_median_sec']} | {r['latency_p95_sec']} | "
                f"{r['throughput_fps']} | {peak} | {delta:+.2f}% |"
            )
        report_lines.append("")
        report_lines.append("## Parity (vs baseline)\n")
        report_lines.append("| Config | Status | Alpha Diff | FG Diff |")
        report_lines.append("|--------|--------|-----------|---------|")
        for p in cl_parity:
            report_lines.append(
                f"| {p['config']} | {p['status']} | {p['alpha_max_diff']:.6f} | {p['fg_max_diff']:.6f} |"
            )
        report_lines.append("")

        report = "\n".join(report_lines)
        print(f"\n{report}")

        output_data = {"phase6_channels_last": cl_results, "phase6_parity": cl_parity}
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(json.dumps(output_data, indent=2))
            print(f"Results written to {output_path}")
        else:
            default_path = PROJECT_ROOT / "scripts" / "phase6_results.json"
            default_path.write_text(json.dumps(output_data, indent=2))
            print(f"Results written to {default_path}")
        return

    elif args.phase5:
        # Phase 5: memory-traffic / non_blocking transfer benchmark
        if not _mps_available():
            print("ERROR: Phase 5 requires MPS device")
            sys.exit(1)
        transfer_results = run_transfer_benchmark("mps", args.img_size, args.iterations or 20)
        output_data = {"phase5_transfer": transfer_results}

        report_lines = ["# Phase 5: Memory-Traffic Results\n"]
        report_lines.append(f"**Device:** mps  **Size:** {args.img_size}  **Iters:** {transfer_results['iterations']}\n")
        report_lines.append("## Transfer Costs\n")
        report_lines.append("| Direction | Mode | Median (ms) | P95 (ms) |")
        report_lines.append("|-----------|------|-------------|----------|")
        report_lines.append(f"| CPU→MPS | blocking | {transfer_results['to_device_blocking_median_ms']} | {transfer_results['to_device_blocking_p95_ms']} |")
        report_lines.append(f"| CPU→MPS | non_blocking | {transfer_results['to_device_non_blocking_median_ms']} | {transfer_results['to_device_non_blocking_p95_ms']} |")
        report_lines.append(f"| MPS→CPU | .cpu().numpy() | {transfer_results['to_cpu_numpy_median_ms']} | {transfer_results['to_cpu_numpy_p95_ms']} |")
        report_lines.append("")
        report_lines.append("## End-to-End (fp16 autocast, includes transfers)\n")
        report_lines.append("| Mode | Median (s) | P95 (s) | FPS | Delta |")
        report_lines.append("|------|-----------|---------|-----|-------|")
        delta = transfer_results.get("non_blocking_delta_pct", 0)
        report_lines.append(f"| blocking | {transfer_results['e2e_blocking_median_sec']} | {transfer_results['e2e_blocking_p95_sec']} | {transfer_results['e2e_blocking_fps']} | baseline |")
        report_lines.append(f"| non_blocking | {transfer_results['e2e_non_blocking_median_sec']} | {transfer_results['e2e_non_blocking_p95_sec']} | {transfer_results['e2e_non_blocking_fps']} | {delta:+.2f}% |")
        report_lines.append("")

        report = "\n".join(report_lines)
        print(f"\n{report}")

        if args.output:
            output_path = Path(args.output)
            output_path.write_text(json.dumps(output_data, indent=2))
            print(f"Results written to {output_path}")
        else:
            default_path = PROJECT_ROOT / "scripts" / "phase5_results.json"
            default_path.write_text(json.dumps(output_data, indent=2))
            print(f"Results written to {default_path}")
        return

    elif args.phase4:
        # Phase 4 dtype audit matrix
        # (device, dtype, compile, no_autocast)
        configs = [
            ("mps", "float32", False, True),   # fp32 no-autocast control
            ("mps", "float16", False, False),   # fp16 autocast (current default)
            ("mps", "bfloat16", False, False),  # bf16 autocast (test support)
        ]
        for dev, dt, comp, no_ac in configs:
            result = _run_config(dev, dt, comp, no_ac)
            if result is not None:
                all_results.append(result)

        # Parity checks for each dtype
        if _mps_available():
            for dt in ("float32", "float16", "bfloat16"):
                try:
                    parity = run_parity("mps", dt, args.img_size)
                    all_parity.append(parity)
                except RuntimeError as exc:
                    print(f"SKIPPED parity {dt}: {exc}")

    elif args.all:
        # Full matrix
        configs = [
            ("cpu", "float32", False, False),
            ("mps", "float32", False, False),
            ("mps", "float16", False, False),
            ("mps", "bfloat16", False, False),
        ]
        for dev, dt, comp, no_ac in configs:
            result = _run_config(dev, dt, comp, no_ac)
            if result is not None:
                all_results.append(result)

        # Parity for MPS configs
        if _mps_available():
            for dt in ("float32", "float16", "bfloat16"):
                try:
                    parity = run_parity("mps", dt, args.img_size)
                    all_parity.append(parity)
                except RuntimeError as exc:
                    print(f"SKIPPED parity {dt}: {exc}")

    elif args.parity:
        parity = run_parity(args.device, args.dtype, args.img_size)
        all_parity.append(parity)

    else:
        result = run_benchmark(
            args.device, args.dtype, args.img_size, args.iterations, args.compile,
            verbose=args.verbose, no_autocast=args.no_autocast,
        )
        all_results.append(result)

    # --- Output ---
    report = format_markdown(all_results, all_parity if all_parity else None)
    print(f"\n{report}")

    output_data = {"benchmarks": all_results, "parity": all_parity}

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"Results written to {output_path}")
    else:
        # Always write to default location
        default_path = PROJECT_ROOT / "scripts" / "benchmark_results.json"
        default_path.write_text(json.dumps(output_data, indent=2))
        print(f"Results written to {default_path}")


if __name__ == "__main__":
    main()
