#!/usr/bin/env python3
"""MPS benchmark harness for CorridorKey GreenFormer inference.

Measures sync-aware latency, throughput, peak memory, and output parity
across devices (cpu, mps) and dtypes (float32, float16).

Usage:
    python scripts/benchmark_mps.py --device mps --dtype float16 --img-size 2048 --iterations 10
    python scripts/benchmark_mps.py --parity          # CPU vs MPS output comparison
    python scripts/benchmark_mps.py --all              # full matrix: cpu fp32, mps fp32, mps fp16
"""

from __future__ import annotations

import argparse
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

def run_benchmark(
    device_str: str,
    dtype_str: str,
    img_size: int,
    iterations: int,
    use_compile: bool,
    verbose: bool = False,
) -> dict:
    """Run benchmark and return results dict."""
    device = torch.device(device_str)
    autocast_dtype = _resolve_dtype(dtype_str)

    print(f"\n{'=' * 60}")
    print(f"Benchmark: device={device_str}  dtype={dtype_str}  "
          f"img_size={img_size}  iters={iterations}  compile={use_compile}")
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
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
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
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
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

    # --- Reference: CPU fp32 ---
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

    # --- Target device ---
    target = torch.device(target_device)
    model_target, _ = _build_model(target, img_size)
    inp_target = inp_cpu.to(target)

    with torch.inference_mode(), torch.autocast(device_type=target.type, dtype=autocast_dtype):
        target_out = model_target(inp_target)
    _sync(target)
    target_alpha = target_out["alpha"].cpu()
    target_fg = target_out["fg"].cpu()

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
# Markdown report
# ---------------------------------------------------------------------------

def format_markdown(results: list[dict], parity: list[dict] | None = None) -> str:
    """Format results as a markdown summary table."""
    lines = ["# MPS Benchmark Results\n"]

    if results:
        lines.append("## Latency & Throughput\n")
        lines.append("| Device | dtype | Size | Median (s) | P95 (s) | FPS | Peak Mem (MB) | Compile |")
        lines.append("|--------|-------|------|-----------|---------|-----|---------------|---------|")
        for r in results:
            peak = r.get("peak_memory_mb") or "N/A"
            lines.append(
                f"| {r['device']} | {r['dtype']} | {r['img_size']} | "
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
    parser.add_argument("--parity", action="store_true", help="Run parity check (CPU fp32 vs target)")
    parser.add_argument("--all", action="store_true", help="Run full benchmark matrix")
    parser.add_argument("--verbose", action="store_true", help="Per-iteration memory logging")
    parser.add_argument("--output", type=str, default=None, help="Write JSON results to file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_results: list[dict] = []
    all_parity: list[dict] = []

    if args.all:
        # Full matrix
        configs = [
            ("cpu", "float32", False),
            ("mps", "float32", False),
            ("mps", "float16", False),
        ]
        for dev, dt, comp in configs:
            # Skip unavailable devices
            if dev == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                print(f"Skipping {dev} — not available")
                continue
            if dev == "cuda" and not torch.cuda.is_available():
                print(f"Skipping {dev} — not available")
                continue

            result = run_benchmark(dev, dt, args.img_size, args.iterations, comp, verbose=args.verbose)
            all_results.append(result)

        # Parity for MPS configs
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            for dt in ("float32", "float16"):
                parity = run_parity("mps", dt, args.img_size)
                all_parity.append(parity)

    elif args.parity:
        parity = run_parity(args.device, args.dtype, args.img_size)
        all_parity.append(parity)

    else:
        result = run_benchmark(args.device, args.dtype, args.img_size, args.iterations, args.compile, verbose=args.verbose)
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
