#!/usr/bin/env python3
"""MPS environment diagnostics for CorridorKey.

Prints system, Python, PyTorch, and MPS capability information
to help debug and baseline Apple Silicon performance work.
"""

import os
import platform
import subprocess
import sys

import torch


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "unavailable"


def main() -> None:
    sep = "-" * 60

    # --- System ---
    print(sep)
    print("SYSTEM")
    print(sep)
    mac_ver = platform.mac_ver()[0]
    print(f"  macOS version:    {mac_ver or 'N/A (not macOS)'}")
    print(f"  Platform:         {platform.platform()}")
    print(f"  Machine:          {platform.machine()}")
    print(f"  Processor:        {platform.processor() or 'Apple Silicon' if platform.machine() == 'arm64' else platform.processor()}")

    # Chip name via sysctl (Apple Silicon only)
    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    print(f"  CPU brand:        {chip}")

    # Memory
    try:
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        print(f"  Physical memory:  {mem_bytes / (1024**3):.1f} GB")
    except Exception:
        print("  Physical memory:  unavailable")

    # --- Python ---
    print(f"\n{sep}")
    print("PYTHON")
    print(sep)
    print(f"  Version:          {sys.version}")
    print(f"  Executable:       {sys.executable}")

    # --- PyTorch ---
    print(f"\n{sep}")
    print("PYTORCH")
    print(sep)
    print(f"  Version:          {torch.__version__}")
    print(f"  Debug build:      {torch.version.debug}")
    print(f"  CUDA available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version:     {torch.version.cuda}")
        print(f"  CUDA device:      {torch.cuda.get_device_name(0)}")

    # --- MPS ---
    print(f"\n{sep}")
    print("MPS (Metal Performance Shaders)")
    print(sep)
    mps_built = torch.backends.mps.is_built() if hasattr(torch.backends, "mps") else False
    mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
    print(f"  Built with MPS:   {mps_built}")
    print(f"  MPS available:    {mps_available}")

    if mps_available:
        dev = torch.device("mps")
        # Memory APIs (PyTorch 2.1+)
        try:
            print(f"  Current alloc:    {torch.mps.current_allocated_memory() / (1024**2):.1f} MB")
        except Exception:
            print("  Current alloc:    API unavailable")
        try:
            print(f"  Driver alloc:     {torch.mps.driver_allocated_memory() / (1024**2):.1f} MB")
        except Exception:
            print("  Driver alloc:     API unavailable")
        try:
            print(f"  Recommended max:  {torch.mps.recommended_max_memory() / (1024**3):.2f} GB")
        except Exception:
            print("  Recommended max:  API unavailable")

        # Quick smoke test
        try:
            t = torch.randn(2, 2, device=dev)
            result = (t @ t.T).sum().item()
            print(f"  Smoke test:       PASS (matmul result={result:.4f})")
        except Exception as e:
            print(f"  Smoke test:       FAIL ({e})")

    # --- Default dtype ---
    print(f"\n{sep}")
    print("DTYPE DEFAULTS")
    print(sep)
    print(f"  torch default:    {torch.get_default_dtype()}")

    # --- Relevant env vars ---
    print(f"\n{sep}")
    print("MPS ENVIRONMENT VARIABLES")
    print(sep)
    mps_env_vars = [
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO",
        "PYTORCH_MPS_FAST_MATH",
        "PYTORCH_MPS_PREFER_METAL",
        "PYTORCH_DEBUG_MPS_ALLOCATOR",
        "CORRIDORKEY_DEVICE",
    ]
    for var in mps_env_vars:
        val = os.environ.get(var)
        print(f"  {var}: {val if val is not None else '(not set)'}")

    # --- Key dependencies ---
    print(f"\n{sep}")
    print("KEY DEPENDENCIES")
    print(sep)
    for pkg in ["timm", "cv2", "numpy", "diffusers", "transformers", "accelerate"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  {pkg:20s} {ver}")
        except ImportError:
            print(f"  {pkg:20s} NOT INSTALLED")

    print(f"\n{sep}")
    print("REPORT COMPLETE")
    print(sep)


if __name__ == "__main__":
    main()
