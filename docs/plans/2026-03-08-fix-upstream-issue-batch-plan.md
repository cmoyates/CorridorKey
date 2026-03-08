---
title: Upstream Issue Batch — Quick Fixes
type: fix
date: 2026-03-08
---

# Upstream Issue Batch — Quick Fixes

## Overview

Batch of 8 low-risk fixes from upstream issues. Each gets its own branch + PR to keep reviews atomic.

## Branch Strategy

Base all branches off `main`. Naming: `fix/<issue-number>-<short-desc>`.

---

## Fix 1: Remove `-x` from pytest addopts (#22)

**Branch:** `fix/22-remove-pytest-failfast`
**File:** `pyproject.toml`
**Change:** Remove `-x` from `addopts = "-x --tb=short"` → `addopts = "--tb=short"`

---

## Fix 2: `torch.load()` add `weights_only=True` (#12)

**Branch:** `fix/12-torch-load-weights-only`
**File:** `CorridorKeyModule/inference_engine.py:65`
**Change:** Add `weights_only=True` to `torch.load()` call. If checkpoint contains non-tensor objects and it fails, switch to `weights_only=False` with security comment.

**Verify:** Run `uv run pytest -m "not gpu"` — checkpoint loading is mocked in tests so this should pass. If GPU available, test with real checkpoint.

---

## Fix 3: `run_inference` name collision (#13)

**Branch:** `fix/13-run-inference-name-collision`
**File:** `clip_manager.py:327`
**Change:**
```python
# Before
from VideoMaMaInferenceModule.inference import load_videomama_model, run_inference
# After
from VideoMaMaInferenceModule.inference import load_videomama_model, run_inference as run_videomama_frames
```
Update call site(s) in `process_videomama_clips()` accordingly.

---

## Fix 4: Narrow warning filters (#23)

**Branch:** `fix/23-narrow-warning-filters`
**File:** `corridorkey_cli.py:47`
**Change:**
```python
# Before
warnings.filterwarnings("ignore")
# After
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
```

---

## Fix 5: Pin torchvision (#16)

**Branch:** `fix/16-pin-torchvision`
**File:** `pyproject.toml`
**Change:** Add `"torchvision==0.21.0"` (or whatever matches `torch==2.10.0`) alongside torch pin. Check current `uv.lock` for resolved version.

**Note:** Must match platform-specific index entries too (cu126 extras). Verify with `uv lock --check` after edit.

---

## Fix 6: opencv-python headless override (#17)

**Branch:** `fix/17-opencv-headless-override`
**File:** `pyproject.toml`
**Change:** Add uv override to prevent headless variant:
```toml
[tool.uv]
override-dependencies = [
    "opencv-python-headless>=0.0.0; extra == 'never'",
]
```
Verify with `uv lock --check`.

---

## Fix 7: Launcher `uv` check (#15)

**Branch:** `fix/15-launcher-uv-check`
**Files:** All `.sh` and `.bat` launchers in root (3 `.sh`, 1 `.bat`)
**Change:** Add guard at top of each script:

`.sh`:
```bash
if ! command -v uv &> /dev/null; then
    echo "[ERROR] uv not installed. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
```

`.bat`:
```batch
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] uv is not installed. Please run Install_CorridorKey_Windows.bat first.
    pause
    exit /b
)
```

---

## Fix 8: Replace `print()` with `logging` (#18)

**Branch:** `fix/18-print-to-logging`
**Files:** `CorridorKeyModule/inference_engine.py`, `CorridorKeyModule/core/model_transformer.py`
**Change:** Add `logger = logging.getLogger(__name__)` to each file. Replace all `print()` calls:
- `print(f"[Warning] ...")` → `logger.warning(...)`
- `print(f"Initializing ...")` → `logger.info(...)`
- Other operational prints → `logger.info(...)` or `logger.debug(...)`

~9 print statements total across both files.

---

## Execution Order

Recommended order (no dependencies between fixes, but complexity-sorted):

1. #22 — 1-line change, zero risk
2. #12 — 1-line change, verify checkpoint compat
3. #13 — 2-line change, rename + update call
4. #23 — 3-line change
5. #16 — 1-line + lockfile
6. #17 — 3-line + lockfile
7. #15 — ~20 lines across 4 files
8. #18 — ~15 lines across 2 files, most mechanical

## Workflow Per Fix

```
git checkout main && git pull
git checkout -b fix/<N>-<desc>
# make changes
uv run ruff check && uv run ruff format --check
uv run pytest -m "not gpu"
git add <files> && git commit
git push -u origin fix/<N>-<desc>
gh pr create --title "Fix #<N>: <desc>" --body "Closes nikopueringer/CorridorKey#<N>"
```

## Unresolved Questions

- #16: What torchvision version pairs with `torch==2.10.0`? Check `uv.lock`.
- #12: Does checkpoint contain non-tensor objects? `weights_only=True` may fail if so.
- #17: Does `[tool.uv]` section already exist in pyproject.toml? If so, merge into it.
- PRs target our fork or upstream? Assuming upstream via fork PRs.
