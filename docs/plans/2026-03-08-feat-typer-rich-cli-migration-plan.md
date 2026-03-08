---
title: "feat: Migrate CLI from argparse to typer + rich"
type: feat
date: 2026-03-08
issue: "#29"
---

# Migrate CLI from argparse to typer + rich

## Overview

Replace argparse CLI with typer subcommands, extract `input()` prompts from library code into an `InferenceSettings` dataclass, add rich progress bars via callbacks, and style the wizard with rich panels/tables.

## Problem Statement

1. **No progress visibility** — long inference loops show only `print(f"  Frame {i}/{num_frames}...", end="\r")` every 10th frame
2. **Wizard is hard to read** — `print()` with manual `"=" * 60` dividers
3. **CLI not self-documenting** — `--action wizard --win_path` not discoverable
4. **No structured errors** — mix of `logger.error()`, `print()`, `traceback.print_exc()`
5. **Library calls `input()` directly** — `clip_manager.py:509-556` blocks non-interactive use (Nuke, Houdini, batch scripts)

## Proposed Solution

### Before / After

```
Before:  corridorkey --action wizard --win_path "V:\..."
After:   corridorkey wizard "V:\..."

Before:  corridorkey --action run_inference
After:   corridorkey run-inference
```

## Implementation Phases

### Phase 1: Dependencies & InferenceSettings dataclass

**Files:** `pyproject.toml`, `clip_manager.py`

- [x] Add `"typer>=0.12"` and `"rich>=13"` to `pyproject.toml` dependencies
- [x] Run `uv lock` to update lockfile
- [x] Create `InferenceSettings` dataclass in `clip_manager.py`:

```python
# clip_manager.py
from dataclasses import dataclass

@dataclass
class InferenceSettings:
    input_is_linear: bool = False
    despill_strength: float = 0.5       # 0.0–1.0 (mapped from user's 0–10)
    auto_despeckle: bool = True
    despeckle_size: int = 400
    refiner_scale: float = 1.0
```

- [x] Refactor `run_inference()` signature to accept `settings: InferenceSettings | None = None`
- [x] Remove 5 `input()` calls (lines 509–556), use `settings` fields instead
- [x] Default `settings` to `InferenceSettings()` when `None`
- [x] Keep `clip_manager.py` free of typer/rich imports

### Phase 2: Progress callbacks

**Files:** `clip_manager.py`

- [x] Add callback params to `run_inference()`:

```python
def run_inference(
    clips,
    device=None,
    backend=None,
    max_frames=None,
    settings: InferenceSettings | None = None,
    *,
    on_clip_start: Callable[[str, int], None] | None = None,
    on_frame_complete: Callable[[int, int], None] | None = None,
) -> None:
```

- [x] Call `on_clip_start(clip.name, num_frames)` before each clip's frame loop
- [x] Call `on_frame_complete(i, num_frames)` after each frame (replaces `print(f"  Frame {i}/{num_frames}...")`)
- [x] Add same callback pattern to `generate_alphas()` (clip-level)
- [x] Add same callback pattern to `run_videomama()` (clip + chunk level)

### Phase 3: Rewrite `corridorkey_cli.py` with typer

**Files:** `corridorkey_cli.py`

- [x] Replace `argparse` with `typer.Typer()` app
- [x] Create subcommands:

| Old | New subcommand | Function name |
|-----|---------------|---------------|
| `--action list` | `list-clips` | `list_clips()` |
| `--action generate_alphas` | `generate-alphas` | `generate_alphas_cmd()` |
| `--action run_inference` | `run-inference` | `run_inference_cmd()` |
| `--action wizard` | `wizard` | `wizard_cmd()` |

- [x] Shared options via typer callback:

```python
app = typer.Typer(rich_markup_mode="rich")

@app.callback()
def main(
    ctx: typer.Context,
    device: Annotated[str, typer.Option(help="Compute device")] = "auto",
):
    _configure_environment()
    ctx.ensure_object(dict)
    ctx.obj["device"] = resolve_device(device)
```

- [x] `wizard` subcommand takes positional `path` arg:

```python
@app.command()
def wizard(
    ctx: typer.Context,
    path: Annotated[str, typer.Argument(help="Target path (Windows or local)")],
):
    interactive_wizard(path, device=ctx.obj["device"])
```

- [x] `run-inference` subcommand collects settings via `rich.prompt` then passes `InferenceSettings`:

```python
@app.command("run-inference")
def run_inference_cmd(
    ctx: typer.Context,
    backend: str = "auto",
    max_frames: int | None = None,
):
    clips = scan_clips()
    settings = _prompt_inference_settings()  # uses rich.prompt
    run_inference(
        clips,
        device=ctx.obj["device"],
        backend=backend,
        max_frames=max_frames,
        settings=settings,
        on_clip_start=_on_clip_start,
        on_frame_complete=_on_frame_complete,
    )
```

- [x] Entry point stays `corridorkey = "corridorkey_cli:main"` where `main()` calls `_configure_environment()` then `app()`
- [x] Preserve `if __name__ == "__main__": main()` for direct `uv run python corridorkey_cli.py` invocation

### Phase 4: Rich progress & styling

**Files:** `corridorkey_cli.py`

- [x] Model loading: `rich.status` spinner ("Loading CorridorKey engine...")
- [x] Frame processing: `rich.progress.Progress` bar with ETA, rate, frame count
- [x] Implement `_on_clip_start()` / `_on_frame_complete()` callbacks that drive `rich.progress`
- [x] Wizard headers: `rich.panel.Panel` instead of `"=" * 60`
- [x] Status report: `rich.table.Table` (columns: Clip, Status, Frames)
- [x] Interactive prompts: `rich.prompt.Prompt` / `rich.prompt.Confirm` instead of `input()`
- [x] Logging: `rich.logging.RichHandler` so logs don't stomp progress bars
- [x] Configure `_configure_environment()` to use `RichHandler`

### Phase 5: Update launcher scripts

**Files:** 4 launcher scripts

| Script | Old | New |
|--------|-----|-----|
| `CorridorKey_DRAG_CLIPS_HERE_local.sh` | `--action wizard --win_path "$TARGET_PATH"` | `wizard "$TARGET_PATH"` |
| `CorridorKey_DRAG_CLIPS_HERE_local.bat` | `--action wizard --win_path "%WIN_PATH%"` | `wizard "%WIN_PATH%"` |
| `RunInferenceOnly.sh` | `--action run_inference` | `run-inference` |
| `RunGVMOnly.sh` | `--action generate_alphas` | `generate-alphas` |

Also update the scripts to use the `corridorkey` entry point instead of `uv run python corridorkey_cli.py`:
```bash
# Before
uv run python "$LOCAL_SCRIPT" --action wizard --win_path "$TARGET_PATH"
# After
uv run corridorkey wizard "$TARGET_PATH"
```

### Phase 6: Remove `clip_manager.py` `__main__` argparser

**Files:** `clip_manager.py`

- [ ] ~~Remove `if __name__ == "__main__":` block~~ — KEPT as fallback per user request
- [x] The `--backend` and `--max-frames` args move to the `run-inference` subcommand (Phase 3)

### Phase 7: CLI tests

**Files:** `tests/test_cli.py`

- [x] Test each subcommand's `--help` renders without error
- [x] Test invalid arguments produce clean error exit
- [x] Test `InferenceSettings` default values
- [x] Test callback protocol receives expected `(clip_name, num_frames)` and `(frame_idx, num_frames)` calls
- [x] Test `_prompt_inference_settings()` with mocked rich prompts
- [x] Use `typer.testing.CliRunner` for invocation tests

```python
from typer.testing import CliRunner
from corridorkey_cli import app

runner = CliRunner()

def test_list_clips_help():
    result = runner.invoke(app, ["list-clips", "--help"])
    assert result.exit_code == 0
    assert "list" in result.output.lower()

def test_wizard_requires_path():
    result = runner.invoke(app, ["wizard"])
    assert result.exit_code != 0
```

## Acceptance Criteria

- [x] `corridorkey --help` shows all subcommands with descriptions
- [x] `corridorkey wizard "/path"` runs full wizard flow
- [x] `corridorkey run-inference` shows rich progress bar with ETA + frame rate
- [x] `corridorkey list-clips` displays clip table
- [x] `corridorkey generate-alphas` shows clip-level progress
- [x] `clip_manager.py` has zero `input()` calls and zero typer/rich imports
- [x] `InferenceSettings` dataclass usable from external code (Nuke, Houdini)
- [x] `NO_COLOR=1 corridorkey wizard "/path"` works in plain text mode
- [x] All 4 launcher scripts use new subcommand syntax
- [x] `tests/test_cli.py` passes with `uv run pytest tests/test_cli.py`
- [x] Existing tests still pass (`uv run pytest`)

## What Does NOT Change

- `clip_manager.py` stays a pure library (no rich/typer imports)
- `gvm_core/` and `VideoMaMaInferenceModule/` untouched
- No model or inference behavior changes
- Wizard workflow identical (same steps, same options — just styled)
- `tqdm` stays for `gvm_core/` internal use

## Edge Cases & Notes

- **Three `run_inference` functions**: `clip_manager.run_inference()`, `corridorkey_cli.run_inference_cmd()`, `VideoMaMaInferenceModule.inference.run_inference()` — naming must be distinct
- **Wizard `input()` calls** (lines 134, 241, 249, 257, 265 in `corridorkey_cli.py`) also migrate to `rich.prompt`
- **`clip_manager.py` `__main__` block** has `--backend` and `--max-frames` not in `corridorkey_cli.py` — these move to the `run-inference` subcommand
- **Terminal compatibility**: Rich auto-detects; `NO_COLOR=1` disables. Document for SSH-to-render-farm
- **Error handling**: Use `typer.Exit(code=1)` for clean exits, `rich.console.Console().print_exception()` for tracebacks

## Dependencies

Blocked by PRs #4–#11 (uv migration, testing, CI, code quality, CLI extraction, type annotations, test expansion).

## References

- Issue: [#29](https://github.com/nikopueringer/CorridorKey/issues/29)
- `corridorkey_cli.py` — current CLI (328 lines)
- `clip_manager.py:509-556` — `input()` calls to extract
- `clip_manager.py:610-612` — `print()` progress to replace
- Launcher scripts: `CorridorKey_DRAG_CLIPS_HERE_local.{sh,bat}`, `RunInferenceOnly.sh`, `RunGVMOnly.sh`
