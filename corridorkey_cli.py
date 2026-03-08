"""CorridorKey command-line interface and interactive wizard.

This module handles CLI subcommands, environment setup, and the
interactive wizard workflow. The pipeline logic lives in clip_manager.py,
which can be imported independently as a library.

Usage:
    uv run corridorkey wizard "V:\\..."
    uv run corridorkey run-inference
    uv run corridorkey generate-alphas
    uv run corridorkey list-clips
"""

from __future__ import annotations

import logging
import sys
import warnings
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, IntPrompt, Prompt

from clip_manager import (
    InferenceSettings,
    generate_alphas,
    run_inference,
    scan_clips,
)
from device_utils import resolve_device

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="corridorkey",
    help="Neural network green screen keying for professional VFX pipelines.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------


def _configure_environment() -> None:
    """Set up logging and warnings for interactive CLI use."""
    warnings.filterwarnings("ignore")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


# ---------------------------------------------------------------------------
# Progress helpers (callback protocol → rich.progress)
# ---------------------------------------------------------------------------

_progress: Progress | None = None
_frame_task_id: int | None = None


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def _on_clip_start(clip_name: str, num_frames: int) -> None:
    """Callback: reset the progress bar for a new clip."""
    global _frame_task_id
    if _progress is not None:
        # Remove old task if any
        if _frame_task_id is not None:
            _progress.remove_task(_frame_task_id)
        _frame_task_id = _progress.add_task(f"[cyan]{clip_name}", total=num_frames)


def _on_frame_complete(frame_idx: int, num_frames: int) -> None:
    """Callback: advance the progress bar by one frame."""
    if _progress is not None and _frame_task_id is not None:
        _progress.advance(_frame_task_id)


def _on_clip_start_simple(clip_name: str, total_clips: int) -> None:
    """Callback for clip-level progress (generate-alphas)."""
    console.print(f"  Processing [bold]{clip_name}[/bold] ({total_clips} total)")


# ---------------------------------------------------------------------------
# Inference settings prompt (rich.prompt — CLI layer only)
# ---------------------------------------------------------------------------


def _prompt_inference_settings(
    *,
    default_linear: bool | None = None,
    default_despill: int | None = None,
    default_despeckle: bool | None = None,
    default_despeckle_size: int | None = None,
    default_refiner: float | None = None,
) -> InferenceSettings:
    """Interactively prompt for inference settings, skipping any pre-filled values."""
    console.print(Panel("Inference Settings", style="bold cyan"))

    # 1. Gamma
    if default_linear is not None:
        input_is_linear = default_linear
    else:
        gamma_choice = Prompt.ask(
            "Input colorspace",
            choices=["linear", "srgb"],
            default="srgb",
        )
        input_is_linear = gamma_choice == "linear"

    # 2. Despill
    if default_despill is not None:
        despill_int = max(0, min(10, default_despill))
    else:
        despill_int = IntPrompt.ask(
            "Despill strength (0–10, 10 = max despill)",
            default=5,
        )
        despill_int = max(0, min(10, despill_int))
    despill_strength = despill_int / 10.0

    # 3. Auto-despeckle
    if default_despeckle is not None:
        auto_despeckle = default_despeckle
    else:
        auto_despeckle = Confirm.ask(
            "Enable auto-despeckle (removes tracking dots)?",
            default=True,
        )

    despeckle_size = default_despeckle_size if default_despeckle_size is not None else 400
    if auto_despeckle and default_despeckle_size is None and default_despeckle is None:
        despeckle_size = IntPrompt.ask(
            "Despeckle size (min pixels for a spot)",
            default=400,
        )
        despeckle_size = max(0, despeckle_size)

    # 4. Refiner strength
    if default_refiner is not None:
        refiner_scale = default_refiner
    else:
        refiner_val = Prompt.ask(
            "Refiner strength multiplier [dim](experimental)[/dim]",
            default="1.0",
        )
        try:
            refiner_scale = float(refiner_val)
        except ValueError:
            refiner_scale = 1.0

    return InferenceSettings(
        input_is_linear=input_is_linear,
        despill_strength=despill_strength,
        auto_despeckle=auto_despeckle,
        despeckle_size=despeckle_size,
        refiner_scale=refiner_scale,
    )


# ---------------------------------------------------------------------------
# Typer callback (shared options)
# ---------------------------------------------------------------------------


@app.callback()
def app_callback(
    ctx: typer.Context,
    device: Annotated[
        str,
        typer.Option(help="Compute device: auto, cuda, mps, cpu"),
    ] = "auto",
) -> None:
    """Neural network green screen keying for professional VFX pipelines."""
    _configure_environment()
    ctx.ensure_object(dict)
    ctx.obj["device"] = resolve_device(device)
    logger.info(f"Using device: {ctx.obj['device']}")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command("list-clips")
def list_clips_cmd(ctx: typer.Context) -> None:
    """List all clips in ClipsForInference and their status."""
    scan_clips()


@app.command("generate-alphas")
def generate_alphas_cmd(ctx: typer.Context) -> None:
    """Generate coarse alpha hints via GVM for clips missing them."""
    clips = scan_clips()
    with console.status("[bold green]Loading GVM model..."):
        generate_alphas(clips, device=ctx.obj["device"], on_clip_start=_on_clip_start_simple)
    console.print("[bold green]Alpha generation complete.")


@app.command("run-inference")
def run_inference_cmd(
    ctx: typer.Context,
    backend: Annotated[
        str,
        typer.Option(help="Inference backend: auto, torch, mlx"),
    ] = "auto",
    max_frames: Annotated[
        Optional[int],
        typer.Option("--max-frames", help="Limit frames per clip"),
    ] = None,
    linear: Annotated[
        Optional[bool],
        typer.Option("--linear/--srgb", help="Input colorspace (default: prompt)"),
    ] = None,
    despill: Annotated[
        Optional[int],
        typer.Option("--despill", help="Despill strength 0–10 (default: prompt)"),
    ] = None,
    despeckle: Annotated[
        Optional[bool],
        typer.Option("--despeckle/--no-despeckle", help="Auto-despeckle toggle (default: prompt)"),
    ] = None,
    despeckle_size: Annotated[
        Optional[int],
        typer.Option("--despeckle-size", help="Min pixel size for despeckle (default: prompt)"),
    ] = None,
    refiner: Annotated[
        Optional[float],
        typer.Option("--refiner", help="Refiner strength multiplier (default: prompt)"),
    ] = None,
) -> None:
    """Run CorridorKey inference on clips with Input + AlphaHint.

    Settings can be passed as flags for non-interactive use, or omitted to
    prompt interactively.
    """
    clips = scan_clips()

    # If all settings provided via flags, skip prompts entirely
    all_flags_set = all(v is not None for v in [linear, despill, despeckle, refiner])
    if all_flags_set:
        despill_clamped = max(0, min(10, despill))  # type: ignore[arg-type]
        settings = InferenceSettings(
            input_is_linear=linear,  # type: ignore[arg-type]
            despill_strength=despill_clamped / 10.0,
            auto_despeckle=despeckle,  # type: ignore[arg-type]
            despeckle_size=despeckle_size if despeckle_size is not None else 400,
            refiner_scale=refiner,  # type: ignore[arg-type]
        )
    else:
        settings = _prompt_inference_settings(
            default_linear=linear,
            default_despill=despill,
            default_despeckle=despeckle,
            default_despeckle_size=despeckle_size,
            default_refiner=refiner,
        )

    global _progress
    progress = _make_progress()
    _progress = progress

    with progress:
        run_inference(
            clips,
            device=ctx.obj["device"],
            backend=backend,
            max_frames=max_frames,
            settings=settings,
            on_clip_start=_on_clip_start,
            on_frame_complete=_on_frame_complete,
        )

    _progress = None
    console.print("[bold green]Inference complete.")


@app.command()
def wizard(
    ctx: typer.Context,
    path: Annotated[str, typer.Argument(help="Target path (Windows or local)")],
) -> None:
    """Interactive wizard for organizing clips and running the pipeline."""
    from tui.app import CorridorKeyApp

    tui_app = CorridorKeyApp(target_path=path, device=ctx.obj["device"])
    tui_app.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point called by the `corridorkey` console script."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
