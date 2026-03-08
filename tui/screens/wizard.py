"""Wizard screen — main interactive TUI for the CorridorKey pipeline."""

from __future__ import annotations

import glob
import logging
import os
import shutil
from dataclasses import dataclass

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
    Switch,
)

from clip_manager import (
    LINUX_MOUNT_ROOT,
    ClipEntry,
    InferenceSettings,
    generate_alphas,
    is_video_file,
    map_path,
    organize_target,
    run_inference,
    run_videomama,
)

logger = logging.getLogger(__name__)

EXCLUDED_DIRS = {"Output", "AlphaHint", "VideoMamaMaskHint", ".ipynb_checkpoints"}


# ---------------------------------------------------------------------------
# Data helpers (extracted from old wizard logic)
# ---------------------------------------------------------------------------


@dataclass
class ClipStatus:
    ready: list[ClipEntry]
    masked: list[ClipEntry]
    raw: list[ClipEntry]


def resolve_path(win_path: str) -> tuple[str, bool]:
    """Resolve a Windows or local path. Returns (resolved_path, is_local)."""
    if os.path.exists(win_path):
        return win_path, True
    process_path = map_path(win_path)
    return process_path, False


def scan_work_dirs(process_path: str) -> list[str]:
    """Discover clip work directories under a path."""
    target_is_shot = os.path.exists(os.path.join(process_path, "Input")) or bool(
        glob.glob(os.path.join(process_path, "Input.*"))
    )
    if target_is_shot:
        return [process_path]
    return [
        os.path.join(process_path, d)
        for d in os.listdir(process_path)
        if os.path.isdir(os.path.join(process_path, d)) and d not in EXCLUDED_DIRS
    ]


def categorize_clips(work_dirs: list[str]) -> ClipStatus:
    """Categorize work dirs into ready/masked/raw."""
    ready: list[ClipEntry] = []
    masked: list[ClipEntry] = []
    raw: list[ClipEntry] = []

    for d in work_dirs:
        entry = ClipEntry(os.path.basename(d), d)
        try:
            entry.find_assets()
        except (FileNotFoundError, ValueError, OSError):
            pass

        has_mask = False
        mask_dir = os.path.join(d, "VideoMamaMaskHint")
        if os.path.isdir(mask_dir) and len(os.listdir(mask_dir)) > 0:
            has_mask = True
        if not has_mask:
            for f in os.listdir(d):
                stem, _ = os.path.splitext(f)
                if stem.lower() == "videomamamaskhint" and is_video_file(f):
                    has_mask = True
                    break

        if entry.alpha_asset:
            ready.append(entry)
        elif has_mask:
            masked.append(entry)
        else:
            raw.append(entry)

    return ClipStatus(ready=ready, masked=masked, raw=raw)


def find_loose_videos(process_path: str) -> list[str]:
    """Find video files sitting directly in the project root."""
    return [f for f in os.listdir(process_path) if is_video_file(f) and os.path.isfile(os.path.join(process_path, f))]


def find_dirs_needing_org(work_dirs: list[str]) -> list[str]:
    """Find directories missing Input/AlphaHint/VideoMamaMaskHint."""
    result = []
    for d in work_dirs:
        has_input = os.path.exists(os.path.join(d, "Input")) or bool(glob.glob(os.path.join(d, "Input.*")))
        has_alpha = os.path.exists(os.path.join(d, "AlphaHint"))
        has_mask = os.path.exists(os.path.join(d, "VideoMamaMaskHint"))
        if not has_input or not has_alpha or not has_mask:
            result.append(d)
    return result


def organize_clips(process_path: str, loose_videos: list[str], work_dirs: list[str]) -> None:
    """Move loose videos into clip folders and create hint directories."""
    for v in loose_videos:
        clip_name = os.path.splitext(v)[0]
        ext = os.path.splitext(v)[1]
        target_folder = os.path.join(process_path, clip_name)

        if os.path.exists(target_folder):
            logger.warning(f"Skipping loose video '{v}': Target folder '{clip_name}' already exists.")
            continue

        try:
            os.makedirs(target_folder)
            target_file = os.path.join(target_folder, f"Input{ext}")
            shutil.move(os.path.join(process_path, v), target_file)
            logger.info(f"Organized: Moved '{v}' to '{clip_name}/Input{ext}'")
            for hint in ["AlphaHint", "VideoMamaMaskHint"]:
                os.makedirs(os.path.join(target_folder, hint), exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to organize video '{v}': {e}")

    for d in work_dirs:
        organize_target(d)


# ---------------------------------------------------------------------------
# Modal screens
# ---------------------------------------------------------------------------


class ConfirmScreen(ModalScreen[bool]):
    """Simple yes/no confirmation dialog."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label(self.message)
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes", variant="success", id="confirm-yes")
                yield Button("No", variant="default", id="confirm-no")

    @on(Button.Pressed, "#confirm-yes")
    def handle_yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def handle_no(self) -> None:
        self.dismiss(False)


class SettingsScreen(ModalScreen[InferenceSettings | None]):
    """Modal for configuring inference settings."""

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-dialog"):
            yield Label("[b]Inference Settings[/b]")

            yield Label("Input Colorspace")
            yield Select(
                [("sRGB", "srgb"), ("Linear", "linear")],
                value="srgb",
                id="colorspace",
            )

            yield Label("Despill Strength (0–10)")
            yield Input(value="5", type="integer", id="despill")

            yield Label("Auto-Despeckle")
            yield Switch(value=True, id="despeckle-toggle")

            yield Label("Despeckle Size (min pixels)")
            yield Input(value="400", type="integer", id="despeckle-size")

            yield Label("Refiner Strength")
            yield Input(value="1.0", type="number", id="refiner")

            with Horizontal(id="settings-buttons"):
                yield Button("Run", variant="success", id="settings-ok")
                yield Button("Cancel", variant="default", id="settings-cancel")

    @on(Button.Pressed, "#settings-ok")
    def handle_ok(self) -> None:
        colorspace = self.query_one("#colorspace", Select).value
        despill_raw = self.query_one("#despill", Input).value
        despeckle_on = self.query_one("#despeckle-toggle", Switch).value
        despeckle_size_raw = self.query_one("#despeckle-size", Input).value
        refiner_raw = self.query_one("#refiner", Input).value

        try:
            despill_int = max(0, min(10, int(despill_raw)))
        except ValueError:
            despill_int = 5

        try:
            despeckle_size = max(0, int(despeckle_size_raw))
        except ValueError:
            despeckle_size = 400

        try:
            refiner_scale = float(refiner_raw)
        except ValueError:
            refiner_scale = 1.0

        self.dismiss(
            InferenceSettings(
                input_is_linear=colorspace == "linear",
                despill_strength=despill_int / 10.0,
                auto_despeckle=despeckle_on,
                despeckle_size=despeckle_size,
                refiner_scale=refiner_scale,
            )
        )

    @on(Button.Pressed, "#settings-cancel")
    def handle_cancel(self) -> None:
        self.dismiss(None)


class ProgressScreen(ModalScreen[None]):
    """Modal showing progress during long-running operations."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title_text = title

    def compose(self) -> ComposeResult:
        with Vertical(id="progress-container"):
            yield Label(self.title_text, id="progress-label")
            yield ProgressBar(id="progress-bar", show_eta=True)
            yield Label("", id="progress-detail")

    def update_progress(self, current: int, total: int, detail: str = "") -> None:
        bar = self.query_one("#progress-bar", ProgressBar)
        bar.update(total=total, progress=current)
        if detail:
            self.query_one("#progress-detail", Label).update(detail)

    def finish(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Wizard screen
# ---------------------------------------------------------------------------


class WizardScreen(Screen):
    """The main wizard screen — status table + action buttons."""

    BINDINGS = [
        ("v", "run_videomama", "VideoMaMa"),
        ("g", "run_gvm", "GVM"),
        ("i", "run_inference", "Inference"),
        ("r", "rescan", "Re-scan"),
        ("o", "organize", "Organize"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, target_path: str, device: str | None = None) -> None:
        super().__init__()
        self.target_path = target_path
        self.device = device
        self.process_path = ""
        self.work_dirs: list[str] = []
        self.clip_status = ClipStatus(ready=[], masked=[], raw=[])

    def compose(self) -> ComposeResult:
        yield Static("CORRIDOR KEY — SMART WIZARD", id="header-panel")
        yield Static("", id="path-info")
        yield DataTable(id="status-table")
        yield RichLog(id="log-panel", highlight=True, markup=True, max_lines=50)
        with Horizontal(id="actions-bar"):
            yield Button("VideoMaMa [v]", id="btn-videomama", variant="warning")
            yield Button("GVM [g]", id="btn-gvm", variant="warning")
            yield Button("Inference [i]", id="btn-inference", variant="success")
            yield Button("Organize [o]", id="btn-organize", variant="default")
            yield Button("Re-scan [r]", id="btn-rescan", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self._resolve_and_scan()

    def _resolve_and_scan(self) -> None:
        """Resolve the target path and perform initial scan."""
        path_info = self.query_one("#path-info", Static)
        resolved, is_local = resolve_path(self.target_path)

        if is_local:
            path_info.update(f"Local path: [bold]{resolved}[/bold]")
        else:
            path_info.update(f"Windows: {self.target_path}\nMapped:  [bold]{resolved}[/bold]")

        if not os.path.exists(resolved):
            path_info.update(
                f"[bold red]ERROR:[/bold red] Path not found!\n"
                f"Tried: {self.target_path}\n"
                f"Mount root: {LINUX_MOUNT_ROOT}"
            )
            return

        self.process_path = resolved
        self._refresh_status()

    def _refresh_status(self) -> None:
        """Re-scan work directories and update the status table."""
        self.work_dirs = scan_work_dirs(self.process_path)
        self.clip_status = categorize_clips(self.work_dirs)
        self._update_table()
        self._update_button_state()
        self._log(f"Scanned {len(self.work_dirs)} clip folders")

    def _update_table(self) -> None:
        """Rebuild the status DataTable."""
        table = self.query_one("#status-table", DataTable)
        table.clear(columns=True)

        table.add_columns("Category", "Count", "Clips")

        ready_names = ", ".join(c.name for c in self.clip_status.ready) or "—"
        masked_names = ", ".join(c.name for c in self.clip_status.masked) or "—"
        raw_names = ", ".join(c.name for c in self.clip_status.raw) or "—"

        table.add_row(
            "[green]Ready[/green] (AlphaHint)",
            str(len(self.clip_status.ready)),
            ready_names,
        )
        table.add_row(
            "[yellow]Masked[/yellow] (VideoMaMaMaskHint)",
            str(len(self.clip_status.masked)),
            masked_names,
        )
        table.add_row(
            "[red]Raw[/red] (Input only)",
            str(len(self.clip_status.raw)),
            raw_names,
        )

    def _update_button_state(self) -> None:
        """Enable/disable action buttons based on clip status."""
        has_missing = bool(self.clip_status.masked) or bool(self.clip_status.raw)
        has_ready = bool(self.clip_status.ready)

        self.query_one("#btn-videomama", Button).disabled = not has_missing
        self.query_one("#btn-gvm", Button).disabled = not has_missing
        self.query_one("#btn-inference", Button).disabled = not has_ready

        loose = find_loose_videos(self.process_path) if self.process_path else []
        needs_org = find_dirs_needing_org(self.work_dirs) if self.work_dirs else []
        self.query_one("#btn-organize", Button).disabled = not (loose or needs_org)

    def _log(self, message: str) -> None:
        """Write a message to the log panel."""
        self.query_one("#log-panel", RichLog).write(message)

    # --- Button handlers ---

    @on(Button.Pressed, "#btn-videomama")
    def on_videomama_pressed(self) -> None:
        self.action_run_videomama()

    @on(Button.Pressed, "#btn-gvm")
    def on_gvm_pressed(self) -> None:
        self.action_run_gvm()

    @on(Button.Pressed, "#btn-inference")
    def on_inference_pressed(self) -> None:
        self.action_run_inference()

    @on(Button.Pressed, "#btn-organize")
    def on_organize_pressed(self) -> None:
        self.action_organize()

    @on(Button.Pressed, "#btn-rescan")
    def on_rescan_pressed(self) -> None:
        self.action_rescan()

    # --- Actions ---

    def action_rescan(self) -> None:
        self._log("Re-scanning…")
        self._refresh_status()

    def action_organize(self) -> None:
        loose = find_loose_videos(self.process_path)
        needs_org = find_dirs_needing_org(self.work_dirs)
        count = len(loose) + len(needs_org)

        def on_confirm(confirmed: bool | None) -> None:
            if confirmed:
                self._do_organize(loose)

        self.app.push_screen(
            ConfirmScreen(f"Organize {count} clips & create hint folders?"),
            on_confirm,
        )

    def _do_organize(self, loose_videos: list[str]) -> None:
        organize_clips(self.process_path, loose_videos, self.work_dirs)
        self._log("[green]Organization complete[/green]")
        self._refresh_status()

    def action_run_videomama(self) -> None:
        missing = self.clip_status.masked + self.clip_status.raw
        if not missing:
            self._log("[yellow]No clips need VideoMaMa[/yellow]")
            return
        self._log(f"Starting VideoMaMa on {len(missing)} clips…")
        self._run_videomama_worker(missing)

    @work(thread=True)
    def _run_videomama_worker(self, clips: list[ClipEntry]) -> None:
        run_videomama(clips, chunk_size=50, device=self.device)
        self.app.call_from_thread(self._on_task_complete, "VideoMaMa")

    def action_run_gvm(self) -> None:
        raw = self.clip_status.raw
        if not raw:
            self._log("[yellow]No raw clips for GVM[/yellow]")
            return

        def on_confirm(confirmed: bool | None) -> None:
            if confirmed:
                self._log(f"Starting GVM on {len(raw)} clips…")
                self._run_gvm_worker(raw)

        self.app.push_screen(
            ConfirmScreen(f"Run GVM auto-matte on {len(raw)} raw clips?"),
            on_confirm,
        )

    @work(thread=True)
    def _run_gvm_worker(self, clips: list[ClipEntry]) -> None:
        generate_alphas(clips, device=self.device)
        self.app.call_from_thread(self._on_task_complete, "GVM")

    def action_run_inference(self) -> None:
        ready = self.clip_status.ready
        if not ready:
            self._log("[yellow]No clips ready for inference[/yellow]")
            return

        def on_settings(settings: InferenceSettings | None) -> None:
            if settings is not None:
                self._log(f"Starting inference on {len(ready)} clips…")
                self._run_inference_worker(ready, settings)

        self.app.push_screen(SettingsScreen(), on_settings)

    @work(thread=True)
    def _run_inference_worker(self, clips: list[ClipEntry], settings: InferenceSettings) -> None:
        def on_clip(name: str, num_frames: int) -> None:
            self.app.call_from_thread(self._log, f"  Processing [bold]{name}[/bold] ({num_frames} frames)")

        def on_frame(idx: int, total: int) -> None:
            pass  # Frame-level updates handled via log if needed

        run_inference(
            clips,
            device=self.device,
            settings=settings,
            on_clip_start=on_clip,
            on_frame_complete=on_frame,
        )
        self.app.call_from_thread(self._on_task_complete, "Inference")

    def _on_task_complete(self, task_name: str) -> None:
        self._log(f"[bold green]{task_name} complete[/bold green]")
        self._refresh_status()

    def action_quit(self) -> None:
        self.app.exit()
