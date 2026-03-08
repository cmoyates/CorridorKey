"""CorridorKey Textual TUI application."""

from __future__ import annotations

from textual.app import App

from tui.screens.wizard import WizardScreen


class CorridorKeyApp(App):
    """Main TUI application for the CorridorKey pipeline."""

    TITLE = "CorridorKey"
    SUB_TITLE = "Neural Network Green Screen Keying"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def __init__(self, target_path: str, device: str | None = None) -> None:
        super().__init__()
        self.target_path = target_path
        self.device = device

    def on_mount(self) -> None:
        self.push_screen(WizardScreen(self.target_path, self.device))
