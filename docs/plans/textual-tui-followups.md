# Textual TUI — Follow-up Tasks

## ProgressBar widget in WizardScreen
- Wire `on_frame_complete` callback to a ProgressBar widget on the main screen
- Currently only logs clip starts; no per-frame visual progress

## Settings persistence
- SettingsScreen should remember last-used values across wizard runs
- Could store in a JSON/TOML config file or keep in-memory per session

## Convert `run-inference` CLI to Textual
- The standalone `run-inference` command still uses Rich `Prompt.ask` / `Confirm.ask`
- Could launch a lightweight Textual app for interactive mode

## CSS hot-reload for styling
- Use `textual run --dev tui/app.py` during development for live CSS reload
