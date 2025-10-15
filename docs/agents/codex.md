# Repository Guidelines

## Project Structure & Module Organization
- `core/` hosts the CSV analytics pipeline (`analyzer.py`, `signal_tools.py`, `report_manager.py`) and should remain UI-agnostic; extend functionality here first.
- `ui/` contains the Textual desktop shell (`main_app.py`) and Tkinter adapter; Streamlit runs from `web_app.py`, while CLI entry points live in `main.py` and `desktop_app_tk.py`.
- `tests/` mirrors the core layout with fixtures and profiling scripts; `tests_csv/` stores curated sample files used across scenarios.
- Runtime artifacts land in `outputs/`, visual exports under `outputs/visual_reports/`, and logs rotate in `logs/`; clean up bulky outputs before committing.

## Build, Test & Development Commands
- Install dependencies with `python -m pip install -r requirements.txt` inside a Python 3.10+ virtual environment.
- Launch the Streamlit UI via `streamlit run web_app.py`; desktop flows use `python desktop_app_tk.py`, while the Textual TUI runs through `python main.py`.
- Execute the automated suite using `pytest`; append `-m "not slow"` when iterating locally, or `--maxfail=1` for focused debugging.
- Generate fresh coverage HTML after changes with `pytest --cov=core --cov-report=html` and inspect `htmlcov/index.html`.

## Coding Style & Naming Conventions
- Follow standard PEP 8 with 4-space indentation, descriptive type-hinted APIs, and keep business logic inside `core/`.
- Prefer explicit, Italian-friendly naming already present (e.g., `CsvAnalyzer`, `resolve_fs`) and keep private helpers underscored.
- Centralize logging through `core/logger.py` rather than creating ad-hoc loggers.
- When introducing configuration, wire it through `config.json` accessors instead of scattering literals.

## Testing Guidelines
- Pytest is configured via `pytest.ini` to target `tests/`, enforce 80% coverage, and fail on marker misuse; respect existing markers (`slow`, `integration`, `unit`, `synthetic`).
- New tests should mirror module names (`test_signal_tools.py`) and reuse fixtures from `tests/fixtures`.
- Place reusable CSVs in `tests/data/` or `tests_csv/` and annotate intent within the test for traceability.

## Commit & Pull Request Guidelines
- Recent history shows short, imperative titles (e.g., `Fix bug FFT`, `Optimize CSV loading performance`) with optional Italian context; keep summaries under ~72 characters.
- Reference issues inline (`Fix reset button… (Issue #44)`) and batch related refactors within the same commit when practical.
- Pull requests should include: 1) a concise problem/solution summary, 2) validation notes (`pytest`, manual UI path), and 3) screenshots or GIFs when the UI changes.
- Request review whenever logic touches `core/` analytics or alters shared resources (`config.json`, `tests_csv/`), even for seemingly minor tweaks.
