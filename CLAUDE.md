# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Analizzatore CSV** is a CSV analysis dashboard for time-series data: automatic CSV format detection, numeric cleaning (EU/US locales, currency, thousands separators), data-quality checks, signal transforms, filtering (Moving Average, Butterworth) and FFT. Code, comments, log messages and UI strings are in Italian.

There are three UIs over a shared `core/` package. The Streamlit web app (`web_app.py`) is the primary, most complete one; it also ships as a portable Windows executable via PyInstaller. The Tkinter (`ui/desktop_app.py`) and Textual (`ui/main_app.py`) UIs are secondary and use the legacy loader.

## Commands

```bash
# Setup (requirements-dev.txt adds textual/rich for the TUI and pyinstaller)
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-dev.txt

# Web app (development)
streamlit run web_app.py
# Web app via the launcher used by the packaged exe (binds localhost:8501, applies settings)
python run_app.py

# Secondary UIs
python desktop_app_tk.py   # Tkinter -> ui/desktop_app.py
python main.py             # Textual TUI -> ui/main_app.py

# Portable Windows build (dist/AnalizzatoreCSV/AnalizzatoreCSV.exe, --onedir)
./build_portable.ps1          # add -Clean to wipe build/ and dist/ first
pyinstaller analizzatore_csv.spec --noconfirm

# Tests (pytest.ini addopts already include --cov=core and --cov-fail-under=80)
pytest
pytest tests/test_signal_tools.py::TestName::test_name --no-cov   # single test; --no-cov avoids the 80% gate failing
pytest -m unit          # markers: unit, integration, slow, synthetic

# Lint (as in CI)
ruff check core/ tests/
```

CI (`.github/workflows/ci.yml`) runs pytest with the 80% coverage gate on Ubuntu/Windows x Python 3.10–3.12, plus `ruff check core/ tests/`. Ruff is configured in `ruff.toml`: only `E4`, `E7`, `E9`, `F` rules, line length 120, target py310 — without it recent ruff versions report hundreds of style warnings. Unused imports/variables (`F401`/`F841`) fail CI. Coverage is measured only on `core/`; `web_app.py` and `ui/` are not covered. Manual edge-case CSVs live in `tests_csv/`; synthetic signal generators in `tests/fixtures/`.

## Architecture

### Runtime paths and configuration (read these first)

- **`core/paths.py`** is the single source of truth for file locations. `resource_path()` resolves read-only bundled assets (`config.json`, `assets/`, `web_app.py`); `outputs_dir()`, `logs_dir()`, `presets_dir()` resolve writable dirs. From source these are the project root; when frozen (PyInstaller, `sys.frozen`) assets come from `sys._MEIPASS` and writable dirs go to `%APPDATA%/AnalizzatoreCSV`. Never hardcode `outputs/`, `logs/`, `presets/` paths.
- **`core/settings.py`**: `effective_config()` deep-merges bundled `config.json` (defaults) with the user override `settings.json` in `user_data_dir()`; it is re-read on every call and is what every module should use instead of reading `config.json` directly. Also provides hardware detection (psutil, optional) and performance profiles (leggero/bilanciato/qualita). `DEFAULTS` there is the fallback when `config.json` is missing/incomplete.
- `config.json` holds desktop-oriented limits (2500 MB, 20M rows, 1200 s timeout); `config_web.json` holds tighter values for a hosted deployment. `web_app.py` additionally has `LIMIT_DEFAULTS` as a last-resort fallback.
- `core/loader_optimized.py` reads its thresholds (`SIZE_THRESHOLD_MB`, `ROWS_THRESHOLD`, `CHUNK_SIZE`, `SAMPLE_SIZE`) **once at import time**, so changed settings only take effect after restart.

### CSV pipeline

1. `core/analyzer.py` (`analyze_csv`) — encoding/BOM, delimiter (csv.Sniffer + fallback), header row detection.
2. `core/csv_cleaner.py` — scores decimal/thousands separator combinations on sampled values, strips currency/%/comparison tokens, vectorized pandas conversion; returns a `CleaningReport`.
3. Loaders: `loader.py` (legacy, used by Tk/TUI), `loader_optimized.py` (web app default when `performance.use_optimized_loader` is true; chunked loading and stratified sampling above size/row thresholds, progress callback). (A former `loader_v2.py` was removed as broken and unused; the untracked `benchmark_ultra_fast.py` still imports it.)
4. In `web_app.py`, `_parse_csv_with_timeout` parses files ≤ 200 MB in a thread; larger files go to a `spawn` subprocess running `core.csv_parser_worker._parsing_worker` for memory isolation and a real kill on timeout. The worker **must** stay in an importable `core` module with no heavy module-level imports: under PyInstaller `__main__` is `run_app.py`, so a worker defined in `web_app.py` fails to unpickle in the child (this was the cause of recent fixes).

### Signal processing

- `core/signal_tools.py`: `resolve_fs(x_values, manual_fs) -> (fs, source)` is the **only** authority for sampling frequency (manual > estimated from numeric/datetime X > None). Call it once, then pass the result as `fs_override` to filters and FFT so both use the same fs. `FilterSpec` + `validate_filter_spec()` (enforces cutoff < fs/2) + `apply_filter()`; `FFTSpec` + `compute_fft()` (needs fs > 0 and ≥ 4 samples). Butterworth/windowing need SciPy — check `_SCIPY_OK`.
- `core/signal_transforms.py`: non-destructive transforms (offset, scale, normalizations, shift, resample, derivative, integral) following the same Spec → Validate → Apply pattern; SciPy-dependent ones gated by `_SCIPY_INTERP_OK` / `_SCIPY_INTEG_OK`.
- `core/downsampling.py` (LTTB / minmax) is for rendering only. Filters, transforms and FFT always run on full-resolution data. The web app auto-enables performance mode above `PERFORMANCE_THRESHOLD` (100k rows).
- `core/quality.py`: `run_quality_checks()` (X monotonicity, gaps `dt > k·median(dt)`, robust-Z spikes) is non-blocking — returns a `DataQualityReport`, never raises.

### Other core modules

- `preset_manager.py` — saves `FilterSpec` + `FFTSpec` + `manual_fs` as JSON in `presets_dir()`; names are sanitized via `_preset_path()` (path traversal). Tuple cutoffs are stored as lists and converted back on load. `create_default_presets()` regenerates the defaults (in frozen mode `presets/` is not bundled).
- `report_manager.py` (statistical CSV/MD/HTML reports), `visual_report_manager.py` (PNG/PDF via Kaleido, falls back to HTML), `plot_manager.py` (Plotly HTML for Tk/TUI; the web app plots inline), `logger.py` (`LogManager`, daily file in `logs_dir()`).

### Web app (`web_app.py`)

A single ~3500-line Streamlit script. Parsed DataFrames are cached keyed on file size + hash + cleaning flag + optimization settings, so changing plot/filter parameters does not re-parse. Per-column widget state for visual reports uses keys like `f"vis_report_title::{column}"` and is purged when columns are deselected; forms that get recreated use a nonce in widget keys to avoid stale bindings. `.streamlit/config.toml` sets the dark theme and `maxUploadSize`; `run_app.py` overrides address/port and aligns `maxUploadSize` with `limits.max_file_mb`.

### Packaging (`analizzatore_csv.spec`)

Entry point `run_app.py` (calls `multiprocessing.freeze_support()` and starts Streamlit in-process via `streamlit.web.cli`). The spec bundles `web_app.py`, configs, `assets/`, `.streamlit/` and the `core/` source as data (Streamlit executes `_internal/web_app.py`), collects all of Streamlit plus package metadata, and excludes textual/rich/tkinter. When adding a new `core` module or a lazily imported dependency, check it is picked up by the spec (`collect_submodules("core")` / `hiddenimports`).

## Conventions and gotchas

- No emoji in the Streamlit UI.
- `fs = 0` in UI input means "auto"; `None` from `resolve_fs` means "unavailable".
- Check `pd.api.types.is_datetime64_any_dtype()` before numeric operations on X; tests run against pandas 3.0 behavior too.
- Validation helpers return `(bool, message)`; filter/FFT functions raise `ValueError` with user-facing messages that UIs catch and display.
- Never enable `aggressive_dtype_optimization` by default (precision loss); `optimize_dtypes` is the safe path.
- Background docs: `docs/ARCHITECTURE.md`, `docs/OTTIMIZZAZIONE_CARICAMENTO_CSV.md`, `docs/reports/CACHE_IMPLEMENTATION.md`, `docs/reports/PERFORMANCE_OPTIMIZATION_REPORT.md`, `tests/README.md`.
