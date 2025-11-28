# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Analizzatore CSV** is a multi-platform CSV analysis dashboard for exploring CSV files, filtering signals, and computing FFT. It provides three different interfaces (Web/Desktop/TUI) for time-series data analysis with automatic CSV format detection, signal filtering (Moving Average and Butterworth), and FFT computation.

## Commands

### Running the Application

```bash
# Web interface (Streamlit) - recommended for most users
streamlit run web_app.py

# Desktop interface (Tkinter)
python desktop_app_tk.py

# Terminal UI (Textual)
python main.py
```

### Development Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Testing

**Automated Test Suite:** Comprehensive pytest-based test suite with 80%+ coverage target on core modules.

```bash
# Run all tests with coverage
pytest tests/ -v --cov=core --cov-report=term-missing

# Run specific test module
pytest tests/test_signal_tools.py -v

# Run tests by marker
pytest -m unit -v
pytest -m integration -v
```

**Test Structure:**
- `tests/` - Test modules for all core components
- `tests/fixtures/` - Synthetic signal generators for validation
- `tests/conftest.py` - Shared fixtures and configuration
- `tests/manual/` - Manual test CSVs for edge cases

**Test Coverage:**
- ✅ `test_analyzer.py` - CSV metadata detection
- ✅ `test_csv_cleaner.py` - Numeric cleaning and format detection
- ✅ `test_signal_tools.py` - Filters, FFT, fs resolution
- ✅ `test_loader.py` - CSV loading pipeline
- ✅ `test_preset_manager.py` - Preset save/load
- ✅ `test_quality.py` - Data quality checks
- ✅ `test_downsampling.py` - LTTB/minmax downsampling
- ✅ `test_web_app_session.py` - Streamlit session state
- ✅ `test_optimization_safety.py` - dtype optimization safety
- ⚡ `benchmark_*.py` - Performance benchmarks
- ⚡ `profile_*.py` - Performance profiling

**Manual Test CSVs** (`tests_csv/`):
- Basic numeric data (`01_basic.csv`)
- Numeric X axis (`02_with_x_numeric.csv`)
- Datetime X axis (`03_with_x_datetime.csv`)
- Noisy signals (`04_noise_signal.csv`)
- Multi-column data (`05_multicolumn.csv`)
- NaN and Inf values (`06_nan_and_inf.csv`)
- Short signals (`07_short_signal.csv`)
- Large datasets (`08_big_signal.csv`)
- Italian locale numbers (`09_locale_it.csv`)
- Various thousands separators (`10_tab_space_thousands.csv`)
- Mixed numeric tokens (`11_mixed_tokens.csv`)
- Currency symbols (`12_currency_euro.csv`)

<<<<<<< HEAD
=======
**CI/CD:** Tests run automatically on GitHub Actions (Python 3.10-3.12, Ubuntu/Windows).

>>>>>>> 4b0004012f7119fd87aa9e87183af428067b9ce9
## Architecture

### Core Components (`core/` directory)

The application follows a modular architecture with separation between data processing, analysis, and UI layers:

#### CSV Processing Pipeline
1. **`analyzer.py`** - `CsvAnalyzer` class auto-detects CSV metadata:
   - BOM/encoding detection (UTF-8, UTF-16, UTF-8-sig)
   - Delimiter detection using Python's `csv.Sniffer` with fallback heuristics
   - Header row detection using text-to-numeric ratio heuristics
   - Column name extraction with BOM cleanup

2. **`csv_cleaner.py`** - Robust numeric data cleaning:
   - Detects decimal/thousands separators by scoring combinations across sample values
   - Handles multiple formats: European (`,` decimal, `.` thousands), US standard, spaces, apostrophes
   - Removes currency symbols, percentage signs, comparison operators
   - Vectorized pandas operations for performance (critical for large CSVs)
   - Returns `CleaningReport` with per-column conversion statistics

<<<<<<< HEAD
3. **`loader.py`** - `load_csv()` function orchestrates the pipeline:
   - Uses metadata from `analyzer.py`
   - Applies cleaning via `csv_cleaner.py`
   - Returns cleaned DataFrame with optional detailed report
   - Supports custom decimal/thousands override
=======
3. **`loader.py` / `loader_optimized.py`** - CSV loading with automatic optimization:
   - **`loader.py`**: Legacy loader for small files (< 50 MB)
   - **`loader_optimized.py`**: Optimized loader with chunked reading for large files
   - Auto-detection of optimal loading strategy based on file size/rows
   - **Chunked loading**: Reduces RAM peak by 45% for large files (500k rows: 1.6 GB → 890 MB)
   - **Stratified sampling**: Fast preview of large files without full load
   - Progress callback support for UI integration
   - Uses metadata from `analyzer.py` and cleaning from `csv_cleaner.py`
   - Returns cleaned DataFrame with optional detailed report
   - Supports custom decimal/thousands override
   - **Thresholds**: Files > 50 MB or > 100k rows use chunked loader
>>>>>>> 4b0004012f7119fd87aa9e87183af428067b9ce9

#### Signal Processing (`signal_tools.py`)

**Critical Design: Single Source of Truth for Sampling Frequency (fs)**

The `resolve_fs()` function is the **only** authority for determining sampling frequency:

```python
def resolve_fs(x_values, manual_fs) -> (fs, source):
    """
    Returns (fs, source) where source ∈ {"manual", "estimated", "none"}
    Priority:
      1. manual_fs > 0 → use manual value
      2. estimate from x_values (numeric or datetime)
      3. None if unavailable
    """
```

This design prevents:
- Inconsistent fs values between filters and FFT
- Re-estimation when manual fs is provided
- Butterworth/FFT errors when fs is unavailable

**Filter System:**
- `FilterSpec` dataclass defines filter parameters
- `validate_filter_spec()` enforces Nyquist limits and parameter validation
- Moving Average: always available, no fs required
- Butterworth (LP/HP/BP): requires fs > 0 and SciPy
- `apply_filter()` returns (filtered_series, fs_used)

**FFT System:**
- `FFTSpec` dataclass for FFT parameters
- `compute_fft()` requires fs > 0 and minimum 4 samples
- Supports detrending and windowing (Hann/others via SciPy)
- Returns (frequencies, amplitudes) or empty arrays if invalid

<<<<<<< HEAD
=======
#### Data Quality (`quality.py`)

**Non-blocking quality checks** that provide warnings without interrupting workflow:

- **X-axis Monotonicity**: Detects duplicate/decreasing values in time series
- **Gap Detection**: Finds irregular sampling where `dt > k * median(dt)`
- **Spike Detection**: Identifies outliers using robust Z-score (median + MAD)
- Returns `DataQualityReport` with:
  - Issue list with examples and percentages
  - Configurable thresholds (`gap_factor_k`, `spike_z`)
  - Notes for UI display (e.g., "irregular sampling may affect FFT")

#### Downsampling (`downsampling.py`)

**Performance optimization for large datasets** without sacrificing visual fidelity:

- **LTTB Algorithm**: Largest-Triangle-Three-Buckets preserves visual features
- **MinMax Method**: Captures peaks and valleys in high-frequency signals
- `downsample_series()` returns `DownsampleResult` with:
  - Downsampled x/y series with original index alignment
  - Reduction ratio and method used
  - Original vs sampled count
- **Design principle**: Filters and FFT always use original data; downsampling only for rendering
- Default: 10,000 points per trace for datasets > 100k rows

#### Preset Management (`preset_manager.py`)

**Save/load analysis configurations** for reproducible workflows:

- Saves `FilterSpec` + `FFTSpec` + `manual_fs` to JSON files in `presets/`
- Filename sanitization prevents path traversal vulnerabilities
- Schema versioning (`PRESET_VERSION = "1.0"`) for future compatibility
- `list_presets()` returns metadata without loading full specs
- Five default presets included:
  - "Media Mobile 5/20" - Light/heavy smoothing
  - "Butterworth LP 50Hz" - Low-pass filtering
  - "Analisi Vibrazione Completa" - BP filter + FFT
  - "Solo FFT" - FFT analysis without filtering

#### Plotting (`plot_manager.py`)

**Centralized Plotly plotting** with advanced features:

- **Data Cleaning**: Handles European/US decimal formats, removes inf/nan
- **X-axis Slicing**: Supports numeric, datetime, and positional ranges
- **Downsampling**: Equispaced downsampling to `max_points_per_trace`
- **Plot Modes**: Overlay (single figure) or Separate (one per series)
- **Configuration**: Reads from `config.json` for defaults
- **Output**: Saves HTML to `outputs/` and optionally opens in browser
- Used by TUI and desktop apps; web app has its own integrated plotting

>>>>>>> 4b0004012f7119fd87aa9e87183af428067b9ce9
#### Reporting

- **`report_manager.py`** - `ReportManager` generates statistical reports:
  - Descriptive statistics per column
  - Exports to CSV, Markdown, and HTML formats
  - Output to `outputs/` directory

- **`visual_report_manager.py`** - `VisualReportManager` creates visual reports:
  - Multi-panel plots (up to 4 series)
  - Exports to PNG, PDF (via Kaleido), or interactive HTML
  - Graceful fallback to HTML when Kaleido unavailable (common in cloud environments)
  - Uses Plotly for rendering

- **`logger.py`** - Centralized logging to `logs/analizzatore_YYYYMMDD.log`

### UI Implementations

Three independent UIs share the same core logic:

1. **`web_app.py`** (Streamlit) - Most feature-rich:
   - Advanced panel for fs override, filters, FFT
   - Three plot modes: overlaid, separate tabs, cascade
<<<<<<< HEAD
   - Visual report generation with per-plot customization
   - File upload with sample CSV loading
   - Caching for performance (important: uses file hash + cleaning flag as cache key)
=======
   - **Preset system**: Save/load analysis configurations
   - **Data quality checks**: Displays warnings for monotonicity, gaps, spikes
   - **Performance mode**: Auto-enables LTTB downsampling for datasets > 100k rows
   - **dtype optimization**: Reduces memory usage by downcasting numeric types
   - Visual report generation with per-plot customization
   - File upload with sample CSV loading
   - Caching for performance (uses file hash + cleaning flag + optimization settings as cache key)
   - Configurable limits via `config.json` (max file size, rows, columns, timeout)
>>>>>>> 4b0004012f7119fd87aa9e87183af428067b9ce9

2. **`ui/desktop_app.py`** (Tkinter):
   - Classic desktop interface
   - X-axis slicing support (numeric/datetime/positional)
   - Plot mode selection
   - Original signal overlay option

3. **`ui/main_app.py`** (Textual):
   - Terminal-based interface
   - Checkbox-based Y column selection
   - HTML plot preview in browser

<<<<<<< HEAD
=======
### Configuration (`config.json`)

The application reads configuration from `config.json` in the project root:

```json
{
  "quality": {
    "gap_factor_k": 5.0,      // Gap detection threshold multiplier
    "spike_z": 4.0,           // Spike detection Z-score threshold
    "min_points": 20,         // Minimum points for robust checks
    "max_examples": 5         // Examples per quality issue
  },
  "performance": {
    "optimize_dtypes": true,           // Enable memory optimization
    "aggressive_dtype_optimization": false  // Unsafe: may lose precision
  },
  "limits": {
    "max_file_mb": 200,       // Maximum CSV file size
    "max_rows": 1000000,      // Maximum rows to load
    "max_cols": 500,          // Maximum columns
    "parse_timeout_s": 120    // Timeout for parsing operations
  }
}
```

**Used by:** `web_app.py` (limits enforcement), `quality.py` (check thresholds), `loader.py` (optimization settings)

>>>>>>> 4b0004012f7119fd87aa9e87183af428067b9ce9
### Key Design Patterns

**Performance Considerations:**
- `csv_cleaner.py` uses vectorized pandas operations (NOT row-by-row iteration)
- Caching in `web_app.py` prevents re-parsing on parameter changes
- Large CSV handling via `MAX_SAMPLE_VALUES` limit in format detection
<<<<<<< HEAD
=======
- **Chunked CSV loading** (`loader_optimized.py`): Reduces RAM peak by 45% for large files
  - Files < 50 MB: standard loader (legacy, fastest for small files)
  - Files > 50 MB or > 100k rows: chunked loader (lower memory footprint)
  - Benchmark: 500k rows file uses 890 MB RAM vs 1.6 GB with standard loader
- **Stratified sampling**: Fast preview of large files without loading all data
- **LTTB downsampling**: Renders 10k points instead of full dataset for large files (>100k rows)
- **dtype optimization**: Automatically downcasts numeric types to reduce memory (int64→int32, float64→float32)
- **Progress tracking**: UI callbacks for long-running operations
- **Multiprocessing**: Optional for time-consuming operations (configurable per-UI)
>>>>>>> 4b0004012f7119fd87aa9e87183af428067b9ce9

**Error Handling Philosophy:**
- Validation functions return `(bool, message)` tuples for UI display
- Filter/FFT functions raise `ValueError` with human-readable messages
- UIs catch exceptions and show warnings/errors without crashing
<<<<<<< HEAD
=======
- Quality checks are non-blocking: display warnings but continue workflow

**Security Considerations:**
- Preset filenames are sanitized to prevent path traversal attacks
- File size limits prevent memory exhaustion attacks
- Parse timeout prevents DoS via malformed CSVs
- No arbitrary code execution (JSON-only config files)
>>>>>>> 4b0004012f7119fd87aa9e87183af428067b9ce9

**Conditional Dependencies:**
- SciPy: optional, Butterworth filters disabled if missing
- Kaleido: optional, PNG/PDF export falls back to HTML

## Important Implementation Details

### CSV Cleaning Cache Performance
The Streamlit app caches cleaned DataFrames using `(file_size, file_hash, apply_cleaning)` as the cache key. This prevents expensive re-cleaning when users toggle plot parameters. The cache is invalidated only when:
- A new file is uploaded
- The cleaning toggle is changed
- The session is reset

### Sampling Frequency Workflow
When working with fs-dependent features (Butterworth/FFT):
1. UI collects manual fs input (0 = auto)
2. Call `resolve_fs(x_values, manual_fs)` **once** at the start of processing
3. Pass the resolved `fs_value` as `fs_override` to all downstream functions
4. This ensures Butterworth and FFT use identical fs values

### Nyquist Validation
Butterworth filters **must** have cutoff < fs/2. The `validate_filter_spec()` function enforces this before filter application. UI code should:
1. Call `validate_filter_spec()` before filtering
2. Display the validation message to users
3. Skip filter application if validation fails (use original signal)

### HTML File Handling
Generated Plotly HTML files use sanitized filenames (spaces/special chars removed). Windows compatibility requires:
```python
# Use forward slashes or pathlib for cross-platform paths
import pathlib
path = pathlib.Path(filename).as_posix()
```

### Visual Report Columns
The Streamlit UI maintains per-column widget state for visual reports using keys like:
```python
f"vis_report_title::{column_name}"
f"vis_report_xlabel::{column_name}"
f"vis_report_ylabel::{column_name}"
```
When columns are deselected, their state is purged to prevent stale data.

<<<<<<< HEAD
=======
### Preset System Workflow
When working with presets for reproducible analyses:
1. **Creating a preset**: Collect `FilterSpec`, `FFTSpec`, and `manual_fs` from UI
2. Call `save_preset(name, description, fspec, fftspec, manual_fs)`
3. **Loading a preset**: Call `load_preset(name)` which returns a dict with all specs
4. Apply the loaded specs to UI widgets (set session state in Streamlit)
5. **Default presets**: Call `create_default_presets()` on app startup to ensure base presets exist
6. **Security**: Never use user-provided preset names directly in file paths; always use `_preset_path()` with sanitization

### Performance Mode
The Streamlit app includes a "Performance Mode" toggle:
- **Auto-enabled** when dataset has > 100k rows
- Uses LTTB downsampling to reduce rendering to ~10k points per trace
- **Critical**: Filtering and FFT always use original data; downsampling only affects plot rendering
- Users can toggle to "High Fidelity" mode to render all points (may cause browser lag)
- Status displayed: "Performance: 500,000 → 10,000 (50.0x, lttb)"

### Data Quality Workflow
Quality checks are performed after CSV loading and displayed as expandable warnings:
1. Load CSV with `load_csv()`
2. Call `run_quality_checks(df, x_col, y_cols, **config)`
3. Display `DataQualityReport` in UI:
   - Show summary: "Data quality: WARNING (x_gap=5, y_spike=2)"
   - List each issue with count, percentage, and examples
   - Display soft recommendations (e.g., "irregular sampling may affect FFT")
4. **Never block workflow**: Users can proceed with analysis even with quality warnings

>>>>>>> 4b0004012f7119fd87aa9e87183af428067b9ce9
## Output Structure

```
outputs/
  *.html              # Interactive Plotly plots
  *.csv               # Statistical reports
  *.md / *.html       # Report text formats
  visual_reports/
    visual_report_YYYYMMDD_HHMMSS.{png,pdf,html}

logs/
  analizzatore_YYYYMMDD.log

<<<<<<< HEAD
tests_csv/
  *.csv               # Test cases covering edge cases
=======
presets/
  *.json              # Saved analysis configurations (FilterSpec + FFTSpec + fs)

tests/
  *.py                # Automated test suite with pytest
  fixtures/           # Synthetic signal generators
  manual/             # Manual test files
  profile_results.txt # Performance profiling results

tests_csv/
  *.csv               # Manual test cases covering edge cases

docs/
  *.md                # Additional documentation and guides

patches/
  *.diff              # Patch files for specific issues/features

scripts/
  csv_spawner.py      # Generate synthetic CSV test files
>>>>>>> 4b0004012f7119fd87aa9e87183af428067b9ce9
```

## Common Gotchas

1. **fs = 0 vs. fs = None**: Always use `resolve_fs()`. Zero means "auto-detect," None means "unavailable."

2. **Filter validation timing**: Validate filters before applying them, not during initialization. Filters may be configured before fs is determined.

3. **Datetime X handling**: Check dtype with `pd.api.types.is_datetime64_any_dtype()` before assuming numeric operations work.

4. **SciPy availability**: Always check `_SCIPY_OK` flag in `signal_tools.py` before enabling Butterworth options in UI.

5. **Minimum FFT samples**: FFT requires ≥4 samples, enforced in `compute_fft()`. Check `MIN_ROWS_FOR_FFT` (128) for UI recommendations.

6. **Streamlit session state**: Use unique keys for widgets when forms are recreated (e.g., `f"widget_{st.session_state.get('_nonce', 0)}"`) to avoid stale bindings.

7. **CSV encoding edge cases**: BOM detection handles UTF-16 LE/BE and UTF-8-sig. Always use detected encoding when reading with pandas.

<<<<<<< HEAD
=======
8. **Downsampling vs. processing**: LTTB/minmax downsampling is for rendering only. Filters and FFT must always operate on the original, full-resolution data before any downsampling.

9. **Quality checks are non-blocking**: `run_quality_checks()` returns warnings but never raises exceptions. Display the warnings to users but allow them to continue their workflow.

10. **Preset JSON tuple handling**: JSON doesn't support tuples, so `FilterSpec.cutoff` (which can be a tuple) is stored as a list in preset files and converted back to tuple on load.

11. **dtype optimization safety**: Use `optimize_dtypes=True` for safe optimization (respects ranges). Never use `aggressive_dtype_optimization=True` in production as it may cause precision loss.

12. **Config.json errors**: If `config.json` is missing or malformed, the application uses hardcoded defaults (`LIMIT_DEFAULTS` in web_app.py). Don't crash on bad config.

13. **Performance thresholds**: Web app auto-enables performance mode at 100k rows (`PERFORMANCE_THRESHOLD`). When manually toggling, ensure state is properly propagated to plotting logic.

14. **Preset name conflicts**: `preset_exists()` checks before saving. If overwriting an existing preset, either delete first with `delete_preset()` or handle the overwrite in UI logic.

15. **Chunked loader vs legacy loader**: `loader_optimized.py` auto-selects strategy based on file size. Use `use_optimization=False` to force legacy loader for debugging. Files < 50 MB use legacy loader by default (faster for small files).

>>>>>>> 4b0004012f7119fd87aa9e87183af428067b9ce9
## Dependencies

Core scientific: pandas ≥2.2, numpy ≥1.26, plotly[kaleido] ≥5.22, scipy ≥1.12
UI: streamlit ≥1.32, textual 0.89-0.90, rich ≥13.9
Optional: kaleido ≥0.2.1 (PNG/PDF export)
<<<<<<< HEAD
=======
Testing: pytest ≥7.0, pytest-cov (for coverage reports)

## Additional Documentation

The repository includes several specialized documentation files:

- **`README.md`** - Main project documentation with features, screenshots, and quick start
- **`CACHE_IMPLEMENTATION.md`** - Deep dive into Streamlit caching strategy and file signature system
- **`PERFORMANCE_OPTIMIZATION_REPORT.md`** - Performance improvements, benchmarks, and optimization strategies
- **`docs/OTTIMIZZAZIONE_CARICAMENTO_CSV.md`** - CSV loading optimization guide (chunked loading, sampling, benchmarks)
- **`FIX_FFT_CHECKBOX.md`** - Technical notes on FFT checkbox state management issue
- **`PIANO_OTTIMIZZAZIONE_CSV.md`** - Italian: CSV parsing optimization plan
- **`PIANO_OTTIMIZZAZIONE_SICUREZZA.md`** - Italian: Security optimization and hardening plan
- **`tests/README.md`** - Test suite documentation with fixtures, coverage targets, and CI/CD info
- **`docs/`** - Additional guides and technical documentation

Consult these files when working on:
- Performance tuning: `PERFORMANCE_OPTIMIZATION_REPORT.md`
- Cache debugging: `CACHE_IMPLEMENTATION.md`
- Adding tests: `tests/README.md`
- Security hardening: `PIANO_OTTIMIZZAZIONE_SICUREZZA.md`
>>>>>>> 4b0004012f7119fd87aa9e87183af428067b9ce9
