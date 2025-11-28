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

No automated test suite is currently present. Manual testing uses sample CSV files in the `tests_csv/` directory that cover various edge cases:
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

3. **`loader.py`** - `load_csv()` function orchestrates the pipeline:
   - Uses metadata from `analyzer.py`
   - Applies cleaning via `csv_cleaner.py`
   - Returns cleaned DataFrame with optional detailed report
   - Supports custom decimal/thousands override

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
   - Visual report generation with per-plot customization
   - File upload with sample CSV loading
   - Caching for performance (important: uses file hash + cleaning flag as cache key)

2. **`ui/desktop_app.py`** (Tkinter):
   - Classic desktop interface
   - X-axis slicing support (numeric/datetime/positional)
   - Plot mode selection
   - Original signal overlay option

3. **`ui/main_app.py`** (Textual):
   - Terminal-based interface
   - Checkbox-based Y column selection
   - HTML plot preview in browser

### Key Design Patterns

**Performance Considerations:**
- `csv_cleaner.py` uses vectorized pandas operations (NOT row-by-row iteration)
- Caching in `web_app.py` prevents re-parsing on parameter changes
- Large CSV handling via `MAX_SAMPLE_VALUES` limit in format detection

**Error Handling Philosophy:**
- Validation functions return `(bool, message)` tuples for UI display
- Filter/FFT functions raise `ValueError` with human-readable messages
- UIs catch exceptions and show warnings/errors without crashing

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

tests_csv/
  *.csv               # Test cases covering edge cases
```

## Common Gotchas

1. **fs = 0 vs. fs = None**: Always use `resolve_fs()`. Zero means "auto-detect," None means "unavailable."

2. **Filter validation timing**: Validate filters before applying them, not during initialization. Filters may be configured before fs is determined.

3. **Datetime X handling**: Check dtype with `pd.api.types.is_datetime64_any_dtype()` before assuming numeric operations work.

4. **SciPy availability**: Always check `_SCIPY_OK` flag in `signal_tools.py` before enabling Butterworth options in UI.

5. **Minimum FFT samples**: FFT requires ≥4 samples, enforced in `compute_fft()`. Check `MIN_ROWS_FOR_FFT` (128) for UI recommendations.

6. **Streamlit session state**: Use unique keys for widgets when forms are recreated (e.g., `f"widget_{st.session_state.get('_nonce', 0)}"`) to avoid stale bindings.

7. **CSV encoding edge cases**: BOM detection handles UTF-16 LE/BE and UTF-8-sig. Always use detected encoding when reading with pandas.

## Dependencies

Core scientific: pandas ≥2.2, numpy ≥1.26, plotly[kaleido] ≥5.22, scipy ≥1.12
UI: streamlit ≥1.32, textual 0.89-0.90, rich ≥13.9
Optional: kaleido ≥0.2.1 (PNG/PDF export)
