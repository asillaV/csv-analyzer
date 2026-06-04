from __future__ import annotations

import concurrent.futures
import hashlib
import html
import inspect
import multiprocessing as mp
import os
import queue
import tempfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as _pio
_pio.templates.default = "plotly_dark"
import streamlit as st

from core.analyzer import analyze_csv
from core.csv_cleaner import CleaningReport
from core.csv_parser_worker import _parsing_worker
from core.paths import resource_path
from core import settings as app_settings

# Import condizionale: usa loader ottimizzato se configurato
import json
_use_optimized = True  # Default
try:
    _use_optimized = app_settings.effective_config().get("performance", {}).get("use_optimized_loader", True)
except Exception:
    pass

if _use_optimized:
    from core.loader_optimized import load_csv
    LOADER_TYPE = "optimized"
else:
    from core.loader import load_csv
    LOADER_TYPE = "legacy"
from core.report_manager import ReportManager
from core.visual_report_manager import VisualPlotSpec, VisualReportManager
from core.downsampling import downsample_series, DownsampleResult
from core.signal_tools import (
    FilterSpec,
    FFTSpec,
    resolve_fs,
    validate_filter_spec,
    apply_filter,
    compute_fft,
)
from core.signal_transforms import (
    TransformSpec,
    TRANSFORM_LABELS,
    TRANSFORM_GUIDE,
    apply_transform_pipeline,
    validate_transform_spec,
    transform_spec_cache_key,
)
from core.quality import run_quality_checks, DataQualityReport
from core.preset_manager import (
    save_preset,
    load_preset,
    list_presets,
    delete_preset,
    preset_exists,
    create_default_presets,
    PresetError,
)
from core.logger import LogManager

FileSignature = Tuple[str, int, str]

# ---------------------- Reset helpers ---------------------- #
RESETTABLE_KEYS = {
    # Form principali
    "x_col",
    "y_cols",
    "plot_mode",
    "ax_x_min",
    "ax_x_max",
    "ax_y_mode",
    "ax_y_min_global",
    "ax_y_max_global",
    "fillna_forward",
    # Advanced
    "manual_fs",
    "enable_filter",
    "f_kind",
    "ma_win",
    "f_order",
    "f_lo",
    "f_hi",
    "overlay_orig",
    # sidebar FFT widgets (moved from form Advanced section)
    "sidebar_enable_fft",
    "sidebar_fft_use",
    "sidebar_detrend",
    # Trasformazioni
    "_n_transforms",
    # Report testuale
    "report_format",
    "report_base_name",
    # Report visivo (campi globali)
    "vis_report_main_title",
    "vis_report_base",
    "vis_report_format",
    "vis_report_legend",
    "_sample_error",
    "plot_quality_mode",
}

MIN_ROWS_FOR_FFT = 128
PERFORMANCE_THRESHOLD = 100_000
PERFORMANCE_MAX_POINTS = 10_000
PERFORMANCE_METHOD = "lttb"

LIMIT_DEFAULTS = {
    "max_file_mb": 200,
    "max_rows": 1_000_000,
    "max_cols": 500,
    "parse_timeout_s": 120,
}


def _load_limits_config() -> Dict[str, float]:
    """Legge i limiti dalla config effettiva (default + override utente).

    Letta fresca ad ogni chiamata: gli override salvati dall'utente nel pannello
    Impostazioni hanno effetto immediato sul prossimo caricamento.
    """
    merged: Dict[str, float] = dict(LIMIT_DEFAULTS)
    try:
        limits = app_settings.effective_config().get("limits") or {}
        for key in LIMIT_DEFAULTS:
            value = limits.get(key)
            if isinstance(value, (int, float)):
                merged[key] = float(value)
    except Exception:
        # In caso di problemi con il file di config, manteniamo i default.
        pass
    return merged


def _check_size_limit(size_bytes: int, limits: Dict[str, float]) -> Optional[str]:
    """Ritorna un messaggio di errore se la dimensione supera i limiti."""
    if size_bytes <= 0:
        return "Il file caricato è vuoto."
    max_bytes = limits["max_file_mb"] * 1024 * 1024
    if size_bytes > max_bytes:
        return (
            f"File troppo grande ({size_bytes / (1024**2):.1f} MB). "
            f"Limite massimo: {limits['max_file_mb']:.0f} MB."
        )
    return None


def _check_dataframe_limits(df: pd.DataFrame, limits: Dict[str, float]) -> Optional[str]:
    """Verifica che il dataframe rispetti i limiti di righe e colonne."""
    max_rows = int(limits["max_rows"])
    max_cols = int(limits["max_cols"])
    if len(df) > max_rows:
        return (
            f"Dataset troppo grande: {len(df):,} righe. "
            f"Limite massimo consentito: {max_rows:,}."
        )
    if len(df.columns) > max_cols:
        return (
            f"Dataset con troppe colonne: {len(df.columns):,}. "
            f"Limite massimo consentito: {max_cols:,}."
        )
    return None


def _clear_cached_dataset() -> None:
    """Rimuove dataframe e metadati cache dalla sessione."""
    st.session_state.pop("_cached_df", None)
    st.session_state.pop("_cached_cleaning_report", None)
    st.session_state.pop("_cached_meta", None)
    st.session_state.pop("_cached_file_sig", None)
    st.session_state.pop("_cached_apply_cleaning", None)


def _parse_csv_in_thread(file_bytes: bytes, apply_cleaning: bool) -> Tuple[pd.DataFrame, CleaningReport, Dict[str, Any]]:
    """Parsing CSV nel thread corrente — nessun subprocess spawn, zero overhead fisso."""
    fd, tmp_name = tempfile.mkstemp(prefix="csv_upload_", suffix=".csv")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(file_bytes)
        meta = analyze_csv(str(tmp_path))
        df, cleaning_report = load_csv(
            str(tmp_path),
            encoding=meta.get("encoding"),
            delimiter=meta.get("delimiter"),
            header=meta.get("header"),
            apply_cleaning=apply_cleaning,
            return_details=True,
        )
        return df, cleaning_report, dict(meta)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


# File < soglia: usa threading (zero spawn overhead).
# File >= soglia: usa subprocess per memory isolation e kill reale in caso di OOM.
_THREADING_SIZE_THRESHOLD = 200 * 1024 * 1024  # 200 MB

# Rilevamento ambiente Streamlit Cloud (impostare IS_STREAMLIT_CLOUD=true nelle env vars del deploy)
_IS_STREAMLIT_CLOUD = bool(
    os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("IS_STREAMLIT_CLOUD")
)


def _parse_csv_with_timeout(file_bytes: bytes, apply_cleaning: bool, timeout_s: float) -> Tuple[pd.DataFrame, CleaningReport, Dict[str, Any]]:
    """Esegue analyze + load con timeout.

    File piccoli (<= 200 MB): thread nello stesso processo (nessun overhead spawn).
    File grandi (> 200 MB): subprocess separato per memory isolation e kill reale se OOM.
    Il worker e' in core.csv_parser_worker (non in __main__) per compatibilita' PyInstaller.
    """
    if len(file_bytes) <= _THREADING_SIZE_THRESHOLD:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_parse_csv_in_thread, file_bytes, apply_cleaning)
            try:
                return future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"Parsing del CSV oltre il tempo massimo di {timeout_s:.0f}s."
                ) from None

    # Slow path: subprocess per file grandi (memory isolation + kill reale se OOM).
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    fd, tmp_name = tempfile.mkstemp(prefix="csv_upload_", suffix=".csv")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(file_bytes)

        process = ctx.Process(
            target=_parsing_worker,
            args=(result_queue, str(tmp_path), apply_cleaning),
            daemon=True,
        )
        process.start()

        try:
            message = result_queue.get(timeout=timeout_s)
        except queue.Empty:
            process.terminate()
            process.join()
            raise TimeoutError(
                f"Parsing del CSV oltre il tempo massimo di {timeout_s:.0f}s."
            ) from None

        process.join()
        status = message[0]
        if status == "ok":
            _, meta, df, cleaning_report = message
            return df, cleaning_report, dict(meta)

        error = message[1] if len(message) > 1 else RuntimeError("Errore sconosciuto nel parsing.")
        if isinstance(error, Exception):
            raise error
        raise RuntimeError(str(error))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
SAMPLE_CSV_PATH = resource_path("assets/sample_timeseries.csv")

# Cache limits
MAX_FILTER_CACHE_SIZE = 32
MAX_FFT_CACHE_SIZE = 16
MAX_QUALITY_CACHE_SIZE = 4

# FIX ISSUE #55: Cache telemetry per monitoraggio hit/miss rate
CACHE_STATS = {
    "filter_hits": 0,
    "filter_misses": 0,
    "fft_hits": 0,
    "fft_misses": 0,
}


# ---------------------- Session helpers (Issue #49) ---------------------- #
def _ensure_session_id() -> str:
    """Return the per-session identifier, initializing it on first access."""
    key = "dataset_id"
    if key not in st.session_state:
        st.session_state[key] = uuid.uuid4().hex
    return st.session_state[key]


def _build_file_signature(file_bytes: bytes) -> FileSignature:
    """Create a file signature bound to the current session."""
    session_id = _ensure_session_id()
    size = len(file_bytes)
    digest = hashlib.sha1(file_bytes).hexdigest()[:16]
    return (session_id, size, digest)


# ---------------------- Session helpers (Issue #49) ---------------------- #
def _ensure_session_id() -> str:
    """Return the per-session identifier, initializing it on first access."""
    key = "dataset_id"
    if key not in st.session_state:
        st.session_state[key] = uuid.uuid4().hex
    return st.session_state[key]


def _build_file_signature(file_bytes: bytes) -> FileSignature:
    """Create a file signature bound to the current session."""
    session_id = _ensure_session_id()
    size = len(file_bytes)
    digest = hashlib.sha1(file_bytes).hexdigest()[:16]
    return (session_id, size, digest)


# ---------------------- Cache helpers (Issue #35) ---------------------- #
def _init_result_caches() -> None:
    """Initialize cache dictionaries in session state if not present."""
    if "_filter_cache" not in st.session_state:
        st.session_state["_filter_cache"] = {}
    if "_fft_cache" not in st.session_state:
        st.session_state["_fft_cache"] = {}
    if "_quality_cache" not in st.session_state:
        st.session_state["_quality_cache"] = {}
    if "_transform_cache" not in st.session_state:
        st.session_state["_transform_cache"] = {}


def _get_filter_cache_key(
    column: str,
    file_sig: FileSignature,
    fspec: FilterSpec,
    fs: Optional[float],
    fs_source: Optional[str],
    fill_stamp: bool,
) -> Tuple:
    """Generate hashable cache key for filter results."""
    from dataclasses import astuple
    # Include fs, fs_source e stato del fill per invalidare correttamente
    return (column, file_sig, astuple(fspec), fs, fs_source, fill_stamp)


def _get_fft_cache_key(
    column: str,
    file_sig: FileSignature,
    is_filtered: bool,
    fftspec: FFTSpec,
    fs: float,
    fs_source: Optional[str],
    fill_stamp: bool,
    filter_sig: Optional[Tuple] = None,
) -> Tuple:
    """Generate hashable cache key for FFT results."""
    from dataclasses import astuple
    # Include fs_source e stato del fill per invalidare quando cambiano questi parametri.
    # Include filter_sig (astuple(fspec)) quando is_filtered: lo spettro del segnale
    # filtrato dipende dal filtro applicato, quindi deve invalidarsi al cambio filtro
    # (es. MA finestra 10 -> 1). Per l'originale resta None (non dipende dal filtro).
    return (column, file_sig, is_filtered, astuple(fftspec), fs, fs_source, fill_stamp, filter_sig)


def _get_cached_filter(key: Tuple) -> Optional[pd.Series]:
    """Retrieve cached filter result with telemetry."""
    result = st.session_state.get("_filter_cache", {}).get(key)
    # FIX ISSUE #55: Track cache hit/miss
    if result is not None:
        CACHE_STATS["filter_hits"] += 1
    else:
        CACHE_STATS["filter_misses"] += 1
    return result


def _cache_filter_result(key: Tuple, result: pd.Series) -> None:
    """Store filter result with LRU eviction."""
    cache = st.session_state.setdefault("_filter_cache", {})
    if len(cache) >= MAX_FILTER_CACHE_SIZE:
        # Simple LRU: remove oldest (first) entry
        oldest_key = next(iter(cache))
        cache.pop(oldest_key)
    cache[key] = result.copy()  # Store copy to avoid reference issues


def _get_cached_fft(key: Tuple) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Retrieve cached FFT result with telemetry."""
    result = st.session_state.get("_fft_cache", {}).get(key)
    # FIX ISSUE #55: Track cache hit/miss
    if result is not None:
        CACHE_STATS["fft_hits"] += 1
    else:
        CACHE_STATS["fft_misses"] += 1
    return result


def _cache_fft_result(key: Tuple, freqs: np.ndarray, amp: np.ndarray) -> None:
    """Store FFT result with LRU eviction."""
    cache = st.session_state.setdefault("_fft_cache", {})
    if len(cache) >= MAX_FFT_CACHE_SIZE:
        # Simple LRU: remove oldest (first) entry
        oldest_key = next(iter(cache))
        cache.pop(oldest_key)
    cache[key] = (freqs.copy(), amp.copy())  # Store copies


def _invalidate_result_caches() -> None:
    """Clear all filter, FFT, transform and quality caches (called on file change)."""
    st.session_state.pop("_filter_cache", None)
    st.session_state.pop("_fft_cache", None)
    st.session_state.pop("_quality_cache", None)
    st.session_state.pop("_transform_cache", None)


def _apply_filter_cached(
    series: pd.Series,
    x_series: Optional[pd.Series],
    fspec: FilterSpec,
    fs_value: Optional[float],
    fs_source: Optional[str],
    file_sig: FileSignature,
    column_name: str,
    fill_stamp: bool,
) -> Optional[pd.Series]:
    """Apply filter with caching. Returns filtered series or None if filter fails."""
    _init_result_caches()
    cache_key = _get_filter_cache_key(column_name, file_sig, fspec, fs_value, fs_source, fill_stamp)
    cached = _get_cached_filter(cache_key)
    if cached is not None:
        if cached.index.equals(series.index):
            return cached
        # Indici cambiati (es. decimazione performance): invalida e ricalcola
        cache_store = st.session_state.get("_filter_cache")
        if isinstance(cache_store, dict):
            cache_store.pop(cache_key, None)
        cached = None
    if cached is not None:
        return cached
    # Cache miss: compute filter
    try:
        filtered, _ = apply_filter(series, x_series, fspec, fs_override=fs_value)
        _cache_filter_result(cache_key, filtered)
        return filtered
    except Exception:
        return None


MAX_TRANSFORM_CACHE_SIZE = 32


def _apply_transform_pipeline_cached(
    series: pd.Series,
    x_series: Optional[pd.Series],
    specs: List[TransformSpec],
    fs_info: Any,
    file_sig: FileSignature,
    column_name: str,
    fill_stamp: bool,
) -> Tuple[pd.Series, Optional[pd.Series], List[str], bool]:
    """Apply transform pipeline with caching. Returns (y, x, descriptions, changed_length)."""
    _init_result_caches()
    specs_key = tuple(transform_spec_cache_key(s) for s in specs)
    fs_val = fs_info.value if fs_info else None
    fs_src = fs_info.source if fs_info else None
    cache_key = (column_name, file_sig, specs_key, fs_val, fs_src, fill_stamp)

    cache = st.session_state.setdefault("_transform_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    y_out, x_out, descs, changed = apply_transform_pipeline(
        series, x_series, specs, fs_info, col_name=column_name
    )
    result: Tuple[pd.Series, Optional[pd.Series], List[str], bool] = (
        y_out.copy(),
        x_out.copy() if x_out is not None else None,
        descs,
        changed,
    )
    if len(cache) >= MAX_TRANSFORM_CACHE_SIZE:
        cache.pop(next(iter(cache)))
    cache[cache_key] = result
    return result


def _compute_fft_cached(
    series: pd.Series,
    fs_value: float,
    fs_source: Optional[str],
    fftspec: FFTSpec,
    file_sig: FileSignature,
    column_name: str,
    is_filtered: bool,
    fill_stamp: bool,
    fspec: Optional[FilterSpec] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FFT with caching. Returns (freqs, amp) arrays."""
    from dataclasses import astuple
    _init_result_caches()
    # La firma del filtro entra nella chiave solo per lo spettro del segnale filtrato,
    # così cambiando il filtro (es. MA 10 -> 1) la FFT del filtrato non resta in cache stale.
    filter_sig = astuple(fspec) if (is_filtered and fspec is not None) else None
    cache_key = _get_fft_cache_key(
        column_name, file_sig, is_filtered, fftspec, fs_value, fs_source, fill_stamp, filter_sig
    )
    cached = _get_cached_fft(cache_key)
    if cached is not None:
        return cached
    # Cache miss: compute FFT
    freqs, amp = compute_fft(series, fs_value, detrend=fftspec.detrend, window=fftspec.window)
    _cache_fft_result(cache_key, freqs, amp)
    return freqs, amp


def _reset_all_settings() -> None:
    """Reset widgets/output while keeping the current file and cached data."""
    for k in list(RESETTABLE_KEYS):
        st.session_state.pop(k, None)

    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("vis_report_"):
            st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and (key.startswith("ax_y_min_") or key.startswith("ax_y_max_")):
            st.session_state.pop(key, None)

    # Reset plot and report outputs
    st.session_state.pop("_plots_ready", None)
    st.session_state.pop("_generated_report", None)
    st.session_state.pop("_generated_report_error", None)
    st.session_state.pop("_generated_visual_report", None)
    st.session_state.pop("_generated_visual_report_error", None)
    st.session_state.pop("_fill_last_stamp", None)

    # Reset visual report tracking state
    st.session_state.pop("_visual_report_prev_selection", None)
    st.session_state.pop("_visual_report_last_default_x_label", None)

    # Reset quality mode to default
    st.session_state.pop("_quality_file_sig", None)

    # Reset preset state
    st.session_state.pop("_loaded_preset", None)
    st.session_state.pop("_loaded_preset_name", None)
    st.session_state.pop("_active_preset_name", None)
    st.session_state.pop("_pending_preset_save", None)

    # Reset pannello trasformazioni (passi individuali + pending action)
    for _k in list(st.session_state.keys()):
        if isinstance(_k, str) and _k.startswith("tr_"):
            st.session_state.pop(_k, None)
    st.session_state.pop("_pending_delete_step", None)

    st.session_state["_controls_nonce"] = st.session_state.get("_controls_nonce", 0) + 1

# ---------------------- Streamlit compatibility helpers ---------------------- #
def _supports_kwarg(func: Any, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _plotly_chart(container: Any, fig: go.Figure, **kwargs: Any) -> Any:
    plot_func = container.plotly_chart
    opts: dict[str, Any] = {}
    if _supports_kwarg(plot_func, "width"):
        opts["width"] = "stretch"
    elif _supports_kwarg(plot_func, "use_container_width"):
        opts["use_container_width"] = True
    opts.update(kwargs)
    return plot_func(fig, **opts)


def _dataframe(data: Any, **kwargs: Any) -> Any:
    opts: dict[str, Any] = {}
    if _supports_kwarg(st.dataframe, "width"):
        opts["width"] = "stretch"
    elif _supports_kwarg(st.dataframe, "use_container_width"):
        opts["use_container_width"] = True
    opts.update(kwargs)
    return st.dataframe(data, **opts)


def _button(label: str, **kwargs: Any) -> Any:
    opts: dict[str, Any] = {}
    if _supports_kwarg(st.button, "width"):
        opts["width"] = "stretch"
    elif _supports_kwarg(st.button, "use_container_width"):
        opts["use_container_width"] = True
    opts.update(kwargs)
    return st.button(label, **opts)


def _image(data: Any, **kwargs: Any) -> Any:
    opts: dict[str, Any] = {}
    if _supports_kwarg(st.image, "width"):
        opts["width"] = "stretch"
    elif _supports_kwarg(st.image, "use_container_width"):
        opts["use_container_width"] = True
    elif _supports_kwarg(st.image, "use_column_width"):
        opts["use_column_width"] = True
    opts.update(kwargs)
    return st.image(data, **opts)


# ---------------------- util ---------------------- #
def _to_float_or_none(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return float(s.replace(",", "."))
    except Exception:
        return None


def _parse_range_num(min_s: str, max_s: str, data: pd.Series) -> Optional[Tuple[float, float]]:
    vmin = _to_float_or_none(min_s)
    vmax = _to_float_or_none(max_s)
    if vmin is None and vmax is None:
        return None
    if vmin is None:
        vmin = float(pd.to_numeric(data, errors="coerce").min())
    if vmax is None:
        vmax = float(pd.to_numeric(data, errors="coerce").max())
    if vmin == vmax:
        return None
    return (vmin, vmax)


def _parse_range_x(min_s: str, max_s: str, x: pd.Series | pd.Index) -> Optional[Tuple]:
    # gestisce sia numerici sia datetime
    if pd.api.types.is_datetime64_any_dtype(x):
        xmin = pd.to_datetime(min_s, errors="coerce") if min_s else None
        xmax = pd.to_datetime(max_s, errors="coerce") if max_s else None
        if xmin is None and xmax is None:
            return None
        if xmin is None:
            xmin = pd.to_datetime(pd.Series(x)).min()
        if xmax is None:
            xmax = pd.to_datetime(pd.Series(x)).max()
        if pd.isna(xmin) or pd.isna(xmax) or xmin == xmax:
            return None
        return (xmin, xmax)
    else:
        # prova come numerico
        xv = pd.to_numeric(pd.Series(x), errors="coerce")
        return _parse_range_num(min_s, max_s, xv)


def _fmt_csv_token(token: Optional[str]) -> str:
    if token is None:
        return "auto"
    if token == "\t" or token == "      ":
        return "\\t"
    if token == " ":
        return "' '"
    if token == "":
        return "vuoto"
    return token


def _meta_info_html(label: str, value: Any) -> str:
    """Format metadata entries for safe HTML rendering."""
    safe_label = html.escape(str(label))
    safe_value = html.escape("" if value is None else str(value))
    return f"**{safe_label}**<br/>{safe_value}"


def _cleaning_stats_table(report: CleaningReport) -> pd.DataFrame:
    rows = []
    for name, stats in report.columns.items():
        percent_non_numeric = (
            stats.non_numeric / stats.candidate_numeric if stats.candidate_numeric else 0.0
        )
        rows.append(
            {
                "Colonna": name,
                "Valori candidati": stats.candidate_numeric,
                "Convertiti": stats.converted,
                "Non numerici": stats.non_numeric,
                "% non numerici": f"{percent_non_numeric:.1%}" if stats.candidate_numeric else "n.d.",
                "Correzione applicata": "si" if stats.applied else "no",
            }
        )
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        columns=[
            "Colonna",
            "Valori candidati",
            "Convertiti",
            "Non numerici",
            "% non numerici",
            "Correzione applicata",
        ]
    )

def _to_datetime_flexible(values: Any) -> pd.Series:
    """
    Converte in datetime riducendo i warning di inferenza formato.

    Usa format=\"mixed\" se disponibile (pandas >=2.1) e ripiega su
    to_datetime standard se non supportato.
    """
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")  # type: ignore[arg-type]
    except TypeError:
        return pd.to_datetime(values, errors="coerce")  # type: ignore[arg-type]

def _parse_x_column_once(df: pd.DataFrame, x_col: Optional[str]) -> Optional[pd.Series]:
    """
    FIX ISSUE #52: Pre-converti colonna X una volta sola prima del loop plot.

    Evita conversioni datetime/numeric ripetute per ogni colonna Y.
    Su 100k righe × 5 cols: risparmio ~1 secondo (200ms × 5).

    Args:
        df: DataFrame contenente la colonna X
        x_col: Nome della colonna X (o None)

    Returns:
        Serie X convertita (datetime/numeric) o None se non disponibile
    """
    if not x_col or x_col not in df.columns:
        return None

    xraw = df[x_col]

    # Se già datetime/timedelta, converti e ritorna
    if pd.api.types.is_datetime64_any_dtype(xraw) or pd.api.types.is_timedelta64_dtype(xraw):
        return _to_datetime_flexible(xraw)

    # Prova coerzione numerica
    xnum = pd.to_numeric(xraw, errors="coerce")
    if xnum.notna().mean() >= 0.8:
        return xnum

    # Fallback: stringhe/datetime
    try:
        xdt = _to_datetime_flexible(xraw)
        return xdt
    except Exception:
        return None


def _make_time_series(
    df: pd.DataFrame,
    x_col: Optional[str],
    y_col: str,
    x_parsed: Optional[pd.Series] = None
) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Estrae serie Y e X per plotting.

    FIX ISSUE #52: Accetta x_parsed pre-processato per evitare conversioni ripetute.

    Args:
        df: DataFrame contenente i dati
        x_col: Nome colonna X (per retrocompatibilità, ignorato se x_parsed è fornito)
        y_col: Nome colonna Y
        x_parsed: Serie X già convertita (opzionale, FIX #52)

    Returns:
        Tupla (y_series, x_series)
    """
    y = pd.to_numeric(df[y_col], errors="coerce")
    y.name = y_col

    # FIX ISSUE #52: Se X già parsato, usa quello (evita ri-conversione)
    if x_parsed is not None:
        return y, x_parsed

    # Fallback legacy: converti X al volo (solo per retrocompatibilità)
    if x_col and x_col in df.columns:
        xraw = df[x_col]
        if pd.api.types.is_datetime64_any_dtype(xraw) or pd.api.types.is_timedelta64_dtype(xraw):
            return y, _to_datetime_flexible(xraw)
        # prova coerzione numerica
        xnum = pd.to_numeric(xraw, errors="coerce")
        if xnum.notna().mean() >= 0.8:
            return y, xnum
        # fallback: stringhe/datetime
        try:
            xdt = _to_datetime_flexible(xraw)
            return y, xdt
        except Exception:
            pass
    return y, None


def _mask_xy(y: pd.Series, x: Optional[pd.Series]) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Rimuove coppie X/Y non valide (NaN) per evitare trace vuote.
    Non modifica le serie originali, ritorna viste filtrate.
    """
    if x is not None:
        mask = y.notna() & x.notna()
    else:
        mask = y.notna()
    if not mask.any():
        empty_y = y.iloc[0:0]
        empty_x = x.iloc[0:0] if isinstance(x, pd.Series) else None
        return empty_y, empty_x
    return y.loc[mask], x.loc[mask] if x is not None else None


def _plot_xy(x: Optional[pd.Series], y: pd.Series, name: str) -> go.Figure:
    fig = go.Figure()
    if x is not None and x.notna().any():
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
        fig.update_xaxes(title="X")
    else:
        fig.add_trace(go.Scatter(y=y, mode="lines", name=name))
        fig.update_xaxes(title="index")
    fig.update_yaxes(title=name)
    fig.update_layout(
        margin=dict(l=40, r=20, t=30, b=40),
        height=420,
        paper_bgcolor="#020617",
        plot_bgcolor="#0a1628",
    )
    return fig


def _plot_fft(freqs: np.ndarray, amp: np.ndarray, title: str = "FFT") -> go.Figure:
    fig = go.Figure()
    if freqs.size > 0 and amp.size > 0:
        fig.add_trace(go.Scatter(x=freqs, y=amp, mode="lines", name="amp"))
        fig.update_xaxes(title="Frequenza [Hz]")
        fig.update_yaxes(title="Ampiezza")
    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=30, b=40),
        height=420,
        paper_bgcolor="#020617",
        plot_bgcolor="#0a1628",
    )
    return fig

# ---------------------- Quality checks ---------------------- #
def _load_quality_config() -> Dict[str, Any]:
    """Load quality check configuration from effective config (default + override)."""
    defaults = {
        "gap_factor_k": 5.0,
        "spike_z": 4.0,
        "min_points": 20,
        "max_examples": 5
    }
    try:
        return app_settings.effective_config().get("quality", defaults)
    except Exception:
        return defaults


def _load_performance_config() -> Dict[str, Any]:
    """Load performance configuration from effective config (default + override)."""
    defaults = {
        "optimize_dtypes": True,
        "aggressive_dtype_optimization": False
    }
    try:
        return app_settings.effective_config().get("performance", defaults)
    except Exception:
        return defaults


def _render_quality_badge_and_details(report: DataQualityReport) -> None:
    """Render quality badge and collapsible details panel."""
    # Badge styling
    if report.status == 'ok':
        badge_color = "#28a745"
        badge_text = "OK"
        badge_icon = ""
    else:
        badge_color = "#ffc107"
        badge_text = "Attenzione"
        badge_icon = ""

    # Count issues
    issue_count = len(report.issues)
    issue_summary = f" ({issue_count} problema{'i' if issue_count != 1 else ''})" if issue_count > 0 else ""
    badge_icon_safe = html.escape(badge_icon)
    badge_text_safe = html.escape(badge_text)
    issue_summary_safe = html.escape(issue_summary)

    st.markdown(
        f"""
        <div style="display: inline-flex; align-items: center; gap: 8px;
                    padding: 8px 16px; border-radius: 8px; margin: 8px 0;
                    background-color: {badge_color}15; border-left: 4px solid {badge_color};">
            <span style="font-size: 1.2rem;">{badge_icon_safe}</span>
            <span style="font-weight: 600; color: {badge_color};">
                Qualità dati: {badge_text_safe}{issue_summary_safe}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Details panel - NEVER auto-expand
    if report.has_issues() or report.notes:
        with st.expander("Dettagli qualità", expanded=False):
            # Configuration info
            st.markdown("**Configurazione controlli:**")
            config_cols = st.columns(3)
            with config_cols[0]:
                st.metric(
                    "Gap factor (k)",
                    f"{report.config['gap_factor_k']:.1f}",
                    help="Moltiplicatore per rilevare gap nel campionamento. Un gap viene segnalato quando la distanza tra due punti supera k volte la distanza mediana. Valori più alti = meno segnalazioni."
                )
            with config_cols[1]:
                st.metric(
                    "Soglia Z-score",
                    f"{report.config['spike_z']:.1f}",
                    help="Sensibilità per rilevare outlier (spike) nei dati. Valori che superano questa soglia rispetto alla mediana vengono segnalati. Valori più bassi = più sensibile."
                )
            with config_cols[2]:
                st.metric(
                    "Min punti",
                    report.config['min_points'],
                    help="Numero minimo di punti necessari per eseguire i controlli statistici. Dataset con meno punti non vengono analizzati."
                )

            st.markdown("---")

            # Notes
            if report.notes:
                st.markdown("**Note informative:**")
                for note in report.notes:
                    st.info(note)

            # Issues
            if report.has_issues():
                st.markdown("**Problemi rilevati:**")
                for idx, issue in enumerate(report.issues, 1):
                    with st.container():
                        if issue.issue_type == 'x_non_monotonic':
                            st.markdown(f"**{idx}. X non monotono**")
                            st.markdown(
                                f"- **Violazioni:** {issue.count} ({issue.percentage:.2f}% dei punti)\n"
                                f"- **Descrizione:** L'asse X contiene valori duplicati o decrescenti"
                            )
                            if issue.examples:
                                with st.expander(f"Mostra {len(issue.examples)} esempi", expanded=False):
                                    for ex in issue.examples:
                                        st.code(
                                            f"Indice {ex['prev_index']} → {ex['index']}: "
                                            f"{ex['prev_value']} → {ex['value']}",
                                            language=None
                                        )

                        elif issue.issue_type == 'x_gap':
                            median_dt = issue.details.get('median_dt', 'n/a')
                            k = issue.details.get('gap_factor_k', 'n/a')
                            st.markdown(f"**{idx}. Gap nel campionamento**")
                            st.markdown(
                                f"- **Gap rilevati:** {issue.count} ({issue.percentage:.2f}% degli intervalli)\n"
                                f"- **Δt mediano:** {median_dt:.4g} unità\n"
                                f"- **Soglia:** {k}× Δt mediano"
                            )
                            if issue.examples:
                                with st.expander(f"Mostra {len(issue.examples)} esempi", expanded=False):
                                    for ex in issue.examples:
                                        st.code(
                                            f"Indice {ex['prev_index']} → {ex['index']}: "
                                            f"gap={ex['gap_size']:.4g} ({ex['gap_ratio']:.1f}×mediana)",
                                            language=None
                                        )

                        elif issue.issue_type == 'y_spike':
                            median_y = issue.details.get('median', 'n/a')
                            mad = issue.details.get('mad', 'n/a')
                            spike_z = issue.details.get('spike_z', 'n/a')
                            col_name = issue.column or 'n/a'
                            st.markdown(f"**{idx}. Spike in '{col_name}'**")
                            st.markdown(
                                f"- **Outlier rilevati:** {issue.count} ({issue.percentage:.2f}% dei punti)\n"
                                f"- **Mediana:** {median_y:.4g}\n"
                                f"- **MAD:** {mad:.4g}\n"
                                f"- **Soglia Z:** {spike_z}"
                            )
                            if issue.examples:
                                with st.expander(f"Mostra {len(issue.examples)} esempi (ordinati per |Z|)", expanded=False):
                                    for ex in issue.examples:
                                        st.code(
                                            f"Indice {ex['index']}: valore={ex['value']:.4g}, "
                                            f"Z-score={ex['z_score']:.2f}",
                                            language=None
                                        )

                        st.markdown("")  # Spacing between issues


# --- HEADER pulito (senza riquadro), logo SINISTRA + bottoni piccoli ---
def render_header():
    import html as _html

    upload = st.session_state.get("file_upload")
    sample_name = st.session_state.get("_sample_file_name")

    if upload is not None:
        fname = _html.escape(upload.name)
        file_part = f'<span class="eng-fname">&#128196; {fname}</span>'
    elif sample_name:
        fname = _html.escape(sample_name)
        file_part = f'<span class="eng-fname">&#128196; {fname}</span>'
    else:
        file_part = '<span class="eng-fname-empty">&#8212; nessun file caricato &#8212;</span>'

    # Sinusoide "segnale": animata quando c'e' un file (in caricamento/caricato), piatta altrimenti
    has_file = upload is not None or bool(sample_name)
    wave_active = "eng-wave--active" if has_file else ""
    wave_title = "segnale attivo" if has_file else "nessun segnale"

    st.markdown(
        f"""
        <style>
          /* ===================== ENGINEERING DARK THEME ===================== */

          /* Top bar */
          .eng-topbar {{
            display: flex;
            align-items: center;
            gap: 18px;
            padding: 9px 16px;
            background: #0f172a;
            border-bottom: 1px solid #1e293b;
            margin: -1rem -1rem 1.4rem -1rem;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.85rem;
          }}
          .eng-logo {{
            font-weight: 700;
            letter-spacing: 0.14em;
            color: #f1f5f9;
            font-size: 0.88rem;
          }}
          .eng-connected {{
            display: flex;
            align-items: center;
            gap: 5px;
            color: #10b981;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.1em;
          }}
          .eng-dot {{
            font-size: 0.6rem;
            animation: eng-pulse 2.2s ease-in-out infinite;
          }}
          @keyframes eng-pulse {{
            0%, 100% {{ opacity: 1; }}
            50%        {{ opacity: 0.35; }}
          }}
          .eng-fname {{ color: #64748b; font-size: 0.76rem; }}
          .eng-fname-empty {{ color: #334155; font-size: 0.76rem; font-style: italic; }}
          .eng-spacer {{ flex: 1; }}
          /* Sinusoide "segnale": linea piatta a riposo, onda che scorre durante il caricamento */
          .eng-wave {{
            display: block;
            width: 120px;
            height: 22px;
            overflow: hidden;
          }}
          .eng-wave svg {{ display: block; width: 120px; height: 22px; }}
          .eng-wave path {{
            fill: none;
            stroke: #1e293b;
            stroke-width: 1.6;
            stroke-linecap: round;
            transition: stroke 0.3s ease;
          }}
          .eng-wave-g {{ transform: translateX(0); }}
          /* Stato attivo: file caricato/in caricamento -> onda verde che trasla */
          .eng-wave--active path {{ stroke: #10b981; }}
          .eng-wave--active .eng-wave-g {{
            animation: eng-wave-move 2.2s linear infinite;
          }}
          @keyframes eng-wave-move {{
            from {{ transform: translateX(0); }}
            to   {{ transform: translateX(-100px); }}
          }}

          /* Sidebar background */
          [data-testid="stSidebar"],
          [data-testid="stSidebar"] > div:first-child {{
            background-color: #0f172a !important;
            border-right: 1px solid #1e293b !important;
          }}

          /* Sidebar section headers (### markdown) */
          [data-testid="stSidebar"] h3 {{
            font-family: 'Courier New', Courier, monospace !important;
            font-size: 0.65rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.14em !important;
            text-transform: uppercase !important;
            color: #10b981 !important;
            margin: 1.1rem 0 0.35rem !important;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 4px;
          }}

          /* Expander (FILTRO, FFT sections) */
          [data-testid="stSidebar"] [data-testid="stExpander"] {{
            border: 1px solid #1e293b !important;
            border-radius: 2px !important;
            background: transparent !important;
            margin-bottom: 4px;
          }}
          [data-testid="stSidebar"] details > summary p {{
            font-family: 'Courier New', Courier, monospace !important;
            font-size: 0.68rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
            color: #10b981 !important;
          }}

          /* Inputs — monospace */
          input[type="number"], input[type="text"] {{
            font-family: 'Courier New', Courier, monospace !important;
            font-size: 0.88rem !important;
          }}
          [data-testid="stSidebar"] input:focus {{
            border-color: #10b981 !important;
            box-shadow: 0 0 0 2px #10b98125 !important;
          }}

          /* Labels */
          [data-testid="stSidebar"] label > div > p,
          [data-testid="stSidebar"] label > p {{
            font-family: 'Courier New', Courier, monospace !important;
            font-size: 0.72rem !important;
            color: #94a3b8 !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
          }}

          /* Caption */
          [data-testid="stCaptionContainer"] p {{
            font-family: 'Courier New', Courier, monospace !important;
            font-size: 0.7rem !important;
            color: #475569 !important;
          }}

          /* Checkbox label */
          [data-testid="stCheckbox"] label p {{
            font-family: 'Courier New', Courier, monospace !important;
            font-size: 0.8rem !important;
          }}

          /* Alert / success badge */
          [data-testid="stAlert"] {{
            border-radius: 2px !important;
            font-family: 'Courier New', Courier, monospace !important;
            font-size: 0.78rem !important;
          }}

          /* HR separator in sidebar */
          [data-testid="stSidebar"] hr {{
            border-color: #1e293b !important;
            margin: 6px 0 !important;
          }}

          /* Form submit / primary button */
          [data-testid="stFormSubmitButton"] button,
          button[kind="primary"] {{
            font-family: 'Courier New', Courier, monospace !important;
            font-weight: 700 !important;
            letter-spacing: 0.1em !important;
            text-transform: uppercase !important;
            background: #10b981 !important;
            color: #020617 !important;
            border: none !important;
            border-radius: 2px !important;
          }}
          [data-testid="stFormSubmitButton"] button:hover {{
            background: #059669 !important;
          }}

          /* Metric labels */
          [data-testid="stMetricLabel"] p {{
            font-family: 'Courier New', Courier, monospace !important;
            font-size: 0.72rem !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            color: #64748b !important;
          }}
          [data-testid="stMetricValue"] {{
            font-family: 'Courier New', Courier, monospace !important;
            color: #f1f5f9 !important;
          }}

          /* File uploader */
          [data-testid="stFileUploader"] {{
            border: 1px dashed #1e293b !important;
            border-radius: 3px !important;
          }}
        </style>

        <div class="eng-topbar">
          <span class="eng-logo">[ CSV_ANALYZER ]</span>
          <span class="eng-connected"><span class="eng-dot">&#9679;</span>&nbsp;CONNECTED</span>
          {file_part}
          <span class="eng-spacer"></span>
          <span class="eng-wave {wave_active}" aria-hidden="true" title="{wave_title}">
            <svg viewBox="0 0 100 24" preserveAspectRatio="none">
              <g class="eng-wave-g">
                <path d="M0,12 Q6.25,4 12.5,12 Q18.75,20 25,12 Q31.25,4 37.5,12 Q43.75,20 50,12 Q56.25,4 62.5,12 Q68.75,20 75,12 Q81.25,4 87.5,12 Q93.75,20 100,12 Q106.25,4 112.5,12 Q118.75,20 125,12 Q131.25,4 137.5,12 Q143.75,20 150,12 Q156.25,4 162.5,12 Q168.75,20 175,12 Q181.25,4 187.5,12 Q193.75,20 200,12" />
              </g>
            </svg>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------- UI principale ---------------------- #
def _reset_generated_reports_marker(current_file: Optional[Any]) -> None:
    """Reset session-state outputs when the uploaded file changes."""

    file_id = None
    if current_file is not None:
        file_id = (current_file.name, getattr(current_file, "size", None))

    last_id = st.session_state.get("_last_uploaded_file_id")
    if last_id != file_id:
        st.session_state["_last_uploaded_file_id"] = file_id
        st.session_state.pop("_generated_report", None)
        st.session_state.pop("_generated_report_error", None)
        st.session_state.pop("_generated_visual_report", None)
        st.session_state.pop("_generated_visual_report_error", None)
        st.session_state.pop("_visual_report_prev_selection", None)
        st.session_state.pop("_visual_report_last_default_x_label", None)
        st.session_state.pop("_plots_ready", None)
        st.session_state.pop("_cached_df", None)
        st.session_state.pop("_cached_meta", None)
        st.session_state.pop("_cached_cleaning_report", None)
        st.session_state.pop("_cached_file_sig", None)
        st.session_state.pop("_cached_apply_cleaning", None)
        # Issue #35: Invalidate filter/FFT caches when file changes
        _invalidate_result_caches()
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and key.startswith("vis_report_"):
                st.session_state.pop(key, None)


def _visual_spec_key(field: str, column: str) -> str:
    return f"vis_report_{field}::{column}"


def _sync_visual_spec_state(selection: Sequence[str], default_x_label: str) -> None:
    """Ensure per-column widget keys exist and purge deselected ones."""

    prev = st.session_state.get("_visual_report_prev_selection", [])
    removed = set(prev) - set(selection)
    for col in removed:
        for field in ("title", "xlabel", "ylabel"):
            st.session_state.pop(_visual_spec_key(field, col), None)

    st.session_state["_visual_report_prev_selection"] = list(selection)

    for col in selection:
        title_key = _visual_spec_key("title", col)
        if title_key not in st.session_state:
            st.session_state[title_key] = col

        xlabel_key = _visual_spec_key("xlabel", col)
        if xlabel_key not in st.session_state:
            st.session_state[xlabel_key] = default_x_label

        ylabel_key = _visual_spec_key("ylabel", col)
        if ylabel_key not in st.session_state:
            st.session_state[ylabel_key] = col


def _apply_preset_to_widgets(preset_data: Dict[str, Any]) -> None:
    """Scrive i valori di un preset nelle chiavi dei widget della sidebar.

    DEVE essere chiamata PRIMA che i widget vengano creati nel run corrente
    (Streamlit vieta di modificare la session_state di un widget già istanziato).
    """
    filter_kind_options = [
        "Media mobile (MA)",
        "Butterworth LP",
        "Butterworth HP",
        "Butterworth BP",
    ]
    kind_to_index = {"ma": 0, "butter_lp": 1, "butter_hp": 2, "butter_bp": 3}

    manual = preset_data.get("manual_fs")
    try:
        st.session_state["manual_fs"] = float(manual) if manual is not None else 0.0
    except (TypeError, ValueError):
        st.session_state["manual_fs"] = 0.0

    fspec = preset_data.get("filter_spec")
    if isinstance(fspec, FilterSpec):
        st.session_state["enable_filter"] = bool(fspec.enabled)
        st.session_state["f_kind"] = filter_kind_options[kind_to_index.get(fspec.kind, 0)]
        try:
            st.session_state["ma_win"] = int(fspec.ma_window)
        except (TypeError, ValueError):
            pass
        try:
            st.session_state["f_order"] = int(fspec.order)
        except (TypeError, ValueError):
            pass
        cutoff = fspec.cutoff or (None, None)
        lo, hi = cutoff if isinstance(cutoff, tuple) else (None, None)
        st.session_state["f_lo"] = "" if lo in (None, "") else str(lo)
        st.session_state["f_hi"] = "" if hi in (None, "") else str(hi)
        st.session_state["overlay_orig"] = True

    fft = preset_data.get("fft_spec")
    if isinstance(fft, FFTSpec):
        st.session_state["sidebar_enable_fft"] = bool(fft.enabled)
        st.session_state["sidebar_detrend"] = bool(fft.detrend)


_SETTINGS_SLIDER_RANGES = {
    "set_max_file_mb": (50, 20000),
    "set_max_rows_milioni": (1, 200),
    "set_timeout_s": (30, 3600),
    "set_chunk_size": (50000, 2000000),
    "set_chunked_threshold_mb": (10, 1000),
    "set_max_cols": (10, 50000),
}

_SETTINGS_KEYS = (
    "set_max_file_mb", "set_max_rows_milioni", "set_max_cols", "set_timeout_s",
    "set_optimize_dtypes", "set_use_optimized_loader", "set_chunked_threshold_mb",
    "set_chunk_size", "set_use_pyarrow", "set_parallel_cleaning", "set_max_workers",
)


def _settings_defaults_from_config(cfg: Dict[str, Any], cores: int) -> Dict[str, Any]:
    """Valori iniziali dei widget Impostazioni dalla config effettiva."""
    lim = cfg.get("limits", {})
    perf = cfg.get("performance", {})
    adv = perf.get("advanced", {})
    return {
        "set_max_file_mb": int(lim.get("max_file_mb", 2500)),
        "set_max_rows_milioni": max(1, int(round(lim.get("max_rows", 20000000) / 1_000_000))),
        "set_max_cols": int(lim.get("max_cols", 5000)),
        "set_timeout_s": int(lim.get("parse_timeout_s", 1200)),
        "set_optimize_dtypes": bool(perf.get("optimize_dtypes", True)),
        "set_use_optimized_loader": bool(perf.get("use_optimized_loader", True)),
        "set_chunked_threshold_mb": int(perf.get("chunked_loading_threshold_mb", 300)),
        "set_chunk_size": int(perf.get("chunk_size", 600000)),
        "set_use_pyarrow": bool(adv.get("use_pyarrow", True)),
        "set_parallel_cleaning": bool(adv.get("parallel_cleaning", True)),
        "set_max_workers": int(adv.get("max_workers") or cores),
    }


def _clamp_settings_state(cores: int, defaults: Dict[str, Any]) -> None:
    """Riporta i valori numerici dentro il range dei widget (evita errori slider)."""
    ranges = dict(_SETTINGS_SLIDER_RANGES)
    ranges["set_max_workers"] = (1, max(1, cores))
    for key, (lo, hi) in ranges.items():
        try:
            st.session_state[key] = min(hi, max(lo, int(st.session_state[key])))
        except (KeyError, TypeError, ValueError):
            st.session_state[key] = defaults[key]


def _apply_profile_to_settings_widgets(profile: str, cores: int) -> None:
    """Scrive i valori di un profilo nelle chiavi dei widget Impostazioni."""
    vals = app_settings.profile_settings(profile, cores)
    lim = vals["limits"]
    perf = vals["performance"]
    adv = perf.get("advanced", {})
    st.session_state["set_max_file_mb"] = int(lim["max_file_mb"])
    st.session_state["set_max_rows_milioni"] = max(1, int(round(lim["max_rows"] / 1_000_000)))
    st.session_state["set_max_cols"] = int(lim["max_cols"])
    st.session_state["set_timeout_s"] = int(lim["parse_timeout_s"])
    st.session_state["set_optimize_dtypes"] = bool(perf["optimize_dtypes"])
    st.session_state["set_use_optimized_loader"] = bool(perf["use_optimized_loader"])
    st.session_state["set_chunked_threshold_mb"] = int(perf["chunked_loading_threshold_mb"])
    st.session_state["set_chunk_size"] = int(perf["chunk_size"])
    st.session_state["set_use_pyarrow"] = bool(adv.get("use_pyarrow", True))
    st.session_state["set_parallel_cleaning"] = bool(adv.get("parallel_cleaning", True))
    st.session_state["set_max_workers"] = int(adv.get("max_workers") or cores)


def _render_settings_panel() -> None:
    """Pannello Impostazioni: profili prestazioni + tuning di limiti/RAM."""
    st.markdown("### Impostazioni prestazioni")
    st.caption(
        "Regola la capacità di caricamento e l'uso di RAM/CPU. "
        "Le impostazioni vengono salvate sul tuo PC e mantenute al riavvio."
    )

    hw = app_settings.detect_hardware()
    cores = int(hw["cores"])
    ram_gb = hw["ram_gb"]
    recommended = app_settings.recommend_profile(ram_gb, cores)

    ram_txt = f"{ram_gb:.1f} GB" if ram_gb else "non rilevata"
    c1, c2, c3 = st.columns(3)
    c1.metric("RAM rilevata", ram_txt)
    c2.metric("Core CPU", str(cores))
    c3.metric("Profilo consigliato", app_settings.PROFILE_LABELS.get(recommended, recommended))

    # Inizializza le chiavi dei widget UNA volta dalla config effettiva, poi usa
    # solo key= (niente value=) per non innescare il warning di Streamlit quando
    # i valori vengono impostati via Session State (es. "Applica profilo").
    cfg = app_settings.effective_config()
    defaults = _settings_defaults_from_config(cfg, cores)
    for _k, _v in defaults.items():
        st.session_state.setdefault(_k, _v)
    _clamp_settings_state(cores, defaults)

    # --- Profili rapidi ---
    profile_keys = ["leggero", "bilanciato", "qualita"]
    profile_labels = [app_settings.PROFILE_LABELS[k] for k in profile_keys]

    st.markdown("#### Profilo rapido")
    pcol1, pcol2 = st.columns([3, 1])
    with pcol1:
        chosen_label = st.radio(
            "Profilo",
            profile_labels,
            index=profile_keys.index(recommended),
            horizontal=True,
            label_visibility="collapsed",
            key="set_profile_choice",
        )
    chosen_profile = profile_keys[profile_labels.index(chosen_label)]
    with pcol2:
        if st.button("Applica profilo", key="apply_profile_btn", use_container_width=True):
            _apply_profile_to_settings_widgets(chosen_profile, cores)
            st.session_state["_settings_applied_profile"] = chosen_profile
            st.rerun()

    if st.session_state.get("_settings_applied_profile"):
        st.info(
            f"Profilo '{app_settings.PROFILE_LABELS[st.session_state['_settings_applied_profile']]}' "
            "applicato ai valori sotto. Premi Salva per renderlo permanente."
        )

    st.markdown("#### Capacità di caricamento file")
    st.slider(
        "Dimensione massima file (MB)", min_value=50, max_value=20000, step=50,
        key="set_max_file_mb",
        help="File più grandi di questo verranno rifiutati. Si applica al riavvio.",
    )
    st.slider(
        "Righe massime (milioni)", min_value=1, max_value=200, step=1,
        key="set_max_rows_milioni",
    )
    cc1, cc2 = st.columns(2)
    with cc1:
        st.number_input(
            "Colonne massime", min_value=10, max_value=50000, step=10,
            key="set_max_cols",
        )
    with cc2:
        st.slider(
            "Timeout caricamento (s)", min_value=30, max_value=3600, step=30,
            key="set_timeout_s",
        )

    st.markdown("#### RAM e velocità")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.toggle(
            "Ottimizza RAM (dtype)", key="set_optimize_dtypes",
            help="Riduce la memoria usata convertendo le colonne numeriche a tipi più leggeri.",
        )
        st.toggle(
            "Loader ottimizzato (chunked)", key="set_use_optimized_loader",
            help="Carica i file grandi a blocchi per ridurre il picco di RAM.",
        )
    with rc2:
        st.toggle("Lettura veloce (pyarrow)", key="set_use_pyarrow")
        st.toggle("Pulizia in parallelo", key="set_parallel_cleaning")
    st.slider(
        "Core CPU da usare (worker)", min_value=1, max_value=max(1, cores), step=1,
        key="set_max_workers",
    )
    st.slider(
        "Dimensione blocco (righe per chunk)", min_value=50000, max_value=2000000, step=50000,
        key="set_chunk_size",
        help="Blocchi più grandi = più veloce ma più RAM.",
    )
    st.slider(
        "Soglia caricamento a blocchi (MB)", min_value=10, max_value=1000, step=10,
        key="set_chunked_threshold_mb",
    )

    st.caption(
        "Alcune impostazioni (dimensione massima file, blocco, soglia, worker) "
        "hanno pieno effetto al riavvio dell'app."
    )

    # --- Salva / Ripristina ---
    st.markdown("---")
    scol1, scol2 = st.columns(2)
    with scol1:
        if st.button("Salva impostazioni", type="primary", key="save_settings_btn", use_container_width=True):
            new_settings = {
                "limits": {
                    "max_file_mb": int(st.session_state["set_max_file_mb"]),
                    "max_rows": int(st.session_state["set_max_rows_milioni"]) * 1_000_000,
                    "max_cols": int(st.session_state["set_max_cols"]),
                    "parse_timeout_s": int(st.session_state["set_timeout_s"]),
                },
                "performance": {
                    "optimize_dtypes": bool(st.session_state["set_optimize_dtypes"]),
                    "use_optimized_loader": bool(st.session_state["set_use_optimized_loader"]),
                    "chunked_loading_threshold_mb": int(st.session_state["set_chunked_threshold_mb"]),
                    "chunk_size": int(st.session_state["set_chunk_size"]),
                    "advanced": {
                        "use_pyarrow": bool(st.session_state["set_use_pyarrow"]),
                        "parallel_cleaning": bool(st.session_state["set_parallel_cleaning"]),
                        "max_workers": int(st.session_state["set_max_workers"]),
                    },
                },
            }
            try:
                app_settings.save_user_settings(new_settings)
                st.session_state.pop("_settings_applied_profile", None)
                st.success("Impostazioni salvate. Riavvia l'app per applicare la dimensione massima file.")
            except Exception as e:
                st.error(f"Errore nel salvataggio: {e}")
    with scol2:
        if st.button("Ripristina default", key="reset_settings_btn", use_container_width=True):
            app_settings.reset_user_settings()
            for _k in _SETTINGS_KEYS + ("_settings_applied_profile",):
                st.session_state.pop(_k, None)
            st.success("Impostazioni ripristinate ai valori predefiniti.")
            st.rerun()

    st.caption(f"File impostazioni: {app_settings.user_settings_path()}")


def _render_main_app():
    # FIX ISSUE #54: Inizializza logger per web_app
    logger = LogManager(component="web_app").get_logger()

    # Inizializza preset di default all'avvio
    try:
        create_default_presets()
    except Exception as e:
        # Usa logger diretto senza variabile per evitare UnboundLocalError
        LogManager(component="preset").get_logger().warning(f"Impossibile creare preset default: {e}")

    st.caption("Upload CSV → seleziona X/Y → limiti assi → Sidebar (fs/filtri/FFT) → report")

    st.markdown(
        """
        <style>
        .file-upload-wrapper {
            position: relative;
        }
        .file-upload-wrapper div[data-testid="stFileUploader"] > div:first-child {
            padding-bottom: 5.6rem;
        }
        .file-upload-wrapper div[data-testid="stButton"] {
            position: absolute;
            right: 1.2rem;
            bottom: 1.2rem;
            margin: 0;
            width: 200px;
            z-index: 2;
        }
        .file-upload-wrapper div[data-testid="stButton"] button {
            width: 100%;
            min-height: 3rem;
            border-radius: 12px;
            background: linear-gradient(135deg, #2b2d35 0%, #1f2027 100%);
            color: #f1f3f6;
            font-weight: 600;
            border: 1px solid #34353d;
            transition: all .18s ease-in-out;
        }
        .file-upload-wrapper div[data-testid="stButton"] button:hover {
            background: linear-gradient(135deg, #353741 0%, #2a2b33 100%);
            border-color: #4b4d58;
        }
        .file-upload-wrapper div[data-testid="stButton"] button:active {
            transform: translateY(1px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.pop("_clear_file_uploader", False):
        st.session_state.pop("file_upload", None)

    sample_bytes = st.session_state.get("_sample_bytes")
    sample_name = st.session_state.get("_sample_file_name", SAMPLE_CSV_PATH.name)

    sample_available = SAMPLE_CSV_PATH.exists()

    with st.container():
        st.markdown('<div class="file-upload-wrapper">', unsafe_allow_html=True)
        upload = st.file_uploader(
            "Carica un file CSV",
            type=["csv"],
            key="file_upload",
        )
        sample_disabled = not sample_available or upload is not None
        sample_help = (
            "Devi eliminare il CSV in memoria prima di caricare il sample."
            if upload is not None
            else "Carica un dataset demo multi-canale (segnale + rumore)."
        )
        sample_clicked = st.button(
            "Carica sample",
            key="load_sample",
            disabled=sample_disabled,
            help=sample_help,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if sample_clicked:
        if sample_available:
            try:
                data = SAMPLE_CSV_PATH.read_bytes()
                st.session_state["_sample_bytes"] = data
                st.session_state["_sample_file_name"] = SAMPLE_CSV_PATH.name
                st.session_state["_clear_file_uploader"] = True
                st.session_state.pop("_sample_error", None)

                # FIX #46: Libera TUTTA la cache upload prima del rerun
                st.session_state.pop("_cached_df", None)
                st.session_state.pop("_cached_cleaning_report", None)
                st.session_state.pop("_cached_meta", None)
                st.session_state.pop("_cached_file_sig", None)
                st.session_state.pop("_cached_apply_cleaning", None)
                _invalidate_result_caches()  # Pulisce filter/FFT cache

                st.rerun()
            except Exception as exc:
                st.session_state.pop("_sample_bytes", None)
                st.session_state.pop("_sample_file_name", None)
                st.session_state["_sample_error"] = str(exc)
                st.rerun()
    if not sample_available:
        st.caption("Sample non disponibile.")

    sample_error = st.session_state.pop("_sample_error", None)
    if sample_error:
        st.error(f"Caricamento sample fallito: {sample_error}")

    sample_bytes = st.session_state.get("_sample_bytes")
    sample_name = st.session_state.get("_sample_file_name", SAMPLE_CSV_PATH.name)

    if upload is not None:
        st.session_state.pop("_sample_bytes", None)
        st.session_state.pop("_sample_file_name", None)
        sample_bytes = None
        sample_name = SAMPLE_CSV_PATH.name

    current_file: Optional[Any] = upload
    if current_file is None and sample_bytes is not None:
        current_file = SimpleNamespace(name=sample_name, size=len(sample_bytes))

    _reset_generated_reports_marker(current_file)

    if current_file is None:
        hint = "Carica un file per iniziare."
        if SAMPLE_CSV_PATH.exists():
            hint = "Carica un file oppure usa 'Carica sample' per iniziare."
        st.info(hint)
        return

    using_sample = upload is None

    limits = _load_limits_config()

    apply_cleaning = st.checkbox(
        "Applica correzione suggerita",
        value=False,
        key="_apply_cleaning",
        help="Rimuove separatori migliaia/decimali incoerenti e converte le colonne numeriche.",
    )

    cleaning_report: Optional[CleaningReport] = None
    meta: Dict[str, Any]
    file_bytes: bytes

    if using_sample:
        if sample_bytes is None:
            st.error("Sample non disponibile.")
            return
        size_error = _check_size_limit(len(sample_bytes), limits)
        if size_error:
            _clear_cached_dataset()
            st.error(size_error)
            st.stop()
        file_bytes = sample_bytes
    else:
        upload_size = getattr(upload, "size", 0)
        size_error = _check_size_limit(upload_size, limits)
        if size_error:
            _clear_cached_dataset()
            st.error(size_error)
            st.stop()
        upload_bytes = upload.getvalue()
        if not upload_bytes:
            st.error("Il file caricato è vuoto.")
            return
        file_bytes = upload_bytes
        upload.seek(0)

    file_sig = _build_file_signature(file_bytes)

    cached_df = st.session_state.get("_cached_df")
    cached_report = st.session_state.get("_cached_cleaning_report")
    cached_meta = st.session_state.get("_cached_meta")
    cache_hit = (
        st.session_state.get("_cached_file_sig") == file_sig
        and st.session_state.get("_cached_apply_cleaning") == apply_cleaning
        and cached_df is not None
        and cached_report is not None
        and cached_meta is not None
    )

    df: pd.DataFrame

    try:
        with st.spinner("Analisi CSV..."):
            if cache_hit:
                # FIX ISSUE #51: Proteggi cache da mutazioni con .copy()
                df = cached_df.copy()  # type: ignore[assignment,union-attr]
                # CleaningReport è immutabile (dataclass con campi readonly), shallow copy sufficiente
                from dataclasses import replace as dataclass_replace
                cleaning_report = dataclass_replace(cached_report)  # type: ignore[assignment,arg-type]
                meta = dict(cached_meta)  # type: ignore[arg-type]
                limit_error = _check_dataframe_limits(df, limits)
                if limit_error:
                    _clear_cached_dataset()
                    st.error(limit_error)
                    st.stop()
            else:
                timeout_s = max(limits["parse_timeout_s"], 1.0)
                df, cleaning_report, meta = _parse_csv_with_timeout(
                    file_bytes=file_bytes,
                    apply_cleaning=apply_cleaning,
                    timeout_s=timeout_s,
                )
                limit_error = _check_dataframe_limits(df, limits)
                if limit_error:
                    _clear_cached_dataset()
                    st.error(limit_error)
                    st.stop()
                st.session_state["_cached_df"] = df
                st.session_state["_cached_cleaning_report"] = cleaning_report
                st.session_state["_cached_meta"] = dict(meta)
                st.session_state["_cached_file_sig"] = file_sig
                st.session_state["_cached_apply_cleaning"] = apply_cleaning
    except TimeoutError as exc:
        st.error(str(exc))
        return
    except pd.errors.EmptyDataError:
        st.error(
            "Il file sembra vuoto (nessuna colonna rilevata). Verifica l'esportazione e riprova."
        )
        return
    except ValueError as ve:
        st.error(str(ve))
        return
    except Exception as exc:
        # FIX ISSUE #54: Messaggio generico utente, log tecnico con traceback
        st.error("Errore nel parsing del CSV. Verifica il formato del file.")
        logger.error(
            "CSV parsing failed",
            exc_info=True,
            extra={
                "file_size": len(file_bytes),
                "apply_cleaning": apply_cleaning,
                "session_id": st.session_state.get("_dataset_id", "")[:8]
            }
        )
        return

    if cleaning_report is None:
        st.error("Impossibile generare il report di sanificazione del CSV.")
        return

    meta["cleaning"] = cleaning_report.to_dict()
    st.session_state["_cached_meta"] = dict(meta)
    st.session_state["_cached_file_sig"] = file_sig
    st.session_state["_cached_apply_cleaning"] = apply_cleaning

    if using_sample:
        st.success(f"Sample '{sample_name}' caricato.")
    else:
        st.success("File caricato.")

    # Run quality checks (with cache: avoids O(n×cols) ricalcolo ad ogni rerender)
    _init_result_caches()
    quality_config = _load_quality_config()
    try:
        all_cols = list(df.columns)
        _quality_key = (
            file_sig,
            tuple(all_cols),
            (quality_config['gap_factor_k'], quality_config['spike_z'], quality_config['min_points']),
        )
        quality_report = st.session_state["_quality_cache"].get(_quality_key)
        if quality_report is None:
            quality_report = run_quality_checks(
                df=df,
                x_col=None,
                y_cols=all_cols,
                gap_factor_k=quality_config['gap_factor_k'],
                spike_z=quality_config['spike_z'],
                min_points=quality_config['min_points'],
                max_examples=quality_config['max_examples']
            )
            _qcache = st.session_state["_quality_cache"]
            if len(_qcache) >= MAX_QUALITY_CACHE_SIZE:
                _qcache.pop(next(iter(_qcache)))
            _qcache[_quality_key] = quality_report
            quality_logger = LogManager(component="quality").get_logger()
            quality_logger.info(quality_report.get_summary())

        _render_quality_badge_and_details(quality_report)
    except Exception as e:
        st.warning(f"Impossibile eseguire controlli qualità: {e}")

    with st.expander("Dettagli dati", expanded=False):
        suggestion = cleaning_report.suggestion
        info_cols = st.columns(4)
        encoding_value = meta.get('encoding') or 'utf-8'
        info_cols[0].markdown(
            _meta_info_html("Encoding", encoding_value),
            unsafe_allow_html=True,
        )
        info_cols[1].markdown(
            _meta_info_html("Delimiter", _fmt_csv_token(meta.get('delimiter'))),
            unsafe_allow_html=True,
        )
        info_cols[2].markdown(
            _meta_info_html("Decimal", _fmt_csv_token(suggestion.decimal)),
            unsafe_allow_html=True,
        )
        info_cols[3].markdown(
            _meta_info_html("Migliaia", _fmt_csv_token(suggestion.thousands)),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Correzione automatica: {'attiva' if apply_cleaning else 'disattivata'} - "
            f"Confidenza formato: {suggestion.confidence:.0%} (campione={suggestion.sample_size})"
        )

        if cleaning_report.warnings:
            for warn in cleaning_report.warnings:
                st.warning(warn)

        stats_df = _cleaning_stats_table(cleaning_report)
        if not stats_df.empty:
            st.markdown("**Qualità colonne numeriche**")
            _dataframe(stats_df)
        else:
            st.caption("Nessuna colonna numerica rilevata.")

        if cleaning_report.rows_all_nan_after_clean:
            st.info(
                "Righe con tutte le colonne numeriche a NaN dopo la correzione: "
                f"{len(cleaning_report.rows_all_nan_after_clean)} "
                f"(prime: {cleaning_report.rows_all_nan_after_clean[:5]})"
            )

        raw_name = getattr(current_file, "name", "dataset.csv")
        st.download_button(
            "Scarica CSV originale",
            data=file_bytes,
            file_name=raw_name,
            mime="text/csv",
        )

    n_preview = st.slider("Righe di anteprima", 5, 50, 10)
    _dataframe(df.head(n_preview))
    total_rows = len(df)
    st.caption(f"Mostrate le prime {n_preview} righe su {total_rows} totali.")

    quality_key = "plot_quality_mode"
    file_sig_key = "_quality_file_sig"
    default_quality = "Prestazioni" if total_rows > PERFORMANCE_THRESHOLD else "Alta fedeltà"
    if st.session_state.get(file_sig_key) != file_sig:
        st.session_state[file_sig_key] = file_sig
        st.session_state[quality_key] = default_quality
    else:
        st.session_state.setdefault(quality_key, default_quality)

    # Reset forward fill e interpolazione X su nuovo file per evitare stati appiccicosi
    if st.session_state.get("_fill_file_sig") != file_sig:
        st.session_state["_fill_file_sig"] = file_sig
        st.session_state["fillna_forward"] = False
        st.session_state["interpolate_x_col"] = False
        st.session_state["_fill_last_stamp"] = False

    # Pulsante Reset impostazioni (non rimuove il file caricato)
    rc1, rc2 = st.columns([3, 1])
    with rc2:
        if _button("Reset impostazioni"):
            _reset_all_settings()

    cols = meta.get("columns", list(df.columns))
    fft_available = total_rows >= MIN_ROWS_FOR_FFT

    # Preset defaults for Advanced form
    filter_kind_options = [
        "Media mobile (MA)",
        "Butterworth LP",
        "Butterworth HP",
        "Butterworth BP",
    ]
    preset_manual_fs = 0.0
    preset_enable_filter = False
    preset_filter_kind_idx = 0
    preset_ma_win = 5
    preset_filter_order = 4
    preset_f_lo = ""
    preset_f_hi = ""
    preset_enable_fft = False
    preset_detrend = True
    preset_save_message: Optional[str] = st.session_state.pop("_preset_save_message", None)

    # Applica un preset appena caricato UNA SOLA volta, PRIMA di creare i widget
    # della sidebar (Streamlit vieta di modificarne la session_state dopo). Poi
    # NON lo ri-applichiamo, così le modifiche manuali successive vengono mantenute.
    # L'indicatore "Preset attivo" resta finché l'utente non resetta o ne carica un altro.
    if "_loaded_preset" in st.session_state:
        _apply_preset_to_widgets(st.session_state.pop("_loaded_preset"))
        st.session_state.pop("_loaded_preset_name", None)

    if preset_save_message:
        st.success(preset_save_message)

    # Inizializza UNA volta le chiavi dei widget avanzati. Poi i widget usano solo
    # key= (niente value=/index=): così l'applicazione di un preset via Session
    # State non innesca il warning "created with a default value but also had its
    # value set via the Session State API".
    _advanced_defaults = {
        "manual_fs": float(preset_manual_fs),
        "enable_filter": preset_enable_filter,
        "f_kind": filter_kind_options[preset_filter_kind_idx],
        "ma_win": int(preset_ma_win),
        "f_order": int(preset_filter_order),
        "f_lo": preset_f_lo,
        "f_hi": preset_f_hi,
        "overlay_orig": True,
        "sidebar_enable_fft": preset_enable_fft,
        "sidebar_detrend": preset_detrend,
    }
    for _k, _v in _advanced_defaults.items():
        st.session_state.setdefault(_k, _v)

    # ---- SIDEBAR: ADVANCED CONTROLS ----
    with st.sidebar:
        st.markdown("### Campionamento")
        manual_fs = st.number_input(
            "Frequenza di campionamento (Hz)",
            min_value=0.0,
            step=0.1,
            key="manual_fs",
            help=">0 forza la fs; 0 = stima automatica dalla X",
        )
        st.caption("0 = stima automatica. Filtri Butterworth e FFT useranno la stessa fs.")

        # Live fs indicator
        _cur_manual_fs = st.session_state.get("manual_fs", 0.0)
        if _cur_manual_fs and float(_cur_manual_fs) > 0:
            st.success(f"**fs = {float(_cur_manual_fs):.6g} Hz** (manuale)")
        else:
            _cached_fs = st.session_state.get("_sidebar_fs_info")
            if _cached_fs and _cached_fs.get("value"):
                st.info(f"**fs ≈ {_cached_fs['value']:.6g} Hz** ({_cached_fs['label']})")
            else:
                st.caption("fs: sarà stimata dalla colonna X al primo plot.")

        st.markdown("---")
        with st.expander("Filtro", expanded=False):
            enable_filter = st.checkbox("Abilita filtro", key="enable_filter")
            f_kind = st.selectbox(
                "Tipo filtro",
                filter_kind_options,
                key="f_kind",
            )
            ma_win = st.number_input(
                "MA - finestra (campioni)", min_value=1, step=1, key="ma_win"
            )
            f_order = st.number_input(
                "Butterworth - ordine", min_value=1, step=1, key="f_order"
            )
            cc1, cc2 = st.columns(2)
            with cc1:
                f_lo = st.text_input(
                    "Cutoff low (Hz)", placeholder="es. 5", key="f_lo"
                )
            with cc2:
                f_hi = st.text_input(
                    "Cutoff high (Hz) - solo BP", placeholder="es. 20", key="f_hi"
                )
            overlay_orig = st.checkbox(
                "Sovrapponi originale e filtrato", key="overlay_orig"
            )

        st.markdown("---")
        with st.expander("FFT", expanded=False):
            fft_help = (
                "Calcola lo spettro FFT per ogni serie selezionata."
                if fft_available
                else f"Servono almeno {MIN_ROWS_FOR_FFT} campioni per calcolare l'FFT."
            )
            enable_fft = st.checkbox(
                "Calcola FFT",
                disabled=not fft_available,
                help=fft_help,
                key="sidebar_enable_fft",
            )
            if not fft_available:
                st.caption(
                    f"Servono almeno {MIN_ROWS_FOR_FFT} campioni (dataset attuale: {total_rows})."
                )
            fft_use = st.radio(
                "FFT su",
                ["Filtrato (se attivo)", "Originale"],
                horizontal=True,
                disabled=not fft_available,
                key="sidebar_fft_use",
            )
            detrend = st.checkbox(
                "Detrend (togli media)",
                disabled=not fft_available,
                key="sidebar_detrend",
            )

        st.markdown("---")
        with st.expander("Limiti assi", expanded=False):
            _ax_last_y = [c for c in st.session_state.get("_last_y_cols", []) if c]

            st.caption("Asse X (globale)")
            _axc1, _axc2 = st.columns(2)
            with _axc1:
                st.text_input("X min", key="ax_x_min", placeholder="es. 0")
            with _axc2:
                st.text_input("X max", key="ax_x_max", placeholder="es. 1000")

            st.markdown("---")
            st.radio(
                "Scala Y",
                ["Tutti i segnali", "Per segnale"],
                horizontal=True,
                key="ax_y_mode",
            )

            if st.session_state.get("ax_y_mode", "Tutti i segnali") == "Tutti i segnali":
                _ayc1, _ayc2 = st.columns(2)
                with _ayc1:
                    st.text_input("Y min", key="ax_y_min_global", placeholder="es. -10")
                with _ayc2:
                    st.text_input("Y max", key="ax_y_max_global", placeholder="es. 10")
            else:
                if _ax_last_y:
                    for _ayname in _ax_last_y:
                        _aykey = _ayname.replace(" ", "_").replace(".", "_")
                        st.caption(f"**{_ayname}**")
                        _ayc1, _ayc2 = st.columns(2)
                        with _ayc1:
                            st.text_input("Y min", key=f"ax_y_min_{_aykey}", placeholder="auto")
                        with _ayc2:
                            st.text_input("Y max", key=f"ax_y_max_{_aykey}", placeholder="auto")
                else:
                    st.caption("Seleziona le colonne Y e premi Applica / Plot.")

            if st.button("Azzera limiti", key="ax_reset_btn"):
                for _rk in ["ax_x_min", "ax_x_max", "ax_y_mode", "ax_y_min_global", "ax_y_max_global"]:
                    st.session_state.pop(_rk, None)
                for _rkey in list(st.session_state.keys()):
                    if isinstance(_rkey, str) and (
                        _rkey.startswith("ax_y_min_") or _rkey.startswith("ax_y_max_")
                    ):
                        st.session_state.pop(_rkey, None)
                st.rerun()

    # --- Controlli (form) --- #
    with st.form(f"controls_{st.session_state.get('_controls_nonce', 0)}"):
        _col_x, _col_y = st.columns([1, 2])
        with _col_x:
            x_col = st.selectbox("Colonna X (opzionale)", options=["—"] + cols, index=0)
            fill_missing = st.checkbox(
                "Riempimento valori mancanti (forward fill)",
                key="fillna_forward",
                help="Applica un forward fill alle colonne per colmare eventuali NaN.",
            )
            interpolate_x = st.checkbox(
                "Interpola asse X (marker spaziale sparso)",
                key="interpolate_x_col",
                help=(
                    "Se la colonna X è un marker di posizione aggiornato raramente (es. ogni 250 mm), "
                    "i valori NaN intermedi vengono riempiti con interpolazione lineare invece di forward fill. "
                    "Elimina gli artefatti verticali nel grafico causati da più campioni con lo stesso X."
                ),
            )
        with _col_y:
            y_cols = st.multiselect("Colonne Y", options=cols)
        _col_mode, _col_quality = st.columns(2)
        with _col_mode:
            mode = st.radio("Modalità grafico", ["Sovrapposto", "Separati", "Cascata"], horizontal=True, index=0)
        with _col_quality:
            quality_mode = st.radio(
                "Alta fedeltà / Prestazioni",
                ["Alta fedeltà", "Prestazioni"],
                horizontal=True,
                key=quality_key,
                help="Prestazioni applica downsampling LTTB a circa 10k punti per serie per migliorare la reattività. Filtri e FFT restano sui dati completi.",
            )
            st.caption("Prestazioni = downsampling LTTB ~10k pt.")

        submitted = st.form_submit_button("Applica / Plot")

    fill_stamp = (bool(fill_missing), bool(interpolate_x))
    # Se i flag cambiano, invalida cache filter/FFT per evitare grafici stantii
    prev_fill = st.session_state.get("_fill_last_stamp", fill_stamp)
    if prev_fill != fill_stamp:
        _invalidate_result_caches()
        st.session_state["_fill_last_stamp"] = fill_stamp

    # ---- TRASFORMAZIONI (FUORI DAL FORM) ----
    with st.expander(
        "Trasformazioni",
        expanded=st.session_state.get("_n_transforms", 0) > 0,
    ):
        st.caption("Trasformazioni applicate al segnale prima del filtro, in sequenza.")

        with st.expander("Guida trasformazioni", expanded=False):
            _guide_rows = []
            for _gk, _glabel in TRANSFORM_LABELS.items():
                _g = TRANSFORM_GUIDE[_gk]
                _guide_rows.append({
                    "Tipo": _glabel,
                    "Formula": _g["formula"],
                    "Descrizione": _g["note"],
                    "Asse X": _g["x"],
                    "Cambia lunghezza": _g["len"],
                })
            st.dataframe(
                pd.DataFrame(_guide_rows).set_index("Tipo"),
                use_container_width=True,
                hide_index=False,
            )
            st.caption(
                "Le trasformazioni con 'Cambia lunghezza = Sì' (Derivata, Ricampionamento) "
                "non possono essere incluse nel report statistico insieme alle altre colonne."
            )

        # Processa delete pendente PRIMA di qualsiasi widget, così possiamo
        # scrivere liberamente sulle chiavi dei widget (non ancora istanziati).
        _step_fields = (
            "kind", "val", "method", "all_cols", "target_cols", "kind_prev", "per_col"
        )
        _pending_del = st.session_state.pop("_pending_delete_step", None)
        if _pending_del is not None:
            _nd = st.session_state.get("_n_transforms", 0)
            for _j in range(_pending_del, _nd - 1):
                # Shift chiavi fisse
                for _f in _step_fields:
                    _src, _dst = f"tr_{_j+1}_{_f}", f"tr_{_j}_{_f}"
                    if _src in st.session_state:
                        st.session_state[_dst] = st.session_state[_src]
                    elif _dst in st.session_state:
                        del st.session_state[_dst]
                # Shift chiavi override per-colonna (nomi variabili)
                for _ok in list(st.session_state.keys()):
                    if isinstance(_ok, str) and _ok.startswith(f"tr_{_j}_ov_"):
                        del st.session_state[_ok]
                for _ok in list(st.session_state.keys()):
                    if isinstance(_ok, str) and _ok.startswith(f"tr_{_j+1}_ov_"):
                        st.session_state[f"tr_{_j}_ov_{_ok[len(f'tr_{_j+1}_ov_'):]}"] = st.session_state[_ok]
            # Rimuovi ultimo step
            for _f in _step_fields:
                st.session_state.pop(f"tr_{_nd-1}_{_f}", None)
            for _ok in list(st.session_state.keys()):
                if isinstance(_ok, str) and _ok.startswith(f"tr_{_nd-1}_ov_"):
                    del st.session_state[_ok]
            st.session_state["_n_transforms"] = max(0, _nd - 1)

        n_tr = st.session_state.get("_n_transforms", 0)

        if n_tr == 0:
            st.caption("Nessuna trasformazione attiva. Usa '+ Aggiungi' per aggiungere un passo.")

        _tr_kind_opts = list(TRANSFORM_LABELS.keys())
        # Rimuovi eventuali colonne stale (da un CSV precedente non più presenti)
        # e usa tutte le colonne disponibili come fallback se non è ancora stato eseguito il plot.
        _last_y_cols_stored = [
            c for c in st.session_state.get("_last_y_cols", []) if c in cols
        ]
        _last_y_cols = _last_y_cols_stored or [c for c in cols if c not in ("—", x_col)]

        # st.rerun() viene chiamato UNA SOLA VOLTA dopo il loop completo,
        # così tutti i widget committono il proprio stato prima del rerun.
        _needs_rerun = False

        for _i in range(n_tr):
            st.markdown(f"**Passo {_i + 1}**")
            _tc1, _tc2, _tc3, _tc4 = st.columns([2, 1.5, 1.5, 0.4])
            with _tc1:
                _kind = st.selectbox(
                    "Tipo",
                    options=_tr_kind_opts,
                    format_func=lambda k: TRANSFORM_LABELS[k],
                    key=f"tr_{_i}_kind",
                    label_visibility="collapsed",
                )
            with _tc2:
                _val_key = f"tr_{_i}_val"
                _kind_prev_key = f"tr_{_i}_kind_prev"
                _prev_kind = st.session_state.get(_kind_prev_key)
                # Reset val al default quando il tipo cambia
                if _prev_kind is not None and _prev_kind != _kind:
                    _defaults = {"offset": 0.0, "scale": 1.0, "shift_samples": 0,
                                 "shift_time": 0.0, "resample": 100.0}
                    st.session_state[_val_key] = _defaults.get(_kind, 0.0)
                    for _ok in list(st.session_state.keys()):
                        if isinstance(_ok, str) and _ok.startswith(f"tr_{_i}_ov_"):
                            del st.session_state[_ok]
                st.session_state[_kind_prev_key] = _kind
                # Pre-inizializza val se assente
                if _val_key not in st.session_state:
                    _defaults = {"offset": 0.0, "scale": 1.0, "shift_samples": 0,
                                 "shift_time": 0.0, "resample": 100.0}
                    st.session_state[_val_key] = _defaults.get(_kind, 0.0)
                elif _kind == "resample" and float(st.session_state[_val_key]) <= 0:
                    st.session_state[_val_key] = 100.0

                if _kind == "offset":
                    st.number_input("Valore offset", step=0.1,
                                    key=_val_key, label_visibility="collapsed")
                elif _kind == "scale":
                    st.number_input("Fattore moltiplicativo", step=0.1,
                                    key=_val_key, label_visibility="collapsed")
                elif _kind == "shift_samples":
                    st.number_input("Campioni (intero)", step=1,
                                    key=_val_key, label_visibility="collapsed")
                elif _kind == "shift_time":
                    st.number_input("Δt (unità asse X)", step=0.1,
                                    key=_val_key, label_visibility="collapsed")
                elif _kind == "resample":
                    st.number_input("Target fs (Hz)", min_value=0.01, step=1.0,
                                    key=_val_key, label_visibility="collapsed")
                else:
                    st.caption("—")
            with _tc3:
                if _kind == "resample":
                    st.selectbox("Metodo interpolazione",
                                 options=["linear", "cubic", "nearest"],
                                 key=f"tr_{_i}_method",
                                 label_visibility="collapsed")
                else:
                    st.empty()
            with _tc4:
                if st.button("Elimina", key=f"tr_{_i}_del", help=f"Elimina passo {_i + 1}"):
                    st.session_state["_pending_delete_step"] = _i
                    _needs_rerun = True

            # Selettore colonne target
            _apply_all = True
            if _last_y_cols:
                _all_key = f"tr_{_i}_all_cols"
                if _all_key not in st.session_state:
                    st.session_state[_all_key] = True
                _apply_all = st.checkbox("Applica a tutti i segnali", key=_all_key)
                if not _apply_all:
                    _target_key = f"tr_{_i}_target_cols"
                    if _target_key not in st.session_state:
                        st.session_state[_target_key] = list(_last_y_cols)
                    else:
                        _valid = [v for v in st.session_state[_target_key] if v in _last_y_cols]
                        if _valid != st.session_state[_target_key]:
                            st.session_state[_target_key] = _valid
                    st.multiselect(
                        "Segnali target",
                        options=_last_y_cols,
                        key=_target_key,
                        help="Seleziona i segnali a cui applicare questo passo.",
                    )

            # Override per-colonna (solo per offset e scale)
            if _kind in ("offset", "scale") and _last_y_cols:
                _per_col_key = f"tr_{_i}_per_col"
                if _per_col_key not in st.session_state:
                    st.session_state[_per_col_key] = False
                _per_col = st.checkbox(
                    "Valori diversi per colonna",
                    key=_per_col_key,
                    help="Imposta un valore specifico per ogni segnale selezionato.",
                )
                if _per_col:
                    _cols_for_ov = (
                        _last_y_cols if _apply_all
                        else st.session_state.get(f"tr_{_i}_target_cols", _last_y_cols)
                    )
                    _global_val = float(st.session_state.get(_val_key, 0.0))
                    for _oc in _cols_for_ov:
                        _ov_key = f"tr_{_i}_ov_{_oc}"
                        if _ov_key not in st.session_state:
                            st.session_state[_ov_key] = _global_val
                        _oc1, _oc2 = st.columns([2, 3])
                        with _oc1:
                            st.caption(_oc)
                        with _oc2:
                            st.number_input(
                                f"Override {_oc}",
                                step=0.1,
                                key=_ov_key,
                                label_visibility="collapsed",
                            )

        _btn_col, _ = st.columns([1, 5])
        with _btn_col:
            if st.button("+ Aggiungi", disabled=n_tr >= 5, key="tr_btn_add"):
                st.session_state["_n_transforms"] = n_tr + 1
                _needs_rerun = True

        if _needs_rerun:
            st.rerun()

    # ---- PRESET CONFIGURAZIONI (FUORI DAL FORM) ----
    with st.expander("Preset Configurazioni", expanded=False):
        st.markdown("Salva e riutilizza configurazioni filtri/FFT frequenti.")
        _active_preset = st.session_state.get("_active_preset_name")
        if _active_preset:
            st.success(f"Preset attivo: **{_active_preset}**")
        if _IS_STREAMLIT_CLOUD:
            st.info(
                "Su Streamlit Cloud i preset personalizzati vengono persi al riavvio dell'app. "
                "I preset predefiniti vengono ricreati automaticamente ad ogni avvio."
            )

        # Lista preset disponibili
        try:
            available_presets = list_presets()
            preset_names = [p["name"] for p in available_presets]
        except Exception as e:
            st.error(f"Errore caricamento preset: {e}")
            preset_names = []

        # Layout: selectbox + pulsanti
        pcol1, pcol2, pcol3 = st.columns([3, 1, 1])
        with pcol1:
            selected_preset = st.selectbox(
                "Preset disponibili",
                options=["---"] + preset_names,
                key="preset_selector"
            )
        with pcol2:
            st.write("")  # spacer per allineare il pulsante
            load_clicked = st.button("Carica", disabled=selected_preset == "---", key="load_preset_btn")
        with pcol3:
            st.write("")  # spacer per allineare il pulsante
            delete_clicked = st.button("Elimina", disabled=selected_preset == "---", key="delete_preset_btn")

        # Logica Load Preset: applica SUBITO i parametri (rerun) e segna il preset attivo.
        if load_clicked and selected_preset != "---":
            try:
                preset_data = load_preset(selected_preset)
                st.session_state["_loaded_preset"] = preset_data  # consumato a inizio prossimo run
                st.session_state["_active_preset_name"] = selected_preset
                st.session_state["_preset_save_message"] = f"Preset '{selected_preset}' caricato e applicato."
                st.rerun()
            except PresetError as e:
                st.error(f"Errore caricamento: {e}")

        # Logica Delete Preset
        if delete_clicked and selected_preset != "---":
            try:
                delete_preset(selected_preset)
                st.success(f"Preset '{selected_preset}' eliminato.")
                st.rerun()
            except PresetError as e:
                st.error(f"Errore eliminazione: {e}")

        st.markdown("---")
        st.markdown("**Salva configurazione corrente come preset**")

        save_col1, save_col2, save_col3 = st.columns([2, 2, 1])
        with save_col1:
            new_preset_name = st.text_input("Nome preset", placeholder="es. Vibrazione 50Hz", key="new_preset_name_input")
        with save_col2:
            new_preset_desc = st.text_input("Descrizione (opzionale)", placeholder="es. Butterworth LP + FFT", key="new_preset_desc_input")
        with save_col3:
            st.write("")  # spacer per allineare il pulsante
            save_clicked = st.button("Salva", key="save_new_preset_btn")

        if save_clicked:
            if not new_preset_name.strip():
                st.warning("Inserisci un nome per il preset.")
            else:
                st.session_state["_pending_preset_save"] = {
                    "name": new_preset_name.strip(),
                    "description": new_preset_desc.strip()
                }
                st.info("Compila il form sottostante e premi 'Applica / Plot' per completare il salvataggio.")

    if submitted:
        st.session_state["_plots_ready"] = True
        st.session_state["_last_y_cols"] = y_cols  # usato dal pannello Trasformazioni
        # NB: NON azzeriamo il preset attivo dopo il submit: i suoi valori restano
        # applicati e l'indicatore "Preset attivo" continua a mostrarlo.

    if not st.session_state.get("_plots_ready"):
        st.info("Compila il form e premi 'Applica / Plot' per visualizzare grafici e report.")
        return

    if not y_cols:
        st.warning("Seleziona almeno una colonna Y.")
        return


    tab_analisi, tab_report = st.tabs(["Analisi", "Report"])

    with tab_analisi:
        x_name = x_col if (x_col and x_col != "—") else None

        # Interpolazione X: va applicata PRIMA del ffill globale altrimenti ffill
        # riempie i NaN della colonna X e l'interpolazione non trova più valori da interpolare.
        if interpolate_x and x_name and x_name in df.columns:
            x_nan_mask = df[x_name].isna()
            n_nan_x = int(x_nan_mask.sum())
            if n_nan_x > 0:
                # method='index' usa l'indice intero come asse di interpolazione
                # → interpolazione lineare tra valori noti, proporzionale al n. di righe
                df[x_name] = df[x_name].interpolate(method="index")
                n_residui = int(df[x_name].isna().sum())
                msg = f"Interpolazione X '{x_name}': {n_nan_x:,} valori interpolati linearmente"
                if n_residui > 0:
                    msg += f"; {n_residui:,} NaN residui (testa/coda senza riferimento)"
                msg += "."
                st.caption(msg)
            else:
                st.caption(f"Interpolazione X '{x_name}': nessun NaN da interpolare.")

        if fill_missing:
            nan_before = int(df.isna().sum().sum())
            df = df.ffill()
            nan_after = int(df.isna().sum().sum())
            filled_cells = max(nan_before - nan_after, 0)
            st.caption(
                f"Forward fill attivo: {filled_cells:,} celle riempite; NaN residui: {nan_after:,}."
            )
        x_values = None
        if x_name and x_name in df.columns:
            # cerco di mantenere il tipo più utile possibile
            if pd.api.types.is_datetime64_any_dtype(df[x_name]) or pd.api.types.is_timedelta64_dtype(df[x_name]):
                x_values = _to_datetime_flexible(df[x_name])
            else:
                # preferisco numerico se coerente
                xnum = pd.to_numeric(df[x_name], errors="coerce")
                x_values = xnum if xnum.notna().mean() >= 0.8 else _to_datetime_flexible(df[x_name])

        # Risolvo fs UNA SOLA VOLTA
        fs_info = resolve_fs(x_values, manual_fs if manual_fs > 0 else None)
        fs_value = fs_info.value if fs_info.value and fs_info.value > 0 else None
        if fs_value:
            source_labels = {
                "manual": "manuale",
                "datetime": "stimata da timestamp (mediana Δt)",
                "index": "stimata su indice (passi consecutivi)",
            }
            label = source_labels.get(fs_info.source, fs_info.source)
            info_lines = [f"fs [Hz]: **{fs_value:.6g}** ({label})"]
            median_dt = fs_info.details.get("median_dt") if fs_info.details else None
            if median_dt:
                unit_label = "s" if fs_info.unit == "seconds" else "step"
                info_lines.append(f"Δt mediano: {median_dt:.4g} {unit_label}")
            st.info("  \n".join(info_lines))
            # Cache fs info for sidebar live indicator
            st.session_state["_sidebar_fs_info"] = {"value": fs_value, "label": label}
        else:
            st.warning("fs non disponibile: filtri Butterworth e FFT verranno saltati se richiesti.")
            st.session_state.pop("_sidebar_fs_info", None)
        for warn in fs_info.warnings:
            st.warning(warn)

        # Preparo specs
        kind_map = {
            "Media mobile (MA)": "ma",
            "Butterworth LP": "butter_lp",
            "Butterworth HP": "butter_hp",
            "Butterworth BP": "butter_bp",
        }
        fkind = kind_map[f_kind]
        cutoff: Optional[Tuple[Optional[float], Optional[float]]] = None
        lo = _to_float_or_none(f_lo); hi = _to_float_or_none(f_hi)
        if fkind in ("butter_lp", "butter_hp") and lo is not None:
            cutoff = (lo, None)
        elif fkind == "butter_bp" and lo is not None and hi is not None and hi > lo:
            cutoff = (lo, hi)

        fspec = FilterSpec(
            kind=fkind,
            enabled=bool(enable_filter),
            order=int(f_order),
            cutoff=cutoff,
            ma_window=int(ma_win),
        )
        fftspec = FFTSpec(enabled=bool(enable_fft), detrend=bool(detrend), window="hann")

        # Raccoglie le TransformSpec configurate nel pannello Trasformazioni
        _n_tr = int(st.session_state.get("_n_transforms", 0))
        transform_specs: List[TransformSpec] = []
        transform_targets: List[set] = []  # set vuoto = tutti i segnali
        for _ti in range(_n_tr):
            _tkind = st.session_state.get(f"tr_{_ti}_kind", "offset")
            _tval = st.session_state.get(f"tr_{_ti}_val")
            _tval_f = float(_tval) if _tval is not None else 0.0
            _tmethod = st.session_state.get(f"tr_{_ti}_method", "linear")
            _tper_col = bool(st.session_state.get(f"tr_{_ti}_per_col", False))
            _toverrides: Dict[str, float] = {}
            if _tper_col and _tkind in ("offset", "scale"):
                for _tok, _tov in list(st.session_state.items()):
                    if isinstance(_tok, str) and _tok.startswith(f"tr_{_ti}_ov_"):
                        _toverrides[_tok[len(f"tr_{_ti}_ov_"):]] = float(_tov)
            if _tkind == "offset":
                _ts = TransformSpec(kind="offset", constant=_tval_f,
                                    per_column=_tper_col, per_column_overrides=_toverrides)
            elif _tkind == "scale":
                _ts = TransformSpec(kind="scale", constant=_tval_f,
                                    per_column=_tper_col, per_column_overrides=_toverrides)
            elif _tkind == "shift_samples":
                _ts = TransformSpec(kind="shift_samples", shift_samples=int(_tval_f))
            elif _tkind == "shift_time":
                _ts = TransformSpec(kind="shift_time", shift_time=_tval_f)
            elif _tkind == "resample":
                _ts = TransformSpec(kind="resample", target_fs=_tval_f, interp_method=_tmethod)
            else:
                _ts = TransformSpec(kind=_tkind)  # minmax_norm, zscore_norm, derivative, integral
            transform_specs.append(_ts)
            # Colonne target: set vuoto = tutti; set non vuoto = solo queste colonne
            _tapply_all = bool(st.session_state.get(f"tr_{_ti}_all_cols", True))
            if _tapply_all:
                transform_targets.append(set())
            else:
                _tcols = st.session_state.get(f"tr_{_ti}_target_cols", [])
                transform_targets.append(set(_tcols))

        # Salva preset se richiesto
        pending_save = st.session_state.get("_pending_preset_save")
        if submitted and pending_save:
            try:
                save_preset(
                    name=pending_save["name"],
                    description=pending_save["description"],
                    fspec=fspec,
                    fftspec=fftspec,
                    manual_fs=manual_fs if manual_fs > 0 else None
                )
                st.session_state.pop("_pending_preset_save", None)
                st.session_state["_preset_save_message"] = f"Preset '{pending_save['name']}' salvato con successo!"
                st.rerun()
            except PresetError as e:
                st.error(f"Impossibile salvare preset: {e}")
                st.session_state.pop("_pending_preset_save", None)
            except Exception as exc:
                st.error(f"Errore inatteso salvataggio preset: {exc}")
                st.session_state.pop("_pending_preset_save", None)

        if fftspec.enabled:
            if not fs_value:
                st.warning("FFT disabilitata: fs non disponibile.")
                fftspec.enabled = False
            elif not fs_info.is_uniform:
                detail = '; '.join(fs_info.warnings) if fs_info.warnings else 'campionamento irregolare.'
                st.warning(f"FFT disabilitata: {detail}")
                fftspec.enabled = False

        # --- Parse range assi (da sidebar) --- #
        _ax_x_min_s = st.session_state.get("ax_x_min", "")
        _ax_x_max_s = st.session_state.get("ax_x_max", "")
        _ax_y_mode_s = st.session_state.get("ax_y_mode", "Tutti i segnali")

        y_for_range = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in y_cols], axis=0)
        _ax_y_min_g = st.session_state.get("ax_y_min_global", "")
        _ax_y_max_g = st.session_state.get("ax_y_max_global", "")
        yrange = _parse_range_num(_ax_y_min_g, _ax_y_max_g, y_for_range)

        # Per-column Y ranges dict: {yname: Optional[Tuple]}
        _yrange_per_col: dict = {}
        if _ax_y_mode_s == "Per segnale":
            for _ypc in y_cols:
                _ypc_key = _ypc.replace(" ", "_").replace(".", "_")
                _ypc_min = st.session_state.get(f"ax_y_min_{_ypc_key}", "")
                _ypc_max = st.session_state.get(f"ax_y_max_{_ypc_key}", "")
                _ypc_data = pd.to_numeric(df[_ypc], errors="coerce")
                _yrange_per_col[_ypc] = _parse_range_num(_ypc_min, _ypc_max, _ypc_data)

        xrange = None
        if x_name and x_values is not None:
            xrange = _parse_range_x(_ax_x_min_s, _ax_x_max_s, x_values)
        else:
            xmin_idx = _to_float_or_none(_ax_x_min_s)
            xmax_idx = _to_float_or_none(_ax_x_max_s)
            default_min = 0.0
            default_max = float(len(df) - 1) if len(df) > 0 else 0.0
            if xmin_idx is not None or xmax_idx is not None:
                if xmin_idx is None:
                    xmin_idx = default_min
                if xmax_idx is None:
                    xmax_idx = default_max
                if xmin_idx != xmax_idx:
                    xrange = (xmin_idx, xmax_idx)

        quality_mode = st.session_state.get(quality_key, "Alta fedeltà")
        performance_enabled = quality_mode == "Prestazioni"
        downsample_cache: dict[tuple[int, Optional[int]], DownsampleResult] = {}
        downsample_events: List[tuple[str, DownsampleResult]] = []
        recorded_results: set[int] = set()

        # FIX ISSUE #50: Pre-decima DataFrame UNA volta prima del loop
        df_plot = df
        df_downsampled = False
        downsampled_metadata: Optional[DownsampleResult] = None

        if performance_enabled and total_rows > PERFORMANCE_MAX_POINTS:
            # Usa prima colonna Y o X per determinare gli indici di decimazione
            representative_col = y_cols[0] if y_cols else None
            ds_result: Optional[DownsampleResult] = None
            if representative_col:
                y_repr = pd.to_numeric(df[representative_col], errors="coerce")
                x_repr = x_values if x_values is not None else None

                if y_repr.dropna().empty:
                    st.warning("Downsampling saltato: la serie rappresentativa non ha valori numerici.")
                else:
                    # Calcola indici di downsampling
                    ds_result = downsample_series(
                        y_repr,
                        x_repr,
                        max_points=PERFORMANCE_MAX_POINTS,
                        method=PERFORMANCE_METHOD,
                    )

            # Pre-decima DF intero usando gli indici
            if ds_result and ds_result.sampled_count < total_rows:
                df_plot = df.iloc[ds_result.indices].copy()
                df_downsampled = True
                downsampled_metadata = ds_result
                st.caption(
                    f"Pre-decimazione: {total_rows:,} → {len(df_plot):,} righe "
                    f"({ds_result.reduction_ratio:.1f}×, {ds_result.method.upper()})"
                )

        # FIX ISSUE #52: Pre-converti X UNA volta sola (per plot, evita ri-conversioni per ogni Y)
        # Posizionato DOPO df_plot per accedere sia a df che df_plot
        x_parsed_plot = _parse_x_column_once(df_plot, x_name)  # Per plot mode (con decimazione)
        x_parsed_orig = _parse_x_column_once(df, x_name)       # Per FFT (dati originali)

        def _get_series_sources(
            y_col: str,
        ) -> tuple[pd.Series, Optional[pd.Series], pd.Series, Optional[pd.Series]]:
            """Ritorna serie (plot) e serie originale per la colonna richiesta."""
            series_plot, x_plot = _make_time_series(df_plot, x_name, y_col, x_parsed=x_parsed_plot)
            if df_downsampled:
                series_full, x_full = _make_time_series(df, x_name, y_col, x_parsed=x_parsed_orig)
            else:
                series_full, x_full = series_plot, x_plot
            return series_plot, x_plot, series_full, x_full

        def _legend_label(base: str, meta: Optional[DownsampleResult]) -> str:
            if meta is None or meta.original_count <= meta.sampled_count:
                return base
            return f"{base} [down {meta.original_count:,}->{meta.sampled_count:,}]"

        def _prepare_plot_series(
            label: str,
            y_data: pd.Series,
            x_data: Optional[pd.Series],
            *,
            reuse_index: Optional[pd.Index] = None,
        ) -> tuple[Optional[pd.Series], pd.Series, Optional[DownsampleResult]]:
            if reuse_index is not None:
                y_sel = y_data.loc[reuse_index]
                x_sel = x_data.loc[reuse_index] if x_data is not None else None
                return x_sel, y_sel, None

            # Drop coppie X/Y non valide per evitare linee invisibili
            y_data, x_data = _mask_xy(y_data, x_data)
            if y_data.empty:
                return x_data, y_data, None

            # FIX ISSUE #50: Se DF già pre-decimato, salta downsampling per-series
            if df_downsampled:
                return x_data, y_data, downsampled_metadata

            # Fallback: downsampling per-series (legacy, solo se DF NON pre-decimato)
            if not performance_enabled or len(y_data) <= PERFORMANCE_MAX_POINTS:
                return x_data, y_data, None
            cache_key = (id(y_data), id(x_data) if x_data is not None else None)
            result = downsample_cache.get(cache_key)
            if result is None:
                result = downsample_series(
                    y_data,
                    x_data,
                    max_points=PERFORMANCE_MAX_POINTS,
                    method=PERFORMANCE_METHOD,
                )
                downsample_cache[cache_key] = result
            if result.original_count > result.sampled_count and id(result) not in recorded_results:
                downsample_events.append((label, result))
                recorded_results.add(id(result))
            return result.x, result.y, result

        # ========================= PLOT ========================= #
        if mode == "Sovrapposto":
            # ----- UNICA FIGURA CON TUTTE LE SERIE ----- #
            combined = go.Figure()
            x_label = x_name if x_name else "Index"
            _tr_series_for_fft: Dict[str, Tuple[pd.Series, Optional[pd.Series]]] = {}

            for yname in y_cols:
                series_plot, x_plot, series_full, x_full = _get_series_sources(yname)
                series_orig, x_orig = series_full, x_full  # per overlay "originale" (pre-trasformazione)

                # Applica pipeline trasformazioni (prima del filtro) — solo spec applicabili a questa colonna
                _col_specs = [s for s, t in zip(transform_specs, transform_targets) if not t or yname in t]
                if _col_specs:
                    try:
                        _tr_y, _tr_x, _, _tr_changed = _apply_transform_pipeline_cached(
                            series_full, x_full, _col_specs, fs_info, file_sig, yname, fill_stamp
                        )
                        series_full = _tr_y
                        x_full = _tr_x
                        if _tr_changed:
                            series_plot = _tr_y
                            x_plot = _tr_x
                        else:
                            series_plot = _tr_y.reindex(series_plot.index)
                            x_plot = _tr_x.reindex(x_plot.index) if _tr_x is not None and x_plot is not None else _tr_x
                    except ValueError as _e:
                        st.warning(f"Trasformazione non applicata a '{yname}': {_e}")
                _tr_series_for_fft[yname] = (series_full, x_full)

                series = series_plot
                x_ser = x_plot
                if series.dropna().empty:
                    st.info(f"'{yname}': nessun dato numerico valido.")
                    continue

                # Filtro (se attivo)
                y_filt_full: Optional[pd.Series] = None
                y_filt_plot: Optional[pd.Series] = None
                ok, msg = validate_filter_spec(fspec, fs_value)
                if fspec.enabled and not ok:
                    st.warning(f"Filtro non applicato a {yname}: {msg}")
                    y_plot = series
                else:
                    if fspec.enabled:
                        y_filt_full = _apply_filter_cached(
                            series_full,
                            x_full,
                            fspec,
                            fs_value,
                            fs_info.source,
                            file_sig,
                            yname,
                            fill_stamp,
                        )
                        if y_filt_full is None:
                            st.warning(f"Filtro non applicato a {yname}: errore nel calcolo.")
                            y_plot = series
                        else:
                            y_filt_plot = y_filt_full.reindex(series.index)
                            y_plot = y_filt_plot
                    else:
                        y_plot = series

                name_main = yname + (" (filtrato)" if (fspec.enabled and not overlay_orig) else "")
                x_main, y_main, main_meta = _prepare_plot_series(name_main, y_plot, x_ser)
                if y_main.empty:
                    st.info(f"'{yname}': nessun dato valido dopo la rimozione dei NaN.")
                    continue

                # Originale tratteggiato se richiesto (usa serie pre-trasformazione)
                if overlay_orig and fspec.enabled and y_filt_plot is not None:
                    overlay_label = f"{yname} (originale)"
                    reuse_idx = y_main.index if main_meta and main_meta.original_count > main_meta.sampled_count else None
                    x_overlay_src = x_orig if x_orig is not None else x_ser
                    x_overlay, y_overlay, overlay_meta = _prepare_plot_series(
                        overlay_label,
                        series_orig,
                        x_overlay_src,
                        reuse_index=reuse_idx,
                    )
                    if not y_overlay.empty:
                        combined.add_trace(
                            go.Scatter(
                                x=(x_overlay if x_overlay is not None else None),
                                y=y_overlay,
                                mode="lines",
                            name=_legend_label(overlay_label, overlay_meta or main_meta),
                            line=dict(width=1, dash="dot"),
                        )
                    )

                # Traccia principale (filtrato o originale)
                if not y_main.empty:
                    combined.add_trace(
                        go.Scatter(
                            x=(x_main if x_main is not None else None),
                            y=y_main,
                            mode="lines",
                            name=_legend_label(name_main, main_meta),
                        )
                    )
                if overlay_orig and fspec.enabled and y_filt_plot is not None:
                    combined.data = combined.data[::-1]

            combined.update_layout(
                title="Confronto sovrapposto",
                xaxis_title=x_label,
                yaxis_title="Valore",
                template="plotly_dark",
                paper_bgcolor="#020617",
                plot_bgcolor="#0a1628",
                legend_title="Serie",
                margin=dict(l=50, r=30, t=60, b=50),
            )
            if yrange:
                combined.update_yaxes(range=yrange)
            if xrange:
                combined.update_xaxes(range=xrange)

            _plotly_chart(st, combined)

            # FFT: una per serie, sotto
            if fftspec.enabled:
                for yname in y_cols:
                    # FIX ISSUE #50: FFT usa dati ORIGINALI (non decimati), non df_plot
                    # FIX ISSUE #52: Passa X pre-parsato originale
                    # Se trasformazioni attive, usa la serie trasformata già calcolata nel loop sopra
                    _tr_full = _tr_series_for_fft.get(yname)
                    if _tr_full is not None:
                        series, x_ser = _tr_full
                    else:
                        series, x_ser = _make_time_series(df, x_name, yname, x_parsed=x_parsed_orig)
                    if series.dropna().empty:
                        continue
                    y_filt = None
                    if fspec.enabled:
                        y_filt = _apply_filter_cached(
                            series,
                            x_ser,
                            fspec,
                            fs_value,
                            fs_info.source,
                            file_sig,
                            yname,
                            fill_stamp,
                        )
                    y_fft = y_filt if (fspec.enabled and y_filt is not None and fft_use == "Filtrato (se attivo)") else series
                    if not fs_value or fs_value <= 0:
                        st.warning(f"FFT non calcolata per {yname}: fs non disponibile.")
                    elif not fs_info.is_uniform:
                        detail = "; ".join(fs_info.warnings) if fs_info.warnings else "campionamento irregolare."
                        st.warning(f"FFT non calcolata per {yname}: {detail}")
                    else:
                        is_filt = fspec.enabled and y_filt is not None and fft_use == "Filtrato (se attivo)"
                        freqs, amp = _compute_fft_cached(
                            y_fft,
                            fs_value,
                            fs_info.source,
                            fftspec,
                            file_sig,
                            yname,
                            is_filt,
                            fill_stamp,
                            fspec,
                        )
                        if freqs.size == 0:
                            st.info(f"FFT non calcolabile per {yname} (serie troppo corta o parametri non validi).")
                        else:
                            _plotly_chart(
                                st,
                                _plot_fft(freqs, amp, title=f"FFT — {yname}"),
                            )

        elif mode == "Separati":
            # ----- UNA TAB PER SERIE ----- #
            tabs = st.tabs(y_cols)
            for idx, yname in enumerate(y_cols):
                series_plot, x_plot, series_full, x_full = _get_series_sources(yname)
                series_orig, x_orig = series_full, x_full  # per overlay pre-trasformazione

                # Applica pipeline trasformazioni (prima del filtro) — solo spec applicabili a questa colonna
                _col_specs = [s for s, t in zip(transform_specs, transform_targets) if not t or yname in t]
                if _col_specs:
                    try:
                        _tr_y, _tr_x, _, _tr_changed = _apply_transform_pipeline_cached(
                            series_full, x_full, _col_specs, fs_info, file_sig, yname, fill_stamp
                        )
                        series_full = _tr_y
                        x_full = _tr_x
                        if _tr_changed:
                            series_plot = _tr_y
                            x_plot = _tr_x
                        else:
                            series_plot = _tr_y.reindex(series_plot.index)
                            x_plot = _tr_x.reindex(x_plot.index) if _tr_x is not None and x_plot is not None else _tr_x
                    except ValueError as _e:
                        tabs[idx].warning(f"Trasformazione non applicata a '{yname}': {_e}")

                series = series_plot
                x_ser = x_plot
                host = tabs[idx]

                if series.dropna().empty:
                    host.info(f"'{yname}': nessun dato numerico valido.")
                    continue

                # Filtro
                y_filt_full: Optional[pd.Series] = None
                y_filt_plot: Optional[pd.Series] = None
                ok, msg = validate_filter_spec(fspec, fs_value)
                if fspec.enabled and not ok:
                    host.warning(f"Filtro non applicato a {yname}: {msg}")
                    y_plot = series
                elif fspec.enabled:
                    y_filt_full = _apply_filter_cached(
                        series_full,
                        x_full,
                        fspec,
                        fs_value,
                        fs_info.source,
                        file_sig,
                        yname,
                        fill_stamp,
                    )
                    if y_filt_full is None:
                        host.warning(f"Filtro non applicato a {yname}: errore nel calcolo.")
                        y_plot = series
                    else:
                        y_filt_plot = y_filt_full.reindex(series.index)
                        y_plot = y_filt_plot
                else:
                    y_plot = series

                display_name = yname + (" (filtrato)" if (fspec.enabled and not overlay_orig) else "")
                x_plot, y_plot_ds, main_meta = _prepare_plot_series(display_name, y_plot, x_ser)
                if y_plot_ds.empty:
                    host.info(f"'{yname}': nessun dato valido dopo la rimozione dei NaN.")
                    continue
                fig = _plot_xy(x_plot, y_plot_ds, name=display_name)
                if fig.data:
                    fig.data[0].name = _legend_label(display_name, main_meta)
                _this_yrange = (_yrange_per_col.get(yname) if _ax_y_mode_s == "Per segnale" else None) or yrange
                if _this_yrange:
                    fig.update_yaxes(range=_this_yrange)
                if xrange:
                    fig.update_xaxes(range=xrange)
                if overlay_orig and fspec.enabled and y_filt_plot is not None:
                    overlay_label = f"{yname} (originale)"
                    reuse_idx = y_plot_ds.index if main_meta and main_meta.original_count > main_meta.sampled_count else None
                    x_overlay_src = x_orig if x_orig is not None else x_ser
                    x_overlay, y_overlay, overlay_meta = _prepare_plot_series(
                        overlay_label,
                        series_orig,
                        x_overlay_src,
                        reuse_index=reuse_idx,
                    )
                    if not y_overlay.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=x_overlay if x_overlay is not None else None,
                                y=y_overlay,
                                mode="lines",
                                name=_legend_label(overlay_label, overlay_meta or main_meta),
                                line=dict(width=1, dash="dot"),
                            )
                        )
                        fig.data = fig.data[::-1]
                _plotly_chart(host, fig)

                # FFT per singola serie
                if fftspec.enabled:
                    if fspec.enabled and y_filt_full is not None and fft_use == "Filtrato (se attivo)":
                        y_fft = y_filt_full
                    else:
                        y_fft = series_full
                    if not fs_value or fs_value <= 0:
                        host.warning(f"FFT non calcolata per {yname}: fs non disponibile.")
                    elif not fs_info.is_uniform:
                        detail = "; ".join(fs_info.warnings) if fs_info.warnings else "campionamento irregolare."
                        host.warning(f"FFT non calcolata per {yname}: {detail}")
                    else:
                        is_filt = fspec.enabled and y_filt_full is not None and fft_use == "Filtrato (se attivo)"
                        freqs, amp = _compute_fft_cached(
                            y_fft,
                            fs_value,
                            fs_info.source,
                            fftspec,
                            file_sig,
                            yname,
                            is_filt,
                            fill_stamp,
                            fspec,
                        )
                        if freqs.size == 0:
                            host.info(f"FFT non calcolabile per {yname} (serie troppo corta o parametri non validi).")
                        else:
                            _plotly_chart(
                                host,
                                _plot_fft(freqs, amp, title=f"FFT — {yname}"),
                            )

        else:
            # ----- CASCATA: grafici uno sotto l'altro ----- #
            for yname in y_cols:
                series_plot, x_plot, series_full, x_full = _get_series_sources(yname)
                series_orig, x_orig = series_full, x_full  # per overlay pre-trasformazione

                # Applica pipeline trasformazioni (prima del filtro) — solo spec applicabili a questa colonna
                _col_specs = [s for s, t in zip(transform_specs, transform_targets) if not t or yname in t]
                if _col_specs:
                    try:
                        _tr_y, _tr_x, _, _tr_changed = _apply_transform_pipeline_cached(
                            series_full, x_full, _col_specs, fs_info, file_sig, yname, fill_stamp
                        )
                        series_full = _tr_y
                        x_full = _tr_x
                        if _tr_changed:
                            series_plot = _tr_y
                            x_plot = _tr_x
                        else:
                            series_plot = _tr_y.reindex(series_plot.index)
                            x_plot = _tr_x.reindex(x_plot.index) if _tr_x is not None and x_plot is not None else _tr_x
                    except ValueError as _e:
                        st.warning(f"Trasformazione non applicata a '{yname}': {_e}")

                series = series_plot
                x_ser = x_plot

                if series.dropna().empty:
                    st.info(f"'{yname}': nessun dato numerico valido.")
                    continue

                # Filtro
                y_filt_full: Optional[pd.Series] = None
                y_filt_plot: Optional[pd.Series] = None
                ok, msg = validate_filter_spec(fspec, fs_value)
                if fspec.enabled and not ok:
                    st.warning(f"Filtro non applicato a {yname}: {msg}")
                    y_plot = series
                else:
                    if fspec.enabled:
                        y_filt_full = _apply_filter_cached(
                            series_full,
                            x_full,
                            fspec,
                            fs_value,
                            fs_info.source,
                            file_sig,
                            yname,
                            fill_stamp,
                        )
                        if y_filt_full is None:
                            st.warning(f"Filtro non applicato a {yname}: errore nel calcolo.")
                            y_plot = series
                        else:
                            y_filt_plot = y_filt_full.reindex(series.index)
                            y_plot = y_filt_plot
                    else:
                        y_plot = series

                display_name = yname + (" (filtrato)" if (fspec.enabled and not overlay_orig) else "")
                x_plot, y_plot_ds, main_meta = _prepare_plot_series(display_name, y_plot, x_ser)
                fig = _plot_xy(x_plot, y_plot_ds, name=display_name)
                if fig.data:
                    fig.data[0].name = _legend_label(display_name, main_meta)
                _this_yrange = (_yrange_per_col.get(yname) if _ax_y_mode_s == "Per segnale" else None) or yrange
                if _this_yrange:
                    fig.update_yaxes(range=_this_yrange)
                if xrange:
                    fig.update_xaxes(range=xrange)
                if overlay_orig and fspec.enabled and y_filt_plot is not None:
                    overlay_label = f"{yname} (originale)"
                    reuse_idx = y_plot_ds.index if main_meta and main_meta.original_count > main_meta.sampled_count else None
                    x_overlay_src = x_orig if x_orig is not None else x_ser
                    x_overlay, y_overlay, overlay_meta = _prepare_plot_series(
                        overlay_label,
                        series_orig,
                        x_overlay_src,
                        reuse_index=reuse_idx,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_overlay if x_overlay is not None else None,
                            y=y_overlay,
                            mode="lines",
                            name=_legend_label(overlay_label, overlay_meta or main_meta),
                            line=dict(width=1, dash="dot"),
                        )
                    )
                    fig.data = fig.data[::-1]
                _plotly_chart(st, fig)

                # FFT sotto ogni grafico (se attiva)
                if fftspec.enabled:
                    if fspec.enabled and y_filt_full is not None and fft_use == "Filtrato (se attivo)":
                        y_fft = y_filt_full
                    else:
                        y_fft = series_full
                    if not fs_value or fs_value <= 0:
                        st.warning(f"FFT non calcolata per {yname}: fs non disponibile.")
                    elif not fs_info.is_uniform:
                        detail = "; ".join(fs_info.warnings) if fs_info.warnings else "campionamento irregolare."
                        st.warning(f"FFT non calcolata per {yname}: {detail}")
                    else:
                        is_filt = fspec.enabled and y_filt_full is not None and fft_use == "Filtrato (se attivo)"
                        freqs, amp = _compute_fft_cached(
                            y_fft,
                            fs_value,
                            fs_info.source,
                            fftspec,
                            file_sig,
                            yname,
                            is_filt,
                            fill_stamp,
                            fspec,
                        )
                        if freqs.size == 0:
                            st.info(f"FFT non calcolabile per {yname} (serie troppo corta o parametri non validi).")
                        else:
                            _plotly_chart(
                                st,
                                _plot_fft(freqs, amp, title=f"FFT — {yname}"),
                            )

        if performance_enabled:
            summaries: List[str] = []
            seen_pairs: set[tuple[str, int]] = set()
            for label, res in downsample_events:
                key = (label, res.sampled_count)
                if res.original_count <= res.sampled_count or key in seen_pairs:
                    continue
                seen_pairs.add(key)
                summaries.append(
                    f"{label}: {res.original_count:,}->{res.sampled_count:,} ({res.reduction_ratio:.1f}x)"
                )
            if summaries:
                st.caption("Prestazioni attive (LTTB): " + " · ".join(summaries))

        # FIX ISSUE #55: Log cache hit rate dopo plot
        filter_total = CACHE_STATS["filter_hits"] + CACHE_STATS["filter_misses"]
        fft_total = CACHE_STATS["fft_hits"] + CACHE_STATS["fft_misses"]
        if filter_total > 0 or fft_total > 0:
            filter_hit_rate = (CACHE_STATS["filter_hits"] / filter_total * 100) if filter_total > 0 else 0
            fft_hit_rate = (CACHE_STATS["fft_hits"] / fft_total * 100) if fft_total > 0 else 0
            logger.info(
                "Cache hit rate",
                extra={
                    "filter_hits": CACHE_STATS["filter_hits"],
                    "filter_misses": CACHE_STATS["filter_misses"],
                    "filter_hit_rate": f"{filter_hit_rate:.1f}%",
                    "fft_hits": CACHE_STATS["fft_hits"],
                    "fft_misses": CACHE_STATS["fft_misses"],
                    "fft_hit_rate": f"{fft_hit_rate:.1f}%",
                    "session_id": st.session_state.get("_dataset_id", "")[:8]
                }
            )


    with tab_report:
        # ---- Report ----
        st.divider()
        st.subheader("Report statistici")

        # Costruisce il DataFrame per il report: originale o con trasformazioni applicate.
        # Le colonne per cui la trasformazione cambia la lunghezza (derivata, resample)
        # vengono mantenute originali con un avviso.
        _df_report = df
        if transform_specs:
            _rpt_use_tr = st.checkbox(
                "Usa dati trasformati nel report",
                value=True,
                help="Applica le trasformazioni configurate prima di calcolare statistiche e grafici.",
            )
            if _rpt_use_tr:
                _df_report = df.copy()
                _rpt_skipped: List[str] = []
                for _ryn in [c for c in _df_report.columns if c != x_name]:
                    _rspecs = [
                        s for s, t in zip(transform_specs, transform_targets)
                        if not t or _ryn in t
                    ]
                    if not _rspecs:
                        continue
                    _rx_r = _df_report[x_name] if x_name and x_name in _df_report.columns else None
                    try:
                        _rtr_y, _, _, _rtr_chg = _apply_transform_pipeline_cached(
                            _df_report[_ryn], _rx_r, _rspecs, fs_info, file_sig, _ryn, fill_stamp
                        )
                        if _rtr_chg:
                            _rpt_skipped.append(_ryn)
                        else:
                            _df_report[_ryn] = _rtr_y
                    except ValueError:
                        _rpt_skipped.append(_ryn)
                if _rpt_skipped:
                    st.info(
                        "Le seguenti colonne usano i dati originali nel report "
                        "(la trasformazione ne modifica la lunghezza del segnale): "
                        + ", ".join(_rpt_skipped)
                    )

        # Slice X: filtra _df_report allo stesso intervallo mostrato nel grafico.
        # Usa x_values (già convertito a numerico/datetime) per costruire la maschera,
        # così si evita il confronto dtype=str vs float sulla colonna originale.
        if xrange is not None:
            try:
                if x_values is not None:
                    _xmask = (x_values >= xrange[0]) & (x_values <= xrange[1])
                elif x_name and x_name in _df_report.columns:
                    _xf = pd.to_numeric(_df_report[x_name], errors="coerce")
                    _xmask = (_xf >= xrange[0]) & (_xf <= xrange[1])
                else:
                    _idx_s = _df_report.index.to_series()
                    _xmask = (_idx_s >= xrange[0]) & (_idx_s <= xrange[1])
                _df_report = _df_report.loc[_xmask.values].reset_index(drop=True)
                st.caption(
                    f"Report calcolato sull'intervallo X [{xrange[0]}, {xrange[1]}]"
                    f" — {len(_df_report)} righe."
                )
            except Exception:
                pass

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            fmt = st.selectbox(
                "Formato",
                ["csv", "csv+md", "csv+html", "csv+md+html"],
                index=0,
                key="report_format",
            )
            base_name = st.text_input(
                "Nome base report (opzionale)",
                placeholder="es. report_misura_001",
                key="report_base_name",
            )
        with col_r2:
            st.write("")  # spacer per allineare il pulsante
            if st.button("Genera report"):
                try:
                    manager = ReportManager()
                    out_paths = manager.generate_report(
                        _df_report, x_name, y_cols, formats=fmt, base_name=base_name or None
                    )
                    mime_map = {
                        "csv": "text/csv",
                        "md": "text/markdown",
                        "html": "text/html",
                    }
                    downloads = {}
                    for fmt_name, path in out_paths.items():
                        if path and path.exists():
                            downloads[fmt_name] = {
                                "path": path,
                                "bytes": path.read_bytes(),
                                "mime": mime_map.get(fmt_name, "application/octet-stream"),
                            }
                    st.session_state["_generated_report"] = {
                        "outputs": out_paths,
                        "downloads": downloads,
                    }
                    st.session_state.pop("_generated_report_error", None)
                except Exception as e:
                    # FIX ISSUE #54: Messaggio generico utente, log tecnico con traceback
                    st.session_state.pop("_generated_report", None)
                    st.session_state["_generated_report_error"] = "Errore nella generazione del report."
                    logger.error(
                        "Report generation failed",
                        exc_info=True,
                        extra={
                            "formats": fmt,
                            "columns": len(df.columns),
                            "rows": len(df),
                            "session_id": st.session_state.get("_dataset_id", "")[:8]
                        }
                    )

        report_error = st.session_state.get("_generated_report_error")
        if report_error:
            st.error(f"Generazione report fallita: {report_error}")
        generated_report = st.session_state.get("_generated_report")
        if generated_report:
            st.success("Report generato.")
            outputs = generated_report.get("outputs", {})
            st.json({k: str(v) if v else None for k, v in outputs.items()})
            downloads = generated_report.get("downloads", {})
            for fmt_name, info in downloads.items():
                st.download_button(
                    f"Scarica {fmt_name.upper()}",
                    data=info["bytes"],
                    file_name=info["path"].name,
                    mime=info["mime"],
                    key=f"download_report_{fmt_name}",
                )

        st.divider()
        st.subheader("Report visivo dei grafici")
        st.caption("Scegli fino a 4 serie per creare un'immagine o un PDF con i grafici in cascata.")

        visual_default = y_cols[: min(4, len(y_cols))] if y_cols else cols[: min(4, len(cols))]
        visual_raw_selection = st.multiselect(
            "Serie da includere (max 4)",
            options=cols,
            default=visual_default,
            help="Le serie devono essere numeriche; eventuali NaN verranno ignorati.",
        )

        if len(visual_raw_selection) > 4:
            st.warning("Puoi selezionare al massimo 4 serie: verranno considerate solo le prime quattro.")

        visual_selection = visual_raw_selection[:4]

        default_x_label = x_name if x_name else "Index"
        prev_default = st.session_state.get("_visual_report_last_default_x_label")
        _sync_visual_spec_state(visual_selection, default_x_label)
        if prev_default is not None and prev_default != default_x_label:
            for col in visual_selection:
                key = _visual_spec_key("xlabel", col)
                if st.session_state.get(key) == prev_default:
                    st.session_state[key] = default_x_label
        st.session_state["_visual_report_last_default_x_label"] = default_x_label

        visual_specs: List[VisualPlotSpec] = []
        for idx, yname in enumerate(visual_selection):
            title_key = _visual_spec_key("title", yname)
            xlabel_key = _visual_spec_key("xlabel", yname)
            ylabel_key = _visual_spec_key("ylabel", yname)
            with st.expander(
                f"Grafico {idx + 1} — {yname}",
                expanded=False,
            ):
                plot_title = st.text_input("Titolo grafico", key=title_key)
                x_label = st.text_input("Titolo asse X", key=xlabel_key)
                y_label = st.text_input("Titolo asse Y", key=ylabel_key)
            visual_specs.append(
                VisualPlotSpec(
                    y_column=yname,
                    title=plot_title or None,
                    x_label=x_label or None,
                    y_label=y_label or None,
                )
            )

        col_vis1, col_vis2 = st.columns([2, 1])
        with col_vis1:
            visual_title = st.text_input("Titolo report visivo", key="vis_report_main_title")
            visual_base = st.text_input("Nome file (opzionale)", placeholder="es. report_visivo", key="vis_report_base")
        with col_vis2:
            visual_format = st.radio(
                "Formato",
                ["html"],
                #["png", "pdf", "html"], <-- Per eliminare "pdf" e "png" in cloud (Plotly non supporta più l'export in questi formati)
                horizontal=True,
                key="vis_report_format",
            )
            visual_show_legend = st.checkbox("Mostra legenda", value=False, key="vis_report_legend")

        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
        with btn_col2:
            generate_visual = _button("Genera report visivo")

        if generate_visual:
            if not visual_specs:
                st.warning("Seleziona almeno una serie per il report visivo.")
            else:
                try:
                    with st.spinner("Generazione report visivo..."):
                        manager = VisualReportManager()
                        result = manager.generate_report(
                            df=_df_report,
                            specs=visual_specs,
                            x_column=x_name,
                            title=visual_title or "", #modifica per eliminare "udefined" su report html quando titolo vuoto
                            base_name=visual_base or None,
                            file_format=visual_format,
                            show_legend=visual_show_legend,
                            x_range=xrange,
                            y_range=yrange,
                        )
                    st.session_state["_generated_visual_report"] = result
                    st.session_state.pop("_generated_visual_report_error", None)
                except Exception as e:
                    # FIX ISSUE #54: Messaggio generico utente, log tecnico con traceback
                    st.session_state.pop("_generated_visual_report", None)
                    st.session_state["_generated_visual_report_error"] = "Errore nella generazione del report visivo."
                    logger.error(
                        "Visual report generation failed",
                        exc_info=True,
                        extra={
                            "format": visual_format,
                            "num_series": len(visual_specs),
                            "columns": [spec["column"] for spec in visual_specs],
                            "session_id": st.session_state.get("_dataset_id", "")[:8]
                        }
                    )

        visual_error = st.session_state.get("_generated_visual_report_error")
        if visual_error:
            st.error(f"Generazione report visivo fallita: {visual_error}")
        visual_result = st.session_state.get("_generated_visual_report")
        if visual_result:
            actual_format = visual_result["format"]
            requested_format = visual_result.get("requested_format", actual_format)
            fallback_reason = visual_result.get("fallback_reason")

            st.success(f"Report visivo salvato in {visual_result['path']}")
            if requested_format != actual_format:
                warning_msg = (
                    f"Il formato {requested_format.upper()} non è disponibile in questo ambiente. "
                    f"Il report è stato esportato come {actual_format.upper()}."
                )
                st.warning(warning_msg)
                if fallback_reason:
                    st.caption(f"Dettagli: {fallback_reason}")

            if actual_format == "pdf":
                mime = "application/pdf"
            elif actual_format == "html":
                mime = "text/html"
            else:
                mime = "image/png"
            st.download_button(
                "Scarica report",
                data=visual_result["bytes"],
                file_name=visual_result["path"].name,
                mime=mime,
                key="download_visual_report",
            )
            if actual_format == "png":
                _image(visual_result["bytes"], caption="Anteprima report visivo")
            elif actual_format == "html":
                st.info("Anteprima interattiva generata in formato HTML.")
                _plotly_chart(st, visual_result["figure"], key="visual_report_preview")

        st.divider()
        with st.expander("Info rilevate (clicca per espandere)", expanded=False):
            st.json(meta)

        # Footer: mostra loader type
        loader_desc = "Ottimizzato (chunked)" if LOADER_TYPE == "optimized" else "Standard"
        st.caption(f"Loader: {loader_desc}")


def main():
    """Entry point: header sempre visibile + due schede (Analisi / Impostazioni)."""
    _ensure_session_id()
    st.set_page_config(page_title="Analizzatore CSV - Web", layout="wide")

    render_header()

    tab_app, tab_settings = st.tabs(["Analisi", "Impostazioni"])
    with tab_settings:
        _render_settings_panel()
    with tab_app:
        _render_main_app()


if __name__ == "__main__":
    main()
