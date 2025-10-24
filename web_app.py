from __future__ import annotations

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
import streamlit as st

from core.analyzer import analyze_csv
from core.loader import load_csv, optimize_dtypes
from core.csv_cleaner import CleaningReport
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
    "x_min_txt",
    "x_max_txt",
    "y_min_txt",
    "y_max_txt",
    # Advanced
    "manual_fs",
    "enable_filter",
    "f_kind",
    "ma_win",
    "f_order",
    "f_lo",
    "f_hi",
    "overlay_orig",
    # Note: enable_fft, fft_use, detrend removed: widgets have no key, reset automatically with form nonce
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


@lru_cache(maxsize=1)
def _load_limits_config() -> Dict[str, float]:
    """Legge i limiti di caricamento da config.json con fallback sicuri."""
    import json

    merged: Dict[str, float] = dict(LIMIT_DEFAULTS)
    config_path = Path("config.json")
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                limits = data.get("limits") or {}
                for key, default in LIMIT_DEFAULTS.items():
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


def _parsing_worker(result_queue: "mp.Queue", file_path: str, apply_cleaning: bool) -> None:
    """Worker che analizza e carica il CSV, inviando il risultato tramite coda."""
    try:
        meta = analyze_csv(file_path)
        df, cleaning_report = load_csv(
            file_path,
            encoding=meta.get("encoding"),
            delimiter=meta.get("delimiter"),
            header=meta.get("header"),
            apply_cleaning=apply_cleaning,
            return_details=True,
        )
        result_queue.put(("ok", meta, df, cleaning_report))
    except Exception as exc:  # pragma: no cover - il messaggio viene gestito dal caller
        result_queue.put(("error", exc))


def _parse_csv_with_timeout(file_bytes: bytes, apply_cleaning: bool, timeout_s: float) -> Tuple[pd.DataFrame, CleaningReport, Dict[str, Any]]:
    """Esegue analyze + load in un processo separato e applica un timeout."""
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
SAMPLE_CSV_PATH = Path("assets/sample_timeseries.csv")

# Cache limits
MAX_FILTER_CACHE_SIZE = 32
MAX_FFT_CACHE_SIZE = 16


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


def _get_filter_cache_key(
    column: str, file_sig: FileSignature, fspec: FilterSpec, fs: Optional[float], fs_source: Optional[str]
) -> Tuple:
    """Generate hashable cache key for filter results."""
    from dataclasses import astuple
    # Include fs and fs_source in key to invalidate when sampling frequency or source changes
    return (column, file_sig, astuple(fspec), fs, fs_source)


def _get_fft_cache_key(
    column: str, file_sig: FileSignature, is_filtered: bool, fftspec: FFTSpec, fs: float, fs_source: Optional[str]
) -> Tuple:
    """Generate hashable cache key for FFT results."""
    from dataclasses import astuple
    # Include fs_source to invalidate when fs changes from estimated to manual
    return (column, file_sig, is_filtered, astuple(fftspec), fs, fs_source)


def _get_cached_filter(key: Tuple) -> Optional[pd.Series]:
    """Retrieve cached filter result."""
    return st.session_state.get("_filter_cache", {}).get(key)


def _cache_filter_result(key: Tuple, result: pd.Series) -> None:
    """Store filter result with LRU eviction."""
    cache = st.session_state.setdefault("_filter_cache", {})
    if len(cache) >= MAX_FILTER_CACHE_SIZE:
        # Simple LRU: remove oldest (first) entry
        oldest_key = next(iter(cache))
        cache.pop(oldest_key)
    cache[key] = result.copy()  # Store copy to avoid reference issues


def _get_cached_fft(key: Tuple) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Retrieve cached FFT result."""
    return st.session_state.get("_fft_cache", {}).get(key)


def _cache_fft_result(key: Tuple, freqs: np.ndarray, amp: np.ndarray) -> None:
    """Store FFT result with LRU eviction."""
    cache = st.session_state.setdefault("_fft_cache", {})
    if len(cache) >= MAX_FFT_CACHE_SIZE:
        # Simple LRU: remove oldest (first) entry
        oldest_key = next(iter(cache))
        cache.pop(oldest_key)
    cache[key] = (freqs.copy(), amp.copy())  # Store copies


def _invalidate_result_caches() -> None:
    """Clear all filter and FFT caches (called on file change)."""
    st.session_state.pop("_filter_cache", None)
    st.session_state.pop("_fft_cache", None)


def _apply_filter_cached(
    series: pd.Series,
    x_series: Optional[pd.Series],
    fspec: FilterSpec,
    fs_value: Optional[float],
    fs_source: Optional[str],
    file_sig: FileSignature,
    column_name: str,
) -> Optional[pd.Series]:
    """Apply filter with caching. Returns filtered series or None if filter fails."""
    _init_result_caches()
    cache_key = _get_filter_cache_key(column_name, file_sig, fspec, fs_value, fs_source)
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


def _compute_fft_cached(
    series: pd.Series,
    fs_value: float,
    fs_source: Optional[str],
    fftspec: FFTSpec,
    file_sig: FileSignature,
    column_name: str,
    is_filtered: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FFT with caching. Returns (freqs, amp) arrays."""
    _init_result_caches()
    cache_key = _get_fft_cache_key(column_name, file_sig, is_filtered, fftspec, fs_value, fs_source)
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

    # Reset plot and report outputs
    st.session_state.pop("_plots_ready", None)
    st.session_state.pop("_generated_report", None)
    st.session_state.pop("_generated_report_error", None)
    st.session_state.pop("_generated_visual_report", None)
    st.session_state.pop("_generated_visual_report_error", None)

    # Reset visual report tracking state
    st.session_state.pop("_visual_report_prev_selection", None)
    st.session_state.pop("_visual_report_last_default_x_label", None)

    # Reset quality mode to default
    st.session_state.pop("_quality_file_sig", None)

    # Reset preset state
    st.session_state.pop("_loaded_preset", None)
    st.session_state.pop("_loaded_preset_name", None)
    st.session_state.pop("_pending_preset_save", None)

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
    if token == "\t" or token == "	":
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
        return pd.to_datetime(xraw, errors="coerce")

    # Prova coerzione numerica
    xnum = pd.to_numeric(xraw, errors="coerce")
    if xnum.notna().mean() >= 0.8:
        return xnum

    # Fallback: stringhe/datetime
    try:
        xdt = pd.to_datetime(xraw, errors="coerce")
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
            return y, pd.to_datetime(xraw, errors="coerce")
        # prova coerzione numerica
        xnum = pd.to_numeric(xraw, errors="coerce")
        if xnum.notna().mean() >= 0.8:
            return y, xnum
        # fallback: stringhe/datetime
        try:
            xdt = pd.to_datetime(xraw, errors="coerce")
            return y, xdt
        except Exception:
            pass
    return y, None


def _plot_xy(x: Optional[pd.Series], y: pd.Series, name: str) -> go.Figure:
    fig = go.Figure()
    if x is not None and x.notna().any():
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
        fig.update_xaxes(title="X")
    else:
        fig.add_trace(go.Scatter(y=y, mode="lines", name=name))
        fig.update_xaxes(title="index")
    fig.update_yaxes(title=name)
    fig.update_layout(margin=dict(l=40, r=20, t=30, b=40), height=420)
    return fig


def _plot_fft(freqs: np.ndarray, amp: np.ndarray, title: str = "FFT") -> go.Figure:
    fig = go.Figure()
    if freqs.size > 0 and amp.size > 0:
        fig.add_trace(go.Scatter(x=freqs, y=amp, mode="lines", name="amp"))
        fig.update_xaxes(title="Frequenza [Hz]")
        fig.update_yaxes(title="Ampiezza")
    fig.update_layout(title=title, margin=dict(l=40, r=20, t=30, b=40), height=420)
    return fig

# ---------------------- Quality checks ---------------------- #
def _load_quality_config() -> Dict[str, Any]:
    """Load quality check configuration from config.json with safe defaults."""
    import json
    defaults = {
        "gap_factor_k": 5.0,
        "spike_z": 4.0,
        "min_points": 20,
        "max_examples": 5
    }
    try:
        config_path = Path("config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get("quality", defaults)
    except Exception:
        pass
    return defaults


def _load_performance_config() -> Dict[str, Any]:
    """Load performance configuration from config.json with safe defaults."""
    import json
    defaults = {
        "optimize_dtypes": True,
        "aggressive_dtype_optimization": False
    }
    try:
        config_path = Path("config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get("performance", defaults)
    except Exception:
        pass
    return defaults


def _render_quality_badge_and_details(report: DataQualityReport) -> None:
    """Render quality badge and collapsible details panel."""
    # Badge styling
    if report.status == 'ok':
        badge_color = "#28a745"
        badge_text = "OK"
        badge_icon = "✓"
    else:
        badge_color = "#ffc107"
        badge_text = "Attenzione"
        badge_icon = "⚠"

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
        with st.expander("📋 Dettagli qualità", expanded=False):
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
                st.markdown("**ℹ️ Note informative:**")
                for note in report.notes:
                    st.info(note)

            # Issues
            if report.has_issues():
                st.markdown("**⚠️ Problemi rilevati:**")
                for idx, issue in enumerate(report.issues, 1):
                    with st.container():
                        if issue.issue_type == 'x_non_monotonic':
                            st.markdown(f"**{idx}. 🔴 X non monotono**")
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
                            st.markdown(f"**{idx}. 🟡 Gap nel campionamento**")
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
                            st.markdown(f"**{idx}. 🔵 Spike in '{col_name}'**")
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
    import base64, mimetypes
    from pathlib import Path

    logo_path = Path("assets/logo.png")  # cambia nome/estensione se necessario
    logo_tag = ""
    if logo_path.exists():
        mime = mimetypes.guess_type(logo_path.name)[0] or "image/png"
        b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
        logo_tag = f"<img class='logo' src='data:{mime};base64,{b64}' alt='Logo'>"

    st.markdown(
        f"""
        <style>
          .app-header {{
            background: transparent;
            border: none;
            box-shadow: none;
            padding: 0;
            margin: 0 0 10px 0;
          }}
          .brand {{
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
          }}
          .brand .logo {{
            height: 96px; width: auto; object-fit: contain;
          }}
          .brand h1 {{
            margin: 4px 0 2px 0;
            font-size: 4rem; line-height: .1; letter-spacing: .2px;
          }}
          .brand .subtitle {{
            margin: 0 0 6px 0; font-size: 1rem;
          }}
          .actions {{
            display: flex; gap: 8px; flex-wrap: wrap;
          }}
          .btn {{
            display: inline-flex; align-items: center; gap: 6px;
            padding: 6px 10px; border-radius: 9999px; font-size: .85rem;
            text-decoration: none !important; border: 1px solid #2b2b2b;
            transition: transform .08s ease, opacity .8s ease, border-color .2s ease, background .2s ease;
          }}
          .btn:active {{ transform: translateY(1px); }}
          .btn-primary {{ background: #f4c430; color: #111; border-color: #d6a300; }}
          .btn-primary:hover {{ background: #e0b51e; border-color: #c79c14; }}
          .btn-ghost {{ background: #141414; color: #e8eaed; border-color: #2b2b2b; }}
          .btn-ghost:hover {{ border-color: #3a3a3a; }}
          .emoji {{ font-size: 1rem; }}

          @media (max-width: 760px) {{
            .brand h1 {{ font-size: 1.35rem; }}
            .brand .subtitle {{ font-size: .9rem; }}
          }}
        </style>

        <div class="app-header">
          <div class="brand">
            {logo_tag}
            <h1>Analizzatore CSV — Web</h1>
            <p class="subtitle">Lean data analysis — Plot, Filtri, FFT e Report</p>
            <div class="actions">
              <a class="btn btn-primary" href="https://buymeacoffee.com/asillav" target="_blank" rel="noopener">
                <span class="emoji">☕</span> Buy me a coffee
              </a>
              <a class="btn btn-ghost" href="https://asillav.github.io/" target="_blank" rel="noopener">
                <span class="emoji">🐙</span> GitHub
              </a>
            </div>
          </div>
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


def main():
    _ensure_session_id()
    st.set_page_config(page_title="Analizzatore CSV - Web", layout="wide")

    # Inizializza preset di default all'avvio
    try:
        create_default_presets()
    except Exception as e:
        logger = LogManager(component="preset").get_logger()
        logger.warning(f"Impossibile creare preset default: {e}")

    render_header()

    st.caption("Upload CSV → seleziona X/Y → limiti assi → Advanced (fs/filtri/FFT) → report")

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
        st.error(f"Errore nel parsing del CSV: {exc}")
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

    # Run quality checks
    quality_config = _load_quality_config()
    try:
        # Get all columns for Y checks (use all cols if available)
        all_cols = list(df.columns)
        quality_report = run_quality_checks(
            df=df,
            x_col=None,  # Will use index, X column will be selected later by user
            y_cols=all_cols,  # Check all columns for spikes
            gap_factor_k=quality_config['gap_factor_k'],
            spike_z=quality_config['spike_z'],
            min_points=quality_config['min_points'],
            max_examples=quality_config['max_examples']
        )
        # Log summary
        from core.logger import LogManager
        logger = LogManager(component="quality").get_logger()
        logger.info(quality_report.get_summary())

        # Render badge and details
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
    preset_notice: Optional[str] = None

    # Carica preset se presente (NON fare pop - mantieni finché non viene fatto submit)
    if "_loaded_preset" in st.session_state:
        preset_data = st.session_state["_loaded_preset"]
        preset_name = st.session_state.get("_loaded_preset_name", "")
        preset_notice = f"📂 Preset '{preset_name}' caricato. Compila il form e premi 'Applica / Plot'."

        manual_from_preset = preset_data.get("manual_fs")
        if manual_from_preset is not None:
            try:
                preset_manual_fs = float(manual_from_preset)
            except (TypeError, ValueError):
                preset_manual_fs = 0.0

        fspec_loaded = preset_data.get("filter_spec")
        if isinstance(fspec_loaded, FilterSpec):
            preset_enable_filter = bool(fspec_loaded.enabled)
            kind_to_index = {"ma": 0, "butter_lp": 1, "butter_hp": 2, "butter_bp": 3}
            preset_filter_kind_idx = kind_to_index.get(fspec_loaded.kind, 0)
            try:
                preset_ma_win = int(fspec_loaded.ma_window)
            except (TypeError, ValueError):
                pass
            try:
                preset_filter_order = int(fspec_loaded.order)
            except (TypeError, ValueError):
                pass
            cutoff_loaded = fspec_loaded.cutoff or (None, None)
            if isinstance(cutoff_loaded, tuple):
                lo_val, hi_val = cutoff_loaded
            else:
                lo_val, hi_val = (None, None)
            preset_f_lo = "" if lo_val in (None, "") else str(lo_val)
            preset_f_hi = "" if hi_val in (None, "") else str(hi_val)

        fftspec_loaded = preset_data.get("fft_spec")
        if isinstance(fftspec_loaded, FFTSpec):
            preset_enable_fft = bool(fftspec_loaded.enabled)
            preset_detrend = bool(fftspec_loaded.detrend)

    if preset_save_message:
        st.success(preset_save_message)

    if preset_notice:
        st.info(preset_notice)

    # --- Controlli (form) --- #
    with st.form(f"controls_{st.session_state.get('_controls_nonce', 0)}"):
        x_col = st.selectbox("Colonna X (opzionale)", options=["—"] + cols, index=0)
        y_cols = st.multiselect("Colonne Y", options=cols)
        mode = st.radio("Modalità grafico", ["Sovrapposto", "Separati", "Cascata"], horizontal=True, index=0)
        quality_mode = st.radio(
            "Alta fedeltà / Prestazioni",
            ["Alta fedeltà", "Prestazioni"],
            horizontal=True,
            key=quality_key,
            help="Prestazioni applica downsampling LTTB a circa 10k punti per serie per migliorare la reattività. Filtri e FFT restano sui dati completi.",
        )
        st.caption(
            "Il downsampling riduce la traccia prima del rendering. Usa Alta fedeltà se ti servono tutti i campioni (possibile lag oltre 100k punti)."
        )

        c1, c2 = st.columns(2)
        with c1:
            x_min_txt = st.text_input("X min (numero o datetime)", placeholder="es. 0 oppure 2024-09-01")
            y_min_txt = st.text_input("Y min (numero)", placeholder="es. -10")
        with c2:
            x_max_txt = st.text_input("X max (numero o datetime)", placeholder="es. 1000 oppure 2024-09-02")
            y_max_txt = st.text_input("Y max (numero)", placeholder="es. 10")

        # ---- ADVANCED ----
        with st.expander("Advanced", expanded=False):
            # fs manuale in cima
            manual_fs = st.number_input(
                "Frequenza di campionamento manuale (Hz)",
                min_value=0.0,
                value=float(preset_manual_fs),
                step=0.1,
                help=">0 forza la fs; 0 = stima automatica dalla X"
            )
            st.caption("Filtri Butterworth e FFT useranno la stessa fs (manuale o stimata).")

            enable_filter = st.checkbox("Abilita filtro", value=preset_enable_filter)
            f_kind = st.selectbox(
                "Tipo filtro",
                filter_kind_options,
                index=preset_filter_kind_idx,
            )
            ma_win = st.number_input("MA - finestra (campioni)", min_value=1, value=int(preset_ma_win), step=1)
            f_order = st.number_input("Butterworth - ordine", min_value=1, value=int(preset_filter_order), step=1)

            cc1, cc2 = st.columns(2)
            with cc1:
                f_lo = st.text_input("Cutoff low (Hz) - LP/HP/BP", value=preset_f_lo, placeholder="es. 5")
            with cc2:
                f_hi = st.text_input("Cutoff high (Hz) - solo BP", value=preset_f_hi, placeholder="es. 20")

            overlay_orig = st.checkbox("Sovrapponi originale e filtrato", value=True)

            st.markdown("---")
            fft_help = (
                "Calcola lo spettro FFT per ogni serie selezionata."
                if fft_available
                else f"Servono almeno {MIN_ROWS_FOR_FFT} campioni per calcolare l'FFT."
            )

            # No key= parameter: widget state is local to the form, resets automatically
            enable_fft = st.checkbox(
                "Calcola FFT",
                value=preset_enable_fft,
                disabled=not fft_available,
                help=fft_help,
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
            )
            detrend = st.checkbox(
                "Detrend (togli media)",
                value=preset_detrend,
                disabled=not fft_available,
            )

        submitted = st.form_submit_button("Applica / Plot")

    # ---- PRESET CONFIGURAZIONI (FUORI DAL FORM) ----
    with st.expander("Preset Configurazioni", expanded=False):
        st.markdown("Salva e riutilizza configurazioni filtri/FFT frequenti.")

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

        # Logica Load Preset
        if load_clicked and selected_preset != "---":
            try:
                preset_data = load_preset(selected_preset)
                st.session_state["_loaded_preset"] = preset_data
                st.session_state["_loaded_preset_name"] = selected_preset
                # Non fare st.rerun() - lascia che i valori vengano applicati al rendering successivo
                st.success(f"✅ Preset '{selected_preset}' caricato! I parametri sono ora attivi nel form sottostante.")
            except PresetError as e:
                st.error(f"❌ Errore caricamento: {e}")

        # Logica Delete Preset
        if delete_clicked and selected_preset != "---":
            try:
                delete_preset(selected_preset)
                st.success(f"🗑️ Preset '{selected_preset}' eliminato.")
                st.rerun()
            except PresetError as e:
                st.error(f"❌ Errore eliminazione: {e}")

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
                st.info("ℹ️ Compila il form sottostante e premi 'Applica / Plot' per completare il salvataggio.")

    if submitted:
        st.session_state["_plots_ready"] = True
        # Pulisci il preset caricato dopo il submit per evitare che rimanga attivo
        st.session_state.pop("_loaded_preset", None)
        st.session_state.pop("_loaded_preset_name", None)

    if not st.session_state.get("_plots_ready"):
        st.info("Compila il form e premi 'Applica / Plot' per visualizzare grafici e report.")
        return

    if not y_cols:
        st.warning("Seleziona almeno una colonna Y.")
        return

    x_name = x_col if (x_col and x_col != "—") else None
    x_values = None
    if x_name and x_name in df.columns:
        # cerco di mantenere il tipo più utile possibile
        if pd.api.types.is_datetime64_any_dtype(df[x_name]) or pd.api.types.is_timedelta64_dtype(df[x_name]):
            x_values = pd.to_datetime(df[x_name], errors="coerce")
        else:
            # preferisco numerico se coerente
            xnum = pd.to_numeric(df[x_name], errors="coerce")
            x_values = xnum if xnum.notna().mean() >= 0.8 else pd.to_datetime(df[x_name], errors="coerce")

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
    else:
        st.warning("fs non disponibile: filtri Butterworth e FFT verranno saltati se richiesti.")
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
            st.session_state["_preset_save_message"] = f"💾 Preset '{pending_save['name']}' salvato con successo!"
            st.rerun()
        except PresetError as e:
            st.error(f"❌ Impossibile salvare preset: {e}")
            st.session_state.pop("_pending_preset_save", None)
        except Exception as exc:
            st.error(f"❌ Errore inatteso salvataggio preset: {exc}")
            st.session_state.pop("_pending_preset_save", None)

    if fftspec.enabled:
        if not fs_value:
            st.warning("FFT disabilitata: fs non disponibile.")
            fftspec.enabled = False
        elif not fs_info.is_uniform:
            detail = '; '.join(fs_info.warnings) if fs_info.warnings else 'campionamento irregolare.'
            st.warning(f"FFT disabilitata: {detail}")
            fftspec.enabled = False

    # --- Parse range assi --- #
    y_for_range = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in y_cols], axis=0)
    yrange = _parse_range_num(y_min_txt, y_max_txt, y_for_range)
    xrange = None
    if x_name and x_values is not None:
        xrange = _parse_range_x(x_min_txt, x_max_txt, x_values)
    else:
        xmin_idx = _to_float_or_none(x_min_txt)
        xmax_idx = _to_float_or_none(x_max_txt)
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
        if representative_col:
            y_repr = pd.to_numeric(df[representative_col], errors="coerce")
            x_repr = x_values if x_values is not None else None

            # Calcola indici di downsampling
            ds_result = downsample_series(
                y_repr,
                x_repr,
                max_points=PERFORMANCE_MAX_POINTS,
                method=PERFORMANCE_METHOD,
            )

            # Pre-decima DF intero usando gli indici
            if ds_result.sampled_count < total_rows:
                df_plot = df.iloc[ds_result.indices].copy()
                df_downsampled = True
                downsampled_metadata = ds_result
                st.caption(
                    f"⚡ Pre-decimazione: {total_rows:,} → {len(df_plot):,} righe "
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

        for yname in y_cols:
            series_plot, x_plot, series_full, x_full = _get_series_sources(yname)
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
                    y_filt_full = _apply_filter_cached(series_full, x_full, fspec, fs_value, fs_info.source, file_sig, yname)
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

            # Originale tratteggiato se richiesto
            if overlay_orig and fspec.enabled and y_filt_plot is not None:
                overlay_label = f"{yname} (originale)"
                reuse_idx = y_main.index if main_meta and main_meta.original_count > main_meta.sampled_count else None
                x_overlay_src = x_full if x_full is not None else x_ser
                x_overlay, y_overlay, overlay_meta = _prepare_plot_series(
                    overlay_label,
                    series_full,
                    x_overlay_src,
                    reuse_index=reuse_idx,
                )
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
            template="plotly_white",
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
                series, x_ser = _make_time_series(df, x_name, yname, x_parsed=x_parsed_orig)
                if series.dropna().empty:
                    continue
                y_filt = None
                if fspec.enabled:
                    y_filt = _apply_filter_cached(series, x_ser, fspec, fs_value, fs_info.source, file_sig, yname)
                y_fft = y_filt if (fspec.enabled and y_filt is not None and fft_use == "Filtrato (se attivo)") else series
                if not fs_value or fs_value <= 0:
                    st.warning(f"FFT non calcolata per {yname}: fs non disponibile.")
                elif not fs_info.is_uniform:
                    detail = "; ".join(fs_info.warnings) if fs_info.warnings else "campionamento irregolare."
                    st.warning(f"FFT non calcolata per {yname}: {detail}")
                else:
                    is_filt = fspec.enabled and y_filt_full is not None and fft_use == "Filtrato (se attivo)"
                    freqs, amp = _compute_fft_cached(y_fft, fs_value, fs_info.source, fftspec, file_sig, yname, is_filt)
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
            else:
                if fspec.enabled:
                    y_filt_full = _apply_filter_cached(series_full, x_full, fspec, fs_value, fs_info.source, file_sig, yname)
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
            fig = _plot_xy(x_plot, y_plot_ds, name=display_name)
            if fig.data:
                fig.data[0].name = _legend_label(display_name, main_meta)
            if yrange:
                fig.update_yaxes(range=yrange)
            if xrange:
                fig.update_xaxes(range=xrange)
            if overlay_orig and fspec.enabled and y_filt_plot is not None:
                overlay_label = f"{yname} (originale)"
                reuse_idx = y_plot_ds.index if main_meta and main_meta.original_count > main_meta.sampled_count else None
                x_overlay_src = x_full if x_full is not None else x_ser
                x_overlay, y_overlay, overlay_meta = _prepare_plot_series(
                    overlay_label,
                    series_full,
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
                    freqs, amp = _compute_fft_cached(y_fft, fs_value, fs_info.source, fftspec, file_sig, yname, is_filt)
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
                    y_filt_full = _apply_filter_cached(series_full, x_full, fspec, fs_value, fs_info.source, file_sig, yname)
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
            if yrange:
                fig.update_yaxes(range=yrange)
            if xrange:
                fig.update_xaxes(range=xrange)
            if overlay_orig and fspec.enabled and y_filt_plot is not None:
                overlay_label = f"{yname} (originale)"
                reuse_idx = y_plot_ds.index if main_meta and main_meta.original_count > main_meta.sampled_count else None
                x_overlay_src = x_full if x_full is not None else x_ser
                x_overlay, y_overlay, overlay_meta = _prepare_plot_series(
                    overlay_label,
                    series_full,
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
                    freqs, amp = _compute_fft_cached(y_fft, fs_value, fs_info.source, fftspec, file_sig, yname, is_filt)
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

    # ---- Report ----
    st.divider()
    st.subheader("Report statistici")
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
                    df, x_name, y_cols, formats=fmt, base_name=base_name or None
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
                st.session_state.pop("_generated_report", None)
                st.session_state["_generated_report_error"] = str(e)

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
                        df=df,
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
                st.session_state.pop("_generated_visual_report", None)
                st.session_state["_generated_visual_report_error"] = str(e)

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
    with st.expander("ℹ️ Info rilevate (clicca per espandere)", expanded=False):
        st.json(meta)


if __name__ == "__main__":
    main()
