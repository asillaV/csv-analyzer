from __future__ import annotations

import inspect
from pathlib import Path
import hashlib
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.analyzer import analyze_csv
from core.loader import load_csv
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
    "enable_fft",
    "fft_use",
    "detrend",
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
SAMPLE_CSV_PATH = Path("assets/sample_timeseries.csv")


def _reset_all_settings() -> None:
    """Reset widgets/output while keeping the current file and cached data."""
    for k in list(RESETTABLE_KEYS):
        st.session_state.pop(k, None)

    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("vis_report_"):
            st.session_state.pop(key, None)

    st.session_state.pop("_plots_ready", None)
    st.session_state.pop("_generated_report", None)
    st.session_state.pop("_generated_report_error", None)
    st.session_state.pop("_generated_visual_report", None)
    st.session_state.pop("_generated_visual_report_error", None)

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

def _make_time_series(df: pd.DataFrame, x_col: Optional[str], y_col: str) -> Tuple[pd.Series, Optional[pd.Series]]:
    y = pd.to_numeric(df[y_col], errors="coerce")
    y.name = y_col
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
    st.set_page_config(page_title="Analizzatore CSV — Web", layout="wide")
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
        sample_clicked = st.button(
            "Carica sample",
            key="load_sample",
            disabled=not sample_available,
            help="Carica un dataset demo multi-canale (segnale + rumore).",
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

    apply_cleaning = st.checkbox(
        "Applica correzione suggerita",
        value=True,
        key="_apply_cleaning",
        help="Rimuove separatori migliaia/decimali incoerenti e converte le colonne numeriche.",
    )

    tmp_path = Path("tmp_upload.csv")
    cleaning_report: Optional[CleaningReport] = None
    meta: Dict[str, Any]

    if using_sample:
        if sample_bytes is None:
            st.error("Sample non disponibile.")
            return
        file_bytes = sample_bytes
    else:
        upload_bytes = upload.getvalue()
        if not upload_bytes:
            st.error("Il file caricato è vuoto.")
            return
        file_bytes = upload_bytes
        upload.seek(0)

    file_sig = (len(file_bytes), hashlib.sha1(file_bytes).hexdigest())

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

    try:
        with st.spinner("Analisi CSV..."):
            if cache_hit:
                df = cached_df  # type: ignore[assignment]
                cleaning_report = cached_report  # type: ignore[assignment]
                meta = dict(cached_meta)  # type: ignore[arg-type]
                if not tmp_path.exists():
                    tmp_path.write_bytes(file_bytes)
            else:
                tmp_path.write_bytes(file_bytes)
                meta = analyze_csv(str(tmp_path))
                df, cleaning_report = load_csv(
                    str(tmp_path),
                    encoding=meta.get("encoding"),
                    delimiter=meta.get("delimiter"),
                    header=meta.get("header"),
                    apply_cleaning=apply_cleaning,
                    return_details=True,
                )
                st.session_state["_cached_df"] = df
                st.session_state["_cached_cleaning_report"] = cleaning_report
                st.session_state["_cached_meta"] = dict(meta)
                st.session_state["_cached_file_sig"] = file_sig
                st.session_state["_cached_apply_cleaning"] = apply_cleaning
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

    with st.container():
        suggestion = cleaning_report.suggestion
        info_cols = st.columns(4)
        info_cols[0].markdown(
            f"**Encoding**<br/>{meta.get('encoding', 'utf-8')}",
            unsafe_allow_html=True,
        )
        info_cols[1].markdown(
            f"**Delimiter**<br/>{_fmt_csv_token(meta.get('delimiter'))}",
            unsafe_allow_html=True,
        )
        info_cols[2].markdown(
            f"**Decimal**<br/>{_fmt_csv_token(suggestion.decimal)}",
            unsafe_allow_html=True,
        )
        info_cols[3].markdown(
            f"**Migliaia**<br/>{_fmt_csv_token(suggestion.thousands)}",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Correzione automatica: {'attiva' if apply_cleaning else 'disattivata'} · "
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

        raw_name = getattr(current_file, "name", tmp_path.name)
        try:
            raw_bytes = tmp_path.read_bytes()
            st.download_button(
                "Scarica CSV originale",
                data=raw_bytes,
                file_name=raw_name,
                mime="text/csv",
            )
        except Exception:
            st.caption("Impossibile preparare il download del CSV originale.")

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
                min_value=0.0, value=0.0, step=0.1,
                help=">0 forza la fs; 0 = stima automatica dalla X"
            )
            st.caption("Filtri Butterworth e FFT useranno la stessa fs (manuale o stimata).")

            enable_filter = st.checkbox("Abilita filtro", value=False)
            f_kind = st.selectbox(
                "Tipo filtro",
                ["Media mobile (MA)", "Butterworth LP", "Butterworth HP", "Butterworth BP"],
                index=0,
            )
            ma_win = st.number_input("MA - finestra (campioni)", min_value=1, value=5, step=1)
            f_order = st.number_input("Butterworth - ordine", min_value=1, value=4, step=1)

            cc1, cc2 = st.columns(2)
            with cc1:
                f_lo = st.text_input("Cutoff low (Hz) - LP/HP/BP", placeholder="es. 5")
            with cc2:
                f_hi = st.text_input("Cutoff high (Hz) - solo BP", placeholder="es. 20")

            overlay_orig = st.checkbox("Sovrapponi originale e filtrato", value=True)

            st.markdown("---")
            fft_help = (
                "Calcola lo spettro FFT per ogni serie selezionata."
                if fft_available
                else f"Servono almeno {MIN_ROWS_FOR_FFT} campioni per calcolare l'FFT."
            )
            if not fft_available:
                st.session_state["enable_fft"] = False

            enable_fft = st.checkbox(
                "Calcola FFT",
                value=bool(st.session_state.get("enable_fft", False)),
                key="enable_fft",
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
                key="fft_use",
                disabled=not fft_available,
            )
            detrend = st.checkbox(
                "Detrend (togli media)",
                value=True,
                key="detrend",
                disabled=not fft_available,
            )

        submitted = st.form_submit_button("Applica / Plot")

    if submitted:
        st.session_state["_plots_ready"] = True

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
        if xmin_idx is not None or xmax_idx is not None:
            # Usa l'indice numerico implicito quando manca una colonna X esplicita
            default_min = 0.0
            default_max = float(len(df) - 1) if len(df) > 0 else 0.0
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
            series, x_ser = _make_time_series(df, x_name, yname)
            if series.dropna().empty:
                st.info(f"'{yname}': nessun dato numerico valido.")
                continue

            # Filtro (se attivo)
            y_filt = None
            ok, msg = validate_filter_spec(fspec, fs_value)
            if fspec.enabled and not ok:
                st.warning(f"Filtro non applicato a {yname}: {msg}")
                y_plot = series
            else:
                if fspec.enabled:
                    try:
                        y_filt, _ = apply_filter(series, x_ser, fspec, fs_override=fs_value)
                        y_plot = y_filt
                    except Exception as e:
                        st.warning(f"Filtro non applicato a {yname}: {e}")
                        y_plot = series
                else:
                    y_plot = series

            name_main = yname + (" (filtrato)" if (fspec.enabled and not overlay_orig) else "")
            x_main, y_main, main_meta = _prepare_plot_series(name_main, y_plot, x_ser)

            # Originale tratteggiato se richiesto
            if overlay_orig and fspec.enabled and y_filt is not None:
                overlay_label = f"{yname} (originale)"
                reuse_idx = y_main.index if main_meta and main_meta.original_count > main_meta.sampled_count else None
                x_overlay, y_overlay, overlay_meta = _prepare_plot_series(
                    overlay_label,
                    series,
                    x_ser,
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
            if overlay_orig and fspec.enabled and y_filt is not None:
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
                series, x_ser = _make_time_series(df, x_name, yname)
                if series.dropna().empty:
                    continue
                y_filt = None
                if fspec.enabled:
                    try:
                        y_filt, _ = apply_filter(series, x_ser, fspec, fs_override=fs_value)
                    except Exception:
                        y_filt = None
                y_fft = y_filt if (fspec.enabled and y_filt is not None and fft_use == "Filtrato (se attivo)") else series
                if not fs_value or fs_value <= 0:
                    st.warning(f"FFT non calcolata per {yname}: fs non disponibile.")
                elif not fs_info.is_uniform:
                    detail = "; ".join(fs_info.warnings) if fs_info.warnings else "campionamento irregolare."
                    st.warning(f"FFT non calcolata per {yname}: {detail}")
                else:
                    freqs, amp = compute_fft(y_fft, fs_value, detrend=fftspec.detrend)
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
            series, x_ser = _make_time_series(df, x_name, yname)
            host = tabs[idx]

            if series.dropna().empty:
                host.info(f"'{yname}': nessun dato numerico valido.")
                continue

            # Filtro
            y_filt = None
            ok, msg = validate_filter_spec(fspec, fs_value)
            if fspec.enabled and not ok:
                host.warning(f"Filtro non applicato a {yname}: {msg}")
                y_plot = series
            else:
                if fspec.enabled:
                    try:
                        y_filt, _ = apply_filter(series, x_ser, fspec, fs_override=fs_value)
                        y_plot = y_filt
                    except Exception as e:
                        host.warning(f"Filtro non applicato a {yname}: {e}")
                        y_plot = series
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
            if overlay_orig and fspec.enabled and y_filt is not None:
                overlay_label = f"{yname} (originale)"
                reuse_idx = y_plot_ds.index if main_meta and main_meta.original_count > main_meta.sampled_count else None
                x_overlay, y_overlay, overlay_meta = _prepare_plot_series(
                    overlay_label,
                    series,
                    x_ser,
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
                y_fft = y_filt if (fspec.enabled and y_filt is not None and fft_use == "Filtrato (se attivo)") else series
                if not fs_value or fs_value <= 0:
                    host.warning(f"FFT non calcolata per {yname}: fs non disponibile.")
                elif not fs_info.is_uniform:
                    detail = "; ".join(fs_info.warnings) if fs_info.warnings else "campionamento irregolare."
                    host.warning(f"FFT non calcolata per {yname}: {detail}")
                else:
                    freqs, amp = compute_fft(y_fft, fs_value, detrend=fftspec.detrend)
                    if freqs.size == 0:
                        host.info(f"FFT non calcolabile per {yname} (serie troppo corta o parametri non validi).")
                    else:
                        _plotly_chart(
                            host,
                            _plot_fft(freqs, amp, title=f"FFT — {yname}"),
                        )

    else:
        # ----- CASCATA: grafici uno sotto l’altro ----- #
        for yname in y_cols:
            series, x_ser = _make_time_series(df, x_name, yname)

            if series.dropna().empty:
                st.info(f"'{yname}': nessun dato numerico valido.")
                continue

            # Filtro
            y_filt = None
            ok, msg = validate_filter_spec(fspec, fs_value)
            if fspec.enabled and not ok:
                st.warning(f"Filtro non applicato a {yname}: {msg}")
                y_plot = series
            else:
                if fspec.enabled:
                    try:
                        y_filt, _ = apply_filter(series, x_ser, fspec, fs_override=fs_value)
                        y_plot = y_filt
                    except Exception as e:
                        st.warning(f"Filtro non applicato a {yname}: {e}")
                        y_plot = series
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
            if overlay_orig and fspec.enabled and y_filt is not None:
                overlay_label = f"{yname} (originale)"
                reuse_idx = y_plot_ds.index if main_meta and main_meta.original_count > main_meta.sampled_count else None
                x_overlay, y_overlay, overlay_meta = _prepare_plot_series(
                    overlay_label,
                    series,
                    x_ser,
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
                y_fft = y_filt if (fspec.enabled and y_filt is not None and fft_use == "Filtrato (se attivo)") else series
                if not fs_value or fs_value <= 0:
                    st.warning(f"FFT non calcolata per {yname}: fs non disponibile.")
                elif not fs_info.is_uniform:
                    detail = "; ".join(fs_info.warnings) if fs_info.warnings else "campionamento irregolare."
                    st.warning(f"FFT non calcolata per {yname}: {detail}")
                else:
                    freqs, amp = compute_fft(y_fft, fs_value, detrend=fftspec.detrend)
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

