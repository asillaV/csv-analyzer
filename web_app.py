from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.analyzer import analyze_csv
from core.loader import load_csv
from core.report_manager import ReportManager
from core.visual_report_manager import VisualPlotSpec, VisualReportManager
from core.signal_tools import (
    FilterSpec,
    FFTSpec,
    resolve_fs,
    validate_filter_spec,
    apply_filter,
    compute_fft,
)

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


def main():
    st.set_page_config(page_title="Analizzatore CSV — Web", layout="wide")
    render_header()

    st.caption("Upload CSV → seleziona X/Y → limiti assi → Advanced (fs/filtri/FFT) → report")

    upload = st.file_uploader("Carica un file CSV", type=["csv"])
    _reset_generated_reports_marker(upload)
    if not upload:
        st.info("Carica un file per iniziare.")
        return

    # Analisi CSV (encoding/delimiter/header/columns)
    with st.spinner("Analisi CSV..."):
        with open(Path("tmp_upload.csv"), "wb") as f:
            f.write(upload.read())
        meta = analyze_csv("tmp_upload.csv")
        df = load_csv(
            "tmp_upload.csv",
            encoding=meta.get("encoding"),
            delimiter=meta.get("delimiter"),
            header=meta.get("header"),
        )

    st.success("File caricato.")
    n_preview = st.slider("Righe di anteprima", 5, 50, 10)
    st.dataframe(df.head(n_preview), use_container_width=True)
    st.caption(f"Mostrate le prime {n_preview} righe su {len(df)} totali.")

    cols = meta.get("columns", list(df.columns))

    # --- Controlli (form) --- #
    with st.form("controls"):
        x_col = st.selectbox("Colonna X (opzionale)", options=["—"] + cols, index=0)
        y_cols = st.multiselect("Colonne Y", options=cols)
        mode = st.radio("Modalità grafico", ["Sovrapposto", "Separati", "Cascata"], horizontal=True, index=0)

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
                f_lo = st.text_input("Cutoff low (Hz) — LP/HP/BP", placeholder="es. 5")
            with cc2:
                f_hi = st.text_input("Cutoff high (Hz) — solo BP", placeholder="es. 20")

            overlay_orig = st.checkbox("Sovrapponi originale e filtrato", value=True)

            st.markdown("---")
            enable_fft = st.checkbox("Calcola FFT", value=False)
            fft_use = st.radio("FFT su", ["Filtrato (se attivo)", "Originale"], horizontal=True, index=0)
            detrend = st.checkbox("Detrend (togli media)", value=True)

        submitted = st.form_submit_button("Applica / Plot")

    if not submitted:
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
    fs_value, fs_src = resolve_fs(x_values, manual_fs if manual_fs > 0 else None)
    if fs_value and fs_value > 0:
        st.info(f"fs [Hz]: **{fs_value:.6g}** ({'manuale' if fs_src=='manual' else 'stimata'})")
    else:
        st.warning("fs non disponibile: filtri Butterworth e FFT verranno saltati se richiesti.")

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

    # --- Parse range assi --- #
    y_for_range = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in y_cols], axis=0)
    yrange = _parse_range_num(y_min_txt, y_max_txt, y_for_range)
    xrange = None
    if x_name and x_values is not None:
        xrange = _parse_range_x(x_min_txt, x_max_txt, x_values)

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

            # Originale tratteggiato se richiesto
            if overlay_orig and fspec.enabled and y_filt is not None:
                combined.add_trace(
                    go.Scatter(
                        x=(x_ser if x_ser is not None else None),
                        y=series,
                        mode="lines",
                        name=f"{yname} (originale)",
                        line=dict(width=1, dash="dot")
                    )
                )

            # Traccia principale (filtrato o originale)
            name_main = yname + (" (filtrato)" if (fspec.enabled and not overlay_orig) else "")
            combined.add_trace(
                go.Scatter(
                    x=(x_ser if x_ser is not None else None),
                    y=y_plot,
                    mode="lines",
                    name=name_main
                )
            )

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

        st.plotly_chart(combined, use_container_width=True)

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
                else:
                    freqs, amp = compute_fft(y_fft, fs_value, detrend=fftspec.detrend)
                    if freqs.size == 0:
                        st.info(f"FFT non calcolabile per {yname} (serie troppo corta o parametri non validi).")
                    else:
                        st.plotly_chart(_plot_fft(freqs, amp, title=f"FFT — {yname}"), use_container_width=True)

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

            fig = _plot_xy(x_ser, y_plot, name=yname + (" (filtrato)" if (fspec.enabled and not overlay_orig) else ""))
            if yrange:
                fig.update_yaxes(range=yrange)
            if xrange:
                fig.update_xaxes(range=xrange)
            if overlay_orig and fspec.enabled and y_filt is not None:
                fig.add_trace(go.Scatter(x=x_ser if x_ser is not None else None, y=series, mode="lines",
                                         name=f"{yname} (originale)", line=dict(width=1, dash="dot")))
                fig.data = fig.data[::-1]
            host.plotly_chart(fig, use_container_width=True)

            # FFT per singola serie
            if fftspec.enabled:
                y_fft = y_filt if (fspec.enabled and y_filt is not None and fft_use == "Filtrato (se attivo)") else series
                if not fs_value or fs_value <= 0:
                    host.warning(f"FFT non calcolata per {yname}: fs non disponibile.")
                else:
                    freqs, amp = compute_fft(y_fft, fs_value, detrend=fftspec.detrend)
                    if freqs.size == 0:
                        host.info(f"FFT non calcolabile per {yname} (serie troppo corta o parametri non validi).")
                    else:
                        host.plotly_chart(_plot_fft(freqs, amp, title=f"FFT — {yname}"), use_container_width=True)

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

            fig = _plot_xy(x_ser, y_plot, name=yname + (" (filtrato)" if (fspec.enabled and not overlay_orig) else ""))
            if yrange:
                fig.update_yaxes(range=yrange)
            if xrange:
                fig.update_xaxes(range=xrange)
            if overlay_orig and fspec.enabled and y_filt is not None:
                fig.add_trace(go.Scatter(x=x_ser if x_ser is not None else None, y=series, mode="lines",
                                         name=f"{yname} (originale)", line=dict(width=1, dash="dot")))
                fig.data = fig.data[::-1]
            st.plotly_chart(fig, use_container_width=True)

            # FFT sotto ogni grafico (se attiva)
            if fftspec.enabled:
                y_fft = y_filt if (fspec.enabled and y_filt is not None and fft_use == "Filtrato (se attivo)") else series
                if not fs_value or fs_value <= 0:
                    st.warning(f"FFT non calcolata per {yname}: fs non disponibile.")
                else:
                    freqs, amp = compute_fft(y_fft, fs_value, detrend=fftspec.detrend)
                    if freqs.size == 0:
                        st.info(f"FFT non calcolabile per {yname} (serie troppo corta o parametri non validi).")
                    else:
                        st.plotly_chart(_plot_fft(freqs, amp, title=f"FFT — {yname}"), use_container_width=True)

    # ---- Report ----
    st.divider()
    st.subheader("Report statistici")
    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        fmt = st.selectbox("Formato", ["csv", "csv+md", "csv+html", "csv+md+html"], index=0)
        base_name = st.text_input("Nome base report (opzionale)", placeholder="es. report_misura_001")
    with col_r2:
        if st.button("Genera report"):
            try:
                out = ReportManager().generate_report(
                    df, x_name, y_cols, formats=fmt, base_name=base_name or None
                )
                st.session_state["_generated_report"] = out
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
        st.json({k: str(v) if v else None for k, v in generated_report.items()})

    st.divider()
    st.subheader("Report visivo dei grafici")
    st.caption("Scegli fino a 4 serie per creare un'immagine o un PDF con i grafici in cascata.")

    visual_selection = st.multiselect(
        "Serie da includere (max 4)",
        options=y_cols,
        default=y_cols[: min(4, len(y_cols))],
        max_selections=4,
        help="Le serie devono essere numeriche; eventuali NaN verranno ignorati.",
    )

    visual_specs: List[VisualPlotSpec] = []
    default_x_label = x_name if x_name else "Index"
    for idx, yname in enumerate(visual_selection):
        with st.expander(f"Grafico {idx + 1} — {yname}", expanded=False):
            plot_title = st.text_input(
                "Titolo grafico",
                value=yname,
                key=f"vis_report_title_{idx}_{yname}",
            )
            x_label = st.text_input(
                "Titolo asse X",
                value=default_x_label,
                key=f"vis_report_xlabel_{idx}_{yname}",
            )
            y_label = st.text_input(
                "Titolo asse Y",
                value=yname,
                key=f"vis_report_ylabel_{idx}_{yname}",
            )
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
        visual_format = st.radio("Formato", ["png", "pdf"], horizontal=True, key="vis_report_format")
        visual_show_legend = st.checkbox("Mostra legenda", value=False, key="vis_report_legend")

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col2:
        generate_visual = st.button("Genera report visivo", use_container_width=True)

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
                        title=visual_title or None,
                        base_name=visual_base or None,
                        file_format=visual_format,
                        show_legend=visual_show_legend,
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
        st.success(f"Report visivo salvato in {visual_result['path']}")
        mime = "application/pdf" if visual_result["format"] == "pdf" else "image/png"
        st.download_button(
            "Scarica report",
            data=visual_result["bytes"],
            file_name=visual_result["path"].name,
            mime=mime,
        )
        if visual_result["format"] == "png":
            st.image(visual_result["bytes"], caption="Anteprima report visivo", use_column_width=True)

    st.divider()
    with st.expander("ℹ️ Info rilevate (clicca per espandere)", expanded=False):
        st.json(meta)


if __name__ == "__main__":
    main()
