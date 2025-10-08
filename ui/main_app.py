from __future__ import annotations

import re
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    Switch,
    Checkbox,
)

from core.analyzer import analyze_csv
from core.loader import load_csv
from core.report_manager import ReportManager
from core.signal_tools import (
    FilterSpec,
    FFTSpec,          # se lo usi altrove
    resolve_fs,
    validate_filter_spec,
    apply_filter,
    compute_fft,
)

__all__ = ["CSVAnalyzerApp"]


# ---------------------- util plotting ---------------------- #
def _plot_xy(x: Optional[pd.Series], y: pd.Series, name: str) -> go.Figure:
    fig = go.Figure()
    if x is not None and x.notna().any():
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
        fig.update_xaxes(title="X")
    else:
        fig.add_trace(go.Scatter(y=y, mode="lines", name=name))
        fig.update_xaxes(title="index")
    fig.update_yaxes(title=name)
    fig.update_layout(margin=dict(l=40, r=20, t=30, b=40), height=520, title=name)
    return fig


def _plot_fft(freqs: np.ndarray, amp: np.ndarray, title: str = "FFT") -> go.Figure:
    fig = go.Figure()
    if freqs.size > 0 and amp.size > 0:
        fig.add_trace(go.Scatter(x=freqs, y=amp, mode="lines", name="amp"))
        fig.update_xaxes(title="Frequenza [Hz]")
        fig.update_yaxes(title="Ampiezza")
    fig.update_layout(title=title, margin=dict(l=40, r=20, t=30, b=40), height=420)
    return fig


def _safe_base(name: str) -> str:
    # toglie caratteri strani per il filesystem
    base = re.sub(r"[^\w\s\-\.,\(\)\[\]]+", "_", name, flags=re.UNICODE).strip()
    return base if base else "plot"


def _save_open_html(fig: go.Figure, base: str) -> Path:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _safe_base(base)
    path = (out_dir / f"{base}.html").resolve()   # <--- assoluto
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    # apertura singola
    try:
        webbrowser.open_new_tab(path.as_uri())
    except Exception:
        # fallback Windows
        webbrowser.open(str(path))
    return path


# ---------------------- App ---------------------- #
class CSVAnalyzerApp(App):
    CSS = """
    #left { width: 42%; min-width: 38%; }
    #right { width: 58%; }
    .box { border: solid #444; padding: 1; margin: 1; }
    """

    BINDINGS = [
        ("q", "quit", "Esci"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.df: Optional[pd.DataFrame] = None
        self.meta: Dict = {}
        self.columns: List[str] = []
        self.x_col: Optional[str] = None
        self.y_cols: List[str] = []  # ricalcolata dai checkbox

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left"):
                # --- Sorgente file
                with VerticalScroll(classes="box"):
                    yield Label("Percorso CSV:")
                    yield Input(placeholder="C:\\path\\file.csv o /path/file.csv", id="csv_path")
                    yield Button("Analizza", id="btn_analyze")

                # --- Scelta X e limiti
                with VerticalScroll(classes="box"):
                    yield Label("Colonna X (opzionale):")
                    yield Select(options=[("—", "none")], id="sel_x")
                    yield Label("Limiti assi (opzionali):")
                    yield Input(placeholder="X min", id="x_min")
                    yield Input(placeholder="X max", id="x_max")
                    yield Input(placeholder="Y min", id="y_min")
                    yield Input(placeholder="Y max", id="y_max")

                # --- Selezione Y con checkbox
                with VerticalScroll(classes="box"):
                    yield Label("Selezione Y (usa i checkbox)")
                    with Horizontal():
                        yield Button("Seleziona tutte", id="btn_y_all")
                        yield Button("Pulisci", id="btn_y_none")
                    yield VerticalScroll(id="y_checks")

                # --- Advanced
                with VerticalScroll(classes="box"):
                    yield Label("Advanced")
                    yield Label("fs [Hz] (0=auto):")
                    yield Input(placeholder="0", id="adv_fs")
                    yield Label("Filtro")
                    yield Select((("Nessuno (MA disatt.)", "none"),
                                  ("Media mobile (MA)", "ma"),
                                  ("Butterworth LP", "butter_lp"),
                                  ("Butterworth HP", "butter_hp"),
                                  ("Butterworth BP", "butter_bp")), id="f_kind")
                    yield Input(placeholder="MA window (campioni, es. 5)", id="ma_win")
                    yield Input(placeholder="Ordine Butterworth (es. 4)", id="f_order")
                    yield Input(placeholder="Cutoff low (Hz)", id="f_lo")
                    yield Input(placeholder="Cutoff high (Hz, solo BP)", id="f_hi")
                    yield Switch(value=True, id="overlay")
                    yield Label("FFT")
                    yield Switch(value=False, id="fft_on")
                    yield Select((("Filtrato (se attivo)", "filtered"),
                                  ("Originale", "original")), id="fft_on_what")
                    yield Switch(value=True, id="fft_detrend")

                # --- Azioni
                with VerticalScroll(classes="box"):
                    yield Button("Plot", id="btn_plot")
                    yield Button("Report CSV", id="btn_report")

            with Vertical(id="right"):
                yield Label("Colonne disponibili (elenco)")
                yield DataTable(id="table")
                yield Static("", id="status")

        yield Footer()

    # ---------- helpers ---------- #
    def _set_status(self, msg: str) -> None:
        self.query_one("#status", Static).update(Text(msg))

    def _refresh_columns(self) -> None:
        # aggiorna Select X
        sel_x = self.query_one("#sel_x", Select)
        opts = [("—", "none")] + [(c, c) for c in self.columns]
        try:
            sel_x.set_options(opts)
        except Exception:
            try:
                sel_x.clear_options()
                sel_x.add_options(opts)
            except Exception:
                pass
        # aggiorna tabella informativa
        self._update_table()
        # ricrea la checklist Y
        self._render_y_checklist()

    def _update_table(self) -> None:
        tbl = self.query_one("#table", DataTable)
        tbl.clear(columns=True)
        tbl.add_column("Colonna", key="col")
        for c in self.columns:
            tbl.add_row(c)

    def _render_y_checklist(self) -> None:
        box = self.query_one("#y_checks", VerticalScroll)
        for child in list(box.children):
            try:
                child.remove()
            except Exception:
                pass
        for i, col in enumerate(self.columns):
            checked = col in self.y_cols
            box.mount(Checkbox(col, value=checked, id=f"ycb_{i}"))

    def _read_y_from_checkboxes(self) -> List[str]:
        box = self.query_one("#y_checks", VerticalScroll)
        y_cols: List[str] = []
        for i, col in enumerate(self.columns):
            try:
                cb = box.query_one(f"#ycb_{i}", Checkbox)
                if getattr(cb, "value", False):
                    y_cols.append(col)
            except Exception:
                continue
        return y_cols

    # ---------- events ---------- #
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_analyze":
            self._do_analyze()
        elif event.button.id == "btn_plot":
            self._do_plot()
        elif event.button.id == "btn_report":
            self._do_report()
        elif event.button.id == "btn_y_all":
            self._y_select_all(True)
        elif event.button.id == "btn_y_none":
            self._y_select_all(False)

    # ---------- actions ---------- #
    def _y_select_all(self, state: bool) -> None:
        box = self.query_one("#y_checks", VerticalScroll)
        for i, col in enumerate(self.columns):
            try:
                cb = box.query_one(f"#ycb_{i}", Checkbox)
                cb.value = state
            except Exception:
                pass
        self.y_cols = list(self.columns) if state else []
        self._set_status(f"Y selezionate: {', '.join(self.y_cols) if self.y_cols else '(nessuna)'}")

    def _do_analyze(self) -> None:
        path = self.query_one("#csv_path", Input).value.strip()
        if not path:
            self.notify("Percorso vuoto.", severity="warning"); return
        try:
            self.meta = analyze_csv(path)
            self.df = load_csv(path, encoding=self.meta.get("encoding"),
                                     delimiter=self.meta.get("delimiter"),
                                     header=self.meta.get("header"))
            self.columns = self.meta.get("columns", list(self.df.columns))
            self._refresh_columns()
            self.notify("CSV caricato.", severity="information")
        except Exception as e:
            self.notify(f"Errore analisi/caricamento: {e}", severity="error")

    def _do_plot(self) -> None:
        if self.df is None or not len(self.df):
            self.notify("Nessun dataframe.", severity="warning"); return

        # rileggo Y dai checkbox
        self.y_cols = self._read_y_from_checkboxes()
        if not self.y_cols:
            self.notify("Seleziona almeno una Y con i checkbox.", severity="warning"); return

        # X
        x_sel = self.query_one("#sel_x", Select).value
        x_col = None if (x_sel in (None, "none") or (not isinstance(x_sel, str))) else str(x_sel)
        x_values: Optional[pd.Series] = None
        if x_col and x_col in self.df.columns:
            xraw = self.df[x_col]
            if pd.api.types.is_datetime64_any_dtype(xraw) or pd.api.types.is_timedelta64_dtype(xraw):
                x_values = pd.to_datetime(xraw, errors="coerce")
            else:
                xnum = pd.to_numeric(xraw, errors="coerce")
                x_values = xnum if xnum.notna().mean() >= 0.8 else pd.to_datetime(xraw, errors="coerce")

        # fs manuale
        fs_txt = self.query_one("#adv_fs", Input).value or "0"
        try:
            manual_fs = float(fs_txt.replace(",", "."))
        except Exception:
            manual_fs = 0.0
        fs_value, fs_src = resolve_fs(x_values, manual_fs if manual_fs > 0 else None)
        if fs_value:
            self.notify(f"fs: {fs_value:.6g} ({'manuale' if fs_src=='manual' else 'stimata'})", severity="information")
        else:
            self.notify("fs non disponibile: filtri Butterworth e FFT verranno saltati.", severity="warning")

        # filtro spec
        val_kind = self.query_one("#f_kind", Select).value
        f_kind = val_kind if isinstance(val_kind, str) and val_kind else "none"   # evita Select.BLANK
        f_order = int((self.query_one("#f_order", Input).value or "4").strip() or "4")
        ma_win = int((self.query_one("#ma_win", Input).value or "5").strip() or "5")
        f_lo = self.query_one("#f_lo", Input).value.strip()
        f_hi = self.query_one("#f_hi", Input).value.strip()

        def _tof(s: str) -> Optional[float]:
            if not s: return None
            try: return float(s.replace(",", "."))
            except Exception: return None

        lo = _tof(f_lo); hi = _tof(f_hi)

        cutoff: Optional[Tuple[Optional[float], Optional[float]]] = None
        if f_kind in ("butter_lp", "butter_hp") and lo is not None:
            cutoff = (lo, None)
        elif f_kind == "butter_bp" and lo is not None and hi is not None and hi > lo:
            cutoff = (lo, hi)

        fspec = FilterSpec(
            kind="ma" if f_kind == "none" else f_kind,
            enabled=(f_kind != "none"),
            order=f_order,
            cutoff=cutoff,
            ma_window=ma_win,
        )
        overlay = self.query_one("#overlay", Switch).value

        # FFT
        fft_on = self.query_one("#fft_on", Switch).value
        val_fft_w = self.query_one("#fft_on_what", Select).value
        fft_on_what = val_fft_w if isinstance(val_fft_w, str) and val_fft_w else "filtered"
        fft_detrend = self.query_one("#fft_detrend", Switch).value

        # limiti (solo Y per semplicità in TUI)
        y_min_txt = self.query_one("#y_min", Input).value
        y_max_txt = self.query_one("#y_max", Input).value
        y_min = None if not y_min_txt else _tof(y_min_txt)
        y_max = None if not y_max_txt else _tof(y_max_txt)

        # loop serie
        for yname in self.y_cols:
            y = pd.to_numeric(self.df[yname], errors="coerce").dropna()
            if y.empty:
                self.notify(f"'{yname}': nessun dato numerico valido.", severity="warning")
                continue

            y_plot = y
            y_filt = None

            ok, msg = validate_filter_spec(fspec, fs_value)
            if not ok and fspec.enabled:
                self.notify(f"Filtro non applicato a {yname}: {msg}", severity="warning")
            else:
                if fspec.enabled:
                    try:
                        y_filt, _ = apply_filter(y, x_values, fspec, fs_override=fs_value)
                        y_plot = y_filt
                    except Exception as e:
                        self.notify(f"Filtro non applicato a {yname}: {e}", severity="warning")
                        y_plot = y

            fig = _plot_xy(x_values, y_plot, name=yname + (" (filtrato)" if (fspec.enabled and not overlay) else ""))
            if y_min is not None or y_max is not None:
                yr = list(fig.layout.yaxis.range) if fig.layout.yaxis.range else [None, None]
                fig.update_yaxes(range=[y_min if y_min is not None else yr[0],
                                        y_max if y_max is not None else yr[1]])

            if overlay and fspec.enabled and y_filt is not None:
                fig.add_trace(go.Scatter(x=x_values if x_values is not None else None, y=y, mode="lines", name=f"{yname} (originale)", line=dict(width=1, dash="dot")))
                fig.data = fig.data[::-1]

            _save_open_html(fig, base=f"plot_{yname}")

            # FFT
            if fft_on:
                y_fft = y_filt if (fspec.enabled and fft_on_what == "filtered" and y_filt is not None) else y
                if not fs_value or fs_value <= 0:
                    self.notify(f"FFT non calcolata per {yname}: fs non disponibile.", severity="warning")
                else:
                    freqs, amp = compute_fft(y_fft, fs_value, detrend=fft_detrend)
                    if freqs.size == 0:
                        self.notify(f"FFT non calcolabile per {yname} (serie troppo corta o parametri non validi).", severity="information")
                    else:
                        _save_open_html(_plot_fft(freqs, amp, title=f"FFT — {yname}"), base=f"fft_{yname}")

        self._set_status("Plot completati. File salvati in outputs/ e aperti nel browser.")

    def _do_report(self) -> None:
        if self.df is None:
            self.notify("Nessun dataframe.", severity="warning"); return

        # rileggo Y dai checkbox
        self.y_cols = self._read_y_from_checkboxes()
        if not self.y_cols:
            self.notify("Seleziona almeno una Y con i checkbox.", severity="warning"); return

        try:
            out = ReportManager().generate_report(self.df, self.x_col, self.y_cols, formats="csv")
            self.notify(f"Report generato: { {k: str(v) for k,v in out.items() if v} }", severity="information")
        except Exception as e:
            self.notify(f"Errore generazione report: {e}", severity="error")


if __name__ == "__main__":
    CSVAnalyzerApp().run()
