from __future__ import annotations

import re
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Tkinter (incluso in Python su Windows)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# Core del progetto
from core.analyzer import analyze_csv
from core.csv_cleaner import CleaningReport
from core.loader import load_csv
from core.report_manager import ReportManager
from core.signal_tools import (
    FilterSpec,
    resolve_fs,
    validate_filter_spec,
    apply_filter,
    compute_fft,
)

# ---------------------- Util ---------------------- #
def _safe_base(name: str) -> str:
    base = re.sub(r"[^\w\s\-\.,\(\)\[\]]+", "_", name, flags=re.UNICODE).strip()
    return base if base else "plot"

def save_plot_html(fig: go.Figure, base: str) -> Path:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _safe_base(base)
    p = (out_dir / f"{base}.html").resolve()
    fig.write_html(str(p), include_plotlyjs="cdn", full_html=True)
    try:
        webbrowser.open_new_tab(p.as_uri())
    except Exception:
        webbrowser.open(str(p))
    return p

def plot_xy(x: Optional[pd.Series], y: pd.Series, name: str) -> go.Figure:
    fig = go.Figure()
    if x is not None and len(x) == len(y):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
        fig.update_xaxes(title="X")
    else:
        fig.add_trace(go.Scatter(y=y, mode="lines", name=name))
        fig.update_xaxes(title="index")
    fig.update_yaxes(title=name)
    fig.update_layout(margin=dict(l=40, r=20, t=30, b=40), height=520, title=name)
    return fig

def plot_fft(freqs: np.ndarray, amp: np.ndarray, title: str="FFT") -> go.Figure:
    fig = go.Figure()
    if freqs.size > 0 and amp.size > 0:
        fig.add_trace(go.Scatter(x=freqs, y=amp, mode="lines", name="amp"))
        fig.update_xaxes(title="Frequenza [Hz]")
        fig.update_yaxes(title="Ampiezza")
    fig.update_layout(title=title, margin=dict(l=40, r=20, t=30, b=40), height=420)
    return fig

def to_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s.replace(",", "."))
    except Exception:
        return None

def resolve_x_values(df: pd.DataFrame, x_col_name: Optional[str]) -> Optional[pd.Series]:
    if not x_col_name or x_col_name == "—" or x_col_name not in df.columns:
        return None
    xraw = df[x_col_name]
    if pd.api.types.is_datetime64_any_dtype(xraw) or pd.api.types.is_timedelta64_dtype(xraw):
        return pd.to_datetime(xraw, errors="coerce")
    xnum = pd.to_numeric(xraw, errors="coerce")
    return xnum if xnum.notna().mean() >= 0.8 else pd.to_datetime(xraw, errors="coerce")

def is_datetime_like(x: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(x) or pd.api.types.is_timedelta64_dtype(x)

def build_x_mask(x: Optional[pd.Series], xmin_txt: str, xmax_txt: str) -> Tuple[Optional[pd.Series], Optional[int], Optional[int], Optional[str]]:
    """
    Ritorna:
      - mask (Series booleana) se X esiste,
      - start_idx, end_idx se X non esiste (indice),
      - errore (stringa) se input non valido.
    """
    xmin_txt = (xmin_txt or "").strip()
    xmax_txt = (xmax_txt or "").strip()

    if x is None:
        # Slice su indice (posizioni)
        start_idx = int(to_float(xmin_txt)) if xmin_txt else None
        end_idx = int(to_float(xmax_txt)) if xmax_txt else None
        if (start_idx is not None) and (end_idx is not None) and start_idx > end_idx:
            return None, None, None, "X min > X max (indice)."
        return None, start_idx, end_idx, None

    # Abbiamo X
    if is_datetime_like(x):
        xmin = pd.to_datetime(xmin_txt, errors="coerce") if xmin_txt else None
        xmax = pd.to_datetime(xmax_txt, errors="coerce") if xmax_txt else None
    else:
        xmin = to_float(xmin_txt) if xmin_txt else None
        xmax = to_float(xmax_txt) if xmax_txt else None

    if (xmin is not None) and (xmax is not None) and (xmin > xmax):
        return None, None, None, "X min > X max."
    mask = pd.Series(True, index=x.index)
    if xmin is not None:
        mask &= (x >= xmin)
    if xmax is not None:
        mask &= (x <= xmax)
    return mask, None, None, None

def align_xy_after_mask(x: Optional[pd.Series], y: pd.Series, mask: Optional[pd.Series], start_idx: Optional[int], end_idx: Optional[int]) -> Tuple[Optional[pd.Series], pd.Series]:
    """
    Applica il masking/slicing a X e Y e allinea (dropna su Y alla fine).
    - Se x è None: usa slicing per indice (posizioni).
    """
    if x is None:
        # Slicing per indice posizionale
        y2 = y.copy()
        if start_idx is not None or end_idx is not None:
            i0 = start_idx or 0
            i1 = (end_idx + 1) if end_idx is not None else len(y2)
            i0 = max(0, min(i0, len(y2)))
            i1 = max(i0, min(i1, len(y2)))
            y2 = y2.iloc[i0:i1]
        y2 = pd.to_numeric(y2, errors="coerce").dropna()
        return None, y2

    # Abbiamo X e maschera
    df_xy = pd.DataFrame({"x": x, "y": pd.to_numeric(y, errors="coerce")})
    if mask is not None:
        df_xy = df_xy[mask]
    # dropna su y e allinea x
    df_xy = df_xy[df_xy["y"].notna()]
    x2 = df_xy["x"]
    y2 = df_xy["y"]
    return x2, y2

# ---------------------- App Tk ---------------------- #
class DesktopAppTk(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CSV Analyzer (Desktop)")
        self.geometry("1250x760")

        # Stato
        self.df: Optional[pd.DataFrame] = None
        self.columns: List[str] = []
        self.cleaning_report: Optional[CleaningReport] = None

        # --- Layout base
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="nsw")
        right = ttk.Frame(self, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # ====== Sorgente file ======
        file_frame = ttk.LabelFrame(left, text="Sorgente CSV", padding=8)
        file_frame.grid(row=0, column=0, sticky="ew", pady=(0,8))
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="CSV:").grid(row=0, column=0, sticky="w")
        self.csv_entry = ttk.Entry(file_frame, width=46)
        self.csv_entry.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(file_frame, text="Sfoglia", command=self._browse).grid(row=0, column=2, padx=2)
        ttk.Button(file_frame, text="Analizza", command=self._analyze).grid(row=0, column=3, padx=2)

        # ====== X + fs + slice X ======
        xf_frame = ttk.LabelFrame(left, text="Asse X + fs", padding=8)
        xf_frame.grid(row=1, column=0, sticky="ew", pady=(0,8))
        xf_frame.columnconfigure(1, weight=1)

        ttk.Label(xf_frame, text="Colonna X:").grid(row=0, column=0, sticky="w")
        self.x_combo = ttk.Combobox(xf_frame, values=["—"], state="readonly", width=42)
        self.x_combo.set("—")
        self.x_combo.grid(row=0, column=1, sticky="ew", padx=4)

        ttk.Label(xf_frame, text="fs [Hz] (0=auto):").grid(row=1, column=0, sticky="w")
        self.fs_entry = ttk.Entry(xf_frame, width=12); self.fs_entry.insert(0, "0")
        self.fs_entry.grid(row=1, column=1, sticky="w", padx=4)

        # Slice X
        ttk.Label(xf_frame, text="X min:").grid(row=2, column=0, sticky="w")
        self.xmin_entry = ttk.Entry(xf_frame, width=18); self.xmin_entry.grid(row=2, column=1, sticky="w", padx=4)
        ttk.Label(xf_frame, text="X max:").grid(row=3, column=0, sticky="w")
        self.xmax_entry = ttk.Entry(xf_frame, width=18); self.xmax_entry.grid(row=3, column=1, sticky="w", padx=4)
        ttk.Label(xf_frame, text="(Numerico o datetime es. 2025-10-08 12:00:00)").grid(row=4, column=1, sticky="w")

        # ====== Selezione Y ======
        y_frame = ttk.LabelFrame(left, text="Selezione Y", padding=8)
        y_frame.grid(row=2, column=0, sticky="ew", pady=(0,8))
        ttk.Button(y_frame, text="Seleziona tutte", command=lambda: self._y_select_all(True)).grid(row=0, column=0, padx=2, pady=2, sticky="w")
        ttk.Button(y_frame, text="Pulisci", command=lambda: self._y_select_all(False)).grid(row=0, column=1, padx=2, pady=2, sticky="w")
        self.y_list = tk.Listbox(y_frame, selectmode=tk.EXTENDED, height=10, width=50)
        self.y_list.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4,0))

        # ====== Limiti Y ======
        lim_frame = ttk.LabelFrame(left, text="Limiti Y", padding=8)
        lim_frame.grid(row=3, column=0, sticky="ew", pady=(0,8))
        ttk.Label(lim_frame, text="Y min:").grid(row=0, column=0, sticky="w")
        self.ymin_entry = ttk.Entry(lim_frame, width=10); self.ymin_entry.grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(lim_frame, text="Y max:").grid(row=0, column=2, sticky="w")
        self.ymax_entry = ttk.Entry(lim_frame, width=10); self.ymax_entry.grid(row=0, column=3, sticky="w", padx=4)

        # ====== Filtro ======
        filt_frame = ttk.LabelFrame(left, text="Filtro", padding=8)
        filt_frame.grid(row=4, column=0, sticky="ew", pady=(0,8))
        ttk.Label(filt_frame, text="Tipo:").grid(row=0, column=0, sticky="w")
        self.fkind_combo = ttk.Combobox(filt_frame, values=["none","ma","butter_lp","butter_hp","butter_bp"], state="readonly", width=12)
        self.fkind_combo.set("none")
        self.fkind_combo.grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(filt_frame, text="MA win:").grid(row=0, column=2, sticky="w")
        self.mawin_entry = ttk.Entry(filt_frame, width=6); self.mawin_entry.insert(0,"5"); self.mawin_entry.grid(row=0, column=3, sticky="w", padx=4)
        ttk.Label(filt_frame, text="Ordine:").grid(row=0, column=4, sticky="w")
        self.ford_entry = ttk.Entry(filt_frame, width=6); self.ford_entry.insert(0,"4"); self.ford_entry.grid(row=0, column=5, sticky="w", padx=4)
        ttk.Label(filt_frame, text="Cutoff lo (Hz):").grid(row=1, column=0, sticky="w")
        self.flo_entry = ttk.Entry(filt_frame, width=10); self.flo_entry.grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(filt_frame, text="Cutoff hi (BP):").grid(row=1, column=2, sticky="w")
        self.fhi_entry = ttk.Entry(filt_frame, width=10); self.fhi_entry.grid(row=1, column=3, sticky="w", padx=4)
        self.overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(filt_frame, text="Overlay originale (filtrato+originale)", variable=self.overlay_var).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4,0))

        # ====== FFT ======
        fft_frame = ttk.LabelFrame(left, text="FFT", padding=8)
        fft_frame.grid(row=5, column=0, sticky="ew", pady=(0,8))
        self.fft_on_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fft_frame, text="Abilita FFT", variable=self.fft_on_var).grid(row=0, column=0, sticky="w")
        ttk.Label(fft_frame, text="Su:").grid(row=0, column=1, sticky="e")
        self.fft_what_combo = ttk.Combobox(fft_frame, values=["filtered","original"], state="readonly", width=10)
        self.fft_what_combo.set("filtered")
        self.fft_what_combo.grid(row=0, column=2, sticky="w", padx=4)
        self.detrend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(fft_frame, text="Detrend", variable=self.detrend_var).grid(row=0, column=3, sticky="w")

        # ====== Modalità di plot ======
        mode_frame = ttk.LabelFrame(left, text="Modalità di plot", padding=8)
        mode_frame.grid(row=6, column=0, sticky="ew", pady=(0,8))
        ttk.Label(mode_frame, text="Serie Y:").grid(row=0, column=0, sticky="w")
        self.plot_mode_combo = ttk.Combobox(mode_frame, values=["separati","sovrapposti"], state="readonly", width=12)
        self.plot_mode_combo.set("separati")
        self.plot_mode_combo.grid(row=0, column=1, sticky="w", padx=4)

        # ====== Azioni ======
        act_frame = ttk.Frame(left, padding=(0,8))
        act_frame.grid(row=7, column=0, sticky="ew")
        ttk.Button(act_frame, text="Plot", command=self._plot, style="Accent.TButton").grid(row=0, column=0, padx=4)
        ttk.Button(act_frame, text="Report CSV", command=self._report).grid(row=0, column=1, padx=4)

        # ====== Log ======
        ttk.Label(right, text="Log").grid(row=0, column=0, sticky="w")
        self.log = ScrolledText(right, height=35)
        self.log.grid(row=1, column=0, sticky="nsew")
        self._append_log("Pronto.")

        # Stili (opzionale)
        try:
            self.style = ttk.Style(self)
            if sys.platform == "win32":
                self.style.theme_use("winnative")
        except Exception:
            pass

    # === Helpers UI ===
    def _append_log(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _browse(self) -> None:
        path = filedialog.askopenfilename(title="Seleziona CSV", filetypes=[("CSV","*.csv"),("Tutti","*.*")])
        if path:
            self.csv_entry.delete(0, "end")
            self.csv_entry.insert(0, path)

    def _refresh_columns(self) -> None:
        vals = ["—"] + self.columns
        self.x_combo.configure(values=vals)
        self.x_combo.set("—")
        self.y_list.delete(0, "end")
        for c in self.columns:
            self.y_list.insert("end", c)

    def _y_select_all(self, state: bool) -> None:
        self.y_list.selection_clear(0, "end")
        if state:
            self.y_list.selection_set(0, "end")

    # === Actions ===
    def _analyze(self) -> None:
        path = self.csv_entry.get().strip()
        if not path:
            messagebox.showwarning("Attenzione", "Seleziona un file CSV.")
            return
        try:
            meta = analyze_csv(path)
            self.df, self.cleaning_report = load_csv(
                path,
                encoding=meta.get("encoding"),
                delimiter=meta.get("delimiter"),
                header=meta.get("header"),
                return_details=True,
            )
            self.columns = meta.get("columns", list(self.df.columns))
            self._refresh_columns()
            self._append_log(f"CSV caricato: {len(self.df)} righe, {len(self.columns)} colonne.")
            if self.cleaning_report:
                suggestion = self.cleaning_report.suggestion
                self._append_log(
                    f"Formato numerico: decimale={suggestion.decimal}, migliaia={suggestion.thousands or 'nessuno'} "
                    f"(conf. {suggestion.confidence:.0%})"
                )
        except Exception as e:
            messagebox.showerror("Errore", f"Analisi/caricamento fallito: {e}")

    def _plot(self) -> None:
        if self.df is None or self.df.empty:
            messagebox.showwarning("Attenzione", "Carica prima un CSV.")
            return
        # Y selezionate
        sel_idx = list(self.y_list.curselection())
        if not sel_idx:
            messagebox.showwarning("Attenzione", "Seleziona almeno una Y dall’elenco.")
            return
        y_names = [self.columns[i] for i in sel_idx]

        # X
        x_col = self.x_combo.get()
        x_series = resolve_x_values(self.df, x_col if x_col and x_col != "—" else None)

        # Slice X
        x_min_txt = self.xmin_entry.get()
        x_max_txt = self.xmax_entry.get()
        mask, start_idx, end_idx, err = build_x_mask(x_series, x_min_txt, x_max_txt)
        if err:
            messagebox.showwarning("Attenzione", f"Slice X non valido: {err}")
            # proseguo senza slice

        # fs (risolta su X slice se presente)
        manual_fs = to_float(self.fs_entry.get()) or 0.0
        x_for_fs = None
        if x_series is not None:
            x_for_fs = x_series[mask] if mask is not None else x_series
        fs_value, fs_src = resolve_fs(x_for_fs, manual_fs if manual_fs > 0 else None)
        if fs_value:
            self._append_log(f"fs: {fs_value:.6g} ({'manuale' if fs_src=='manual' else 'stimata'})")
        else:
            self._append_log("fs non disponibile: Butterworth/FFT verranno saltati se richiesti.")

        # Filtro
        f_kind = self.fkind_combo.get() or "none"
        f_order = int(to_float(self.ford_entry.get()) or 4)
        ma_win = int(to_float(self.mawin_entry.get()) or 5)
        lo = to_float(self.flo_entry.get())
        hi = to_float(self.fhi_entry.get())
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
        overlay = bool(self.overlay_var.get())

        # FFT
        fft_on = bool(self.fft_on_var.get())
        fft_what = self.fft_what_combo.get() or "filtered"
        detrend = bool(self.detrend_var.get())

        # Limiti Y
        y_min = to_float(self.ymin_entry.get())
        y_max = to_float(self.ymax_entry.get())

        # Modalità
        mode = self.plot_mode_combo.get() or "separati"

        # Preparo dati per plotting
        series_list = []  # [(name, x_used, y_used, y_orig_or_none), ...]
        for yname in y_names:
            y_all = pd.to_numeric(self.df[yname], errors="coerce")
            # Applica slice/allineamento
            x_used, y_used = align_xy_after_mask(x_series, y_all, mask, start_idx, end_idx)

            if y_used.empty:
                self._append_log(f"'{yname}': nessun dato numerico valido dopo slice.")
                continue

            y_plot = y_used
            y_orig_for_overlay = None

            # Validazione filtro
            ok, msg = validate_filter_spec(fspec, fs_value)
            if not ok and fspec.enabled:
                self._append_log(f"Filtro non applicato a {yname}: {msg}")
            else:
                if fspec.enabled:
                    try:
                        y_filt, _ = apply_filter(y_used, x_used, fspec, fs_override=fs_value)
                        y_orig_for_overlay = y_used
                        y_plot = y_filt
                    except Exception as e:
                        self._append_log(f"Filtro non applicato a {yname}: {e}")
                        y_plot = y_used

            series_list.append((yname, x_used, y_plot, y_orig_for_overlay))

        if not series_list:
            self._append_log("Nessuna serie valida da plottare.")
            return

        # Plot
        if mode == "sovrapposti":
            fig = go.Figure()
            has_x = any(x is not None for _, x, _, _ in series_list)
            for (yname, x_used, y_plot, y_orig) in series_list:
                if has_x and x_used is not None and len(x_used) == len(y_plot):
                    fig.add_trace(go.Scatter(x=x_used, y=y_plot, mode="lines", name=f"{yname} (filtrato)" if fspec.enabled else yname))
                    if overlay and fspec.enabled and y_orig is not None:
                        fig.add_trace(go.Scatter(x=x_used, y=y_orig, mode="lines", name=f"{yname} (originale)", line=dict(width=1, dash="dot")))
                else:
                    fig.add_trace(go.Scatter(y=y_plot, mode="lines", name=f"{yname} (filtrato)" if fspec.enabled else yname))
                    if overlay and fspec.enabled and y_orig is not None:
                        fig.add_trace(go.Scatter(y=y_orig, mode="lines", name=f"{yname} (originale)", line=dict(width=1, dash="dot")))
            fig.update_xaxes(title="X" if has_x else "index")
            fig.update_yaxes(title="Y")
            fig.update_layout(margin=dict(l=40, r=20, t=30, b=40), height=620, title="Serie Y sovrapposte")
            if y_min is not None or y_max is not None:
                fig.update_yaxes(range=[y_min, y_max])
            save_plot_html(fig, base="plot_Y_sovrapposti")
        else:
            # separati: uno per serie (come prima)
            for (yname, x_used, y_plot, y_orig) in series_list:
                fig = plot_xy(x_used, y_plot, name=yname + (" (filtrato)" if (fspec.enabled and not overlay) else ""))
                if y_min is not None or y_max is not None:
                    fig.update_yaxes(range=[y_min, y_max])
                if overlay and fspec.enabled and y_orig is not None:
                    if x_used is not None and len(x_used) == len(y_orig):
                        fig.add_trace(go.Scatter(x=x_used, y=y_orig, mode="lines", name=f"{yname} (originale)", line=dict(width=1, dash="dot")))
                    else:
                        fig.add_trace(go.Scatter(y=y_orig, mode="lines", name=f"{yname} (originale)", line=dict(width=1, dash="dot")))
                    fig.data = fig.data[::-1]
                save_plot_html(fig, base=f"plot_{yname}")

        # FFT (per serie, file separati anche in modalità sovrapposti)
        if self.fft_on_var.get():
            for (yname, x_used, y_plot, y_orig) in series_list:
                y_fft = y_plot if (fspec.enabled and (self.fft_what_combo.get() == "filtered")) else (y_orig if (y_orig is not None and self.fft_what_combo.get()=="original") else y_plot)
                if not fs_value or fs_value <= 0:
                    self._append_log(f"FFT non calcolata per {yname}: fs non disponibile.")
                else:
                    freqs, amp = compute_fft(y_fft, fs_value, detrend=self.detrend_var.get())
                    if freqs.size == 0:
                        self._append_log(f"FFT non calcolabile per {yname} (serie troppo corta o parametri non validi).")
                    else:
                        save_plot_html(plot_fft(freqs, amp, title=f"FFT — {yname}"), base=f"fft_{yname}")

        self._append_log("Plot completati. File in outputs/ (aperti nel browser).")

    def _report(self) -> None:
        if self.df is None or self.df.empty:
            messagebox.showwarning("Attenzione", "Carica prima un CSV.")
            return
        sel_idx = list(self.y_list.curselection())
        if not sel_idx:
            messagebox.showwarning("Attenzione", "Seleziona almeno una Y dall’elenco.")
            return
        y_names = [self.columns[i] for i in sel_idx]
        try:
            out = ReportManager().generate_report(self.df, None, y_names, formats="csv")
            self._append_log("Report generato: " + str({k: str(v) for k, v in out.items() if v}))
        except Exception as e:
            messagebox.showerror("Errore", f"Generazione report fallita: {e}")

if __name__ == "__main__":
    app = DesktopAppTk()
    app.mainloop()
