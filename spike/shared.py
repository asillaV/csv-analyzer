"""
Spike — logica condivisa headless (framework-agnostica).

Scopo: dimostrare che il dominio in `core/` gira FUORI da Streamlit e da
qualsiasi UI (nessun `webbrowser.open`, nessuno `st.*`), producendo le due
figure Plotly che servono alla schermata di prova: serie temporale (raw +
filtrata) e spettro FFT.

Questo modulo è importato sia da `nicegui_app.py` sia dall'app Reflex, così la
stessa "ricetta" di orchestrazione del core è riusata da entrambe le UI — è
esattamente il pattern che salveremmo da `web_app.py` nella migrazione.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Rende importabile il package `core` senza installarlo (lo spike non tocca il
# layout attuale; l'estrazione a `csv_core` è il passo 2 del piano).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import plotly.graph_objects as go  # noqa: E402

from core.loader_optimized import load_csv  # noqa: E402
from core.signal_tools import (  # noqa: E402
    FilterSpec,
    resolve_fs,
    apply_filter,
    compute_fft,
    validate_filter_spec,
)

DEFAULT_CSV = _PROJECT_ROOT / "assets" / "sample_timeseries.csv"

# Tipi di filtro esposti nello spike (sottoinsieme dimostrativo).
FILTER_CHOICES = ["none", "ma", "butter_lp"]


@dataclass
class AnalysisResult:
    time_fig: go.Figure
    fft_fig: go.Figure
    info: str           # riepilogo testuale (fs, sorgente, filtro, righe)
    warning: str = ""   # eventuale messaggio non bloccante (es. validazione filtro)


def list_columns(path: str | Path) -> list[str]:
    """Colonne disponibili nel CSV (per popolare le select delle UI)."""
    df = load_csv(str(path))
    return list(df.columns)


def _build_filter_spec(kind: str, ma_window: int, cutoff_hz: float) -> FilterSpec:
    if kind == "ma":
        return FilterSpec(kind="ma", enabled=True, ma_window=int(ma_window))
    if kind == "butter_lp":
        return FilterSpec(kind="butter_lp", enabled=True, order=4,
                          cutoff=(float(cutoff_hz), None))
    return FilterSpec(kind="ma", enabled=False)  # "none"


def analyze(
    path: str | Path,
    x_col: Optional[str],
    y_col: str,
    filter_kind: str = "none",
    ma_window: int = 9,
    cutoff_hz: float = 10.0,
    manual_fs: float = 0.0,
) -> AnalysisResult:
    """Carica → risolve fs → filtra → FFT, e restituisce due figure Plotly.

    Tutto headless: nessun side-effect su filesystem condiviso, nessun browser.
    """
    df = load_csv(str(path))
    if y_col not in df.columns:
        raise ValueError(f"Colonna '{y_col}' assente. Disponibili: {list(df.columns)}")

    x = df[x_col] if x_col and x_col in df.columns else None
    y = df[y_col]

    # Fonte di verità unica per fs (come da design del progetto).
    fs_info = resolve_fs(x, manual_fs if manual_fs and manual_fs > 0 else None)

    spec = _build_filter_spec(filter_kind, ma_window, cutoff_hz)
    warning = ""
    y_used = y
    if spec.enabled:
        ok, msg = validate_filter_spec(spec, fs_info.value)
        if ok:
            y_used, _ = apply_filter(y, x, spec, fs_override=fs_info.value)
        else:
            warning = f"Filtro non applicato: {msg}"

    # ---- Figura dominio del tempo ----
    x_plot = x if x is not None else df.index
    time_fig = go.Figure()
    time_fig.add_trace(go.Scatter(x=x_plot, y=y, mode="lines",
                                  name=f"{y_col} (raw)", line=dict(width=1)))
    if spec.enabled and not warning:
        time_fig.add_trace(go.Scatter(x=x_plot, y=y_used, mode="lines",
                                      name=f"{y_col} ({filter_kind})",
                                      line=dict(width=2)))
    time_fig.update_layout(title="Dominio del tempo", template="plotly_white",
                           xaxis_title=(x_col or "indice"), yaxis_title=y_col,
                           margin=dict(l=40, r=20, t=40, b=40))

    # ---- Figura FFT ----
    freqs, amp = compute_fft(y_used, fs_info.value)
    fft_fig = go.Figure()
    if len(freqs):
        fft_fig.add_trace(go.Scatter(x=freqs, y=amp, mode="lines", name="FFT"))
    fft_fig.update_layout(title="Spettro FFT", template="plotly_white",
                          xaxis_title="Frequenza [Hz]", yaxis_title="Ampiezza",
                          margin=dict(l=40, r=20, t=40, b=40))

    fs_txt = f"{fs_info.value:.4g} Hz" if fs_info.value else "n/d"
    info = (f"Righe: {len(df):,} | fs: {fs_txt} (sorgente: {fs_info.source}) | "
            f"filtro: {filter_kind} | punti FFT: {len(freqs)}")
    return AnalysisResult(time_fig=time_fig, fft_fig=fft_fig, info=info, warning=warning)


def write_temp_upload(data: bytes, suffix: str = ".csv") -> str:
    """Scrive bytes caricati su file temporaneo e ritorna il path (per load_csv)."""
    fd, name = tempfile.mkstemp(prefix="spike_upload_", suffix=suffix)
    with open(fd, "wb") as fh:
        fh.write(data)
    return name


if __name__ == "__main__":
    # Smoke test headless: prova che il core gira senza UI.
    cols = list_columns(DEFAULT_CSV)
    res = analyze(DEFAULT_CSV, x_col=cols[0], y_col=cols[3],
                  filter_kind="butter_lp", cutoff_hz=5.0)
    print("Colonne:", cols)
    print(res.info)
    print("Time traces:", len(res.time_fig.data), "| FFT traces:", len(res.fft_fig.data))
    if res.warning:
        print("Warning:", res.warning)
