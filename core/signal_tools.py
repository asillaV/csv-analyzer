from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, filtfilt, get_window
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


# -----------------------------
# Spec filtri / FFT
# -----------------------------
@dataclass
class FilterSpec:
    # kind: "ma" | "butter_lp" | "butter_hp" | "butter_bp"
    kind: str
    enabled: bool = False
    order: int = 4
    cutoff: Optional[Tuple[Optional[float], Optional[float]]] = None  # (low, high) per BP; (fc, None) per LP/HP
    ma_window: int = 5  # per media mobile


@dataclass
class FFTSpec:
    enabled: bool = False
    detrend: bool = True
    window: str = "hann"  # se SciPy manca, ricade su finestra uniforme


# -----------------------------
# Utilità campionamento (fs)
# -----------------------------
def _is_datetime_like(x: pd.Index | pd.Series) -> bool:
    if isinstance(x, pd.Series):
        return pd.api.types.is_datetime64_any_dtype(x) or pd.api.types.is_timedelta64_dtype(x)
    if isinstance(x, pd.Index):
        return pd.api.types.is_datetime64_any_dtype(x) or pd.api.types.is_timedelta64_dtype(x)
    return False


def estimate_fs(x: pd.Index | pd.Series | None) -> Optional[float]:
    """
    Stima fs dai valori X:
      - Datetime/Timedelta: differenze in secondi -> fs = 1/mediana(dt)
      - Numerico: differenze dirette -> fs = 1/mediana(dx)
    Restituisce None se non stimabile o se i delta non sono positivi.
    """
    if x is None:
        return None

    if isinstance(x, pd.Series):
        xv = x.dropna().to_numpy()
    else:
        xv = x.dropna().values

    if len(xv) < 2:
        return None

    if _is_datetime_like(x):
        xv = pd.to_datetime(xv).astype("int64") / 1e9  # ns -> s
    else:
        try:
            xv = pd.to_numeric(xv, errors="coerce").astype(float)
        except Exception:
            return None

    dx = np.diff(xv)
    dx = dx[np.isfinite(dx)]
    dx = dx[dx > 0]
    if dx.size == 0:
        return None
    med = float(np.median(dx))
    if med <= 0:
        return None
    return float(1.0 / med)


def resolve_fs(x_values: pd.Index | pd.Series | None,
               manual_fs: Optional[float]) -> tuple[Optional[float], Literal["manual", "estimated", "none"]]:
    """
    Ritorna (fs, source): fs>0 se disponibile, altrimenti None; source ∈ {"manual","estimated","none"}.
    Regole:
      - manual_fs > 0 -> ("manual")
      - altrimenti prova estimate_fs(x_values) -> ("estimated")
      - altrimenti -> (None, "none")
    """
    if manual_fs is not None:
        try:
            if float(manual_fs) > 0:
                return float(manual_fs), "manual"
        except Exception:
            pass
    fs = estimate_fs(x_values)
    if fs is not None and fs > 0:
        return float(fs), "estimated"
    return None, "none"


# -----------------------------
# Validazione filtri
# -----------------------------
def validate_filter_spec(spec: FilterSpec, fs: Optional[float]) -> tuple[bool, str]:
    """
    (ok, message)
      - spec.enabled False -> ok True
      - kind "ma" -> ok se finestra >= 1
      - Butterworth:
          * SciPy deve esserci
          * fs > 0
          * order >= 1
          * cutoff presenti/positivi
          * LP/HP: 0 < fc < fs/2
          * BP: 0 < flo < fhi < fs/2
    """
    if not spec.enabled:
        return True, "Filtro disabilitato."
    if spec.kind == "ma":
        if int(spec.ma_window) < 1:
            return False, "Finestra MA non valida (>=1)."
        return True, "MA ok."

    # Butterworth
    if not _SCIPY_OK:
        return False, "SciPy non disponibile: filtri Butterworth non utilizzabili."
    if fs is None or fs <= 0:
        return False, "fs richiesta (>0) per filtri Butterworth."
    if int(spec.order) < 1:
        return False, "Ordine Butterworth deve essere >=1."
    if spec.cutoff is None:
        return False, "Frequenza/e di taglio mancanti."

    flo, fhi = spec.cutoff
    nyq = 0.5 * fs

    if spec.kind in ("butter_lp", "butter_hp"):
        fc = flo
        if fc is None or fc <= 0:
            return False, "Cutoff non valida (deve essere >0)."
        if fc >= nyq:
            return False, f"Cutoff {fc} Hz ≥ Nyquist {nyq:.6g} Hz."
        return True, "Butterworth LP/HP ok."

    if spec.kind == "butter_bp":
        if flo is None or fhi is None or flo <= 0 or fhi <= 0:
            return False, "Cutoff banda non valide (flo/hi > 0)."
        if not (flo < fhi):
            return False, "Cutoff banda non valide (hi deve essere > lo)."
        if fhi >= nyq:
            return False, f"Cutoff alta {fhi} Hz ≥ Nyquist {nyq:.6g} Hz."
        return True, "Butterworth BP ok."

    return True, "Nessun filtro applicato."


# -----------------------------
# Filtri
# -----------------------------
def moving_average(y: pd.Series, window: int) -> pd.Series:
    w = max(1, int(window))
    if w == 1:
        return y.copy()
    return y.rolling(window=w, min_periods=1, center=False).mean()


def butter_filter(y: pd.Series, fs: float, spec: FilterSpec) -> pd.Series:
    ok, msg = validate_filter_spec(spec, fs)
    if not ok:
        raise ValueError(msg)
    if spec.kind not in ("butter_lp", "butter_hp", "butter_bp"):
        return y.copy()

    order = max(1, int(spec.order))
    nyq = 0.5 * fs

    if spec.kind == "butter_lp":
        wn = float(spec.cutoff[0]) / nyq  # type: ignore[index]
        btype = "lowpass"
        WN = wn
    elif spec.kind == "butter_hp":
        wn = float(spec.cutoff[0]) / nyq  # type: ignore[index]
        btype = "highpass"
        WN = wn
    else:  # butter_bp
        lo = float(spec.cutoff[0]) / nyq  # type: ignore[index]
        hi = float(spec.cutoff[1]) / nyq  # type: ignore[index]
        WN = (lo, hi)
        btype = "bandpass"

    b, a = butter(order, WN, btype=btype)
    z = filtfilt(b, a, y.to_numpy(dtype=float))
    return pd.Series(z, index=y.index, name=y.name)


def apply_filter(
    series: pd.Series,
    x_values: Optional[pd.Series | pd.Index],
    spec: FilterSpec,
    fs_override: Optional[float] = None,
) -> tuple[pd.Series, Optional[float]]:
    """
    Applica il filtro:
      - se spec.disabled -> ritorna serie originale
      - MA: non richiede fs
      - Butterworth: usa fs_override se presente, altrimenti prova a stimare da x_values
    Ritorna (serie_filtrata, fs_usata) – fs_usata None se MA o se filtro disabilitato.
    Lancia ValueError con motivo umano in caso di input non validi (da intercettare in UI).
    """
    if not spec.enabled:
        return series.copy(), None
    if spec.kind == "ma":
        return moving_average(series, spec.ma_window), None

    fs = fs_override
    if fs is None:
        fs = estimate_fs(x_values) if x_values is not None else None

    ok, msg = validate_filter_spec(spec, fs)
    if not ok:
        raise ValueError(msg)

    y_f = butter_filter(series, float(fs), spec)  # type: ignore[arg-type]
    return y_f, float(fs)


# -----------------------------
# FFT
# -----------------------------
def compute_fft(
    y: pd.Series,
    fs: Optional[float],
    detrend: bool = True,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ritorna (freqs, amp). Se input non valido, ritorna vettori vuoti.
    """
    y = pd.to_numeric(y, errors="coerce").dropna()
    n = len(y)
    if n < 4 or fs is None or fs <= 0:
        return np.array([]), np.array([])
    y_np = y.to_numpy(dtype=float)
    if detrend:
        y_np = y_np - float(np.mean(y_np))

    if _SCIPY_OK:
        try:
            win = get_window(window, n, fftbins=True)
        except Exception:
            win = np.ones(n)
    else:
        win = np.ones(n)

    yw = y_np * win
    Y = np.fft.rfft(yw)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    amp = (2.0 / np.sum(win)) * np.abs(Y)
    return freqs, amp
