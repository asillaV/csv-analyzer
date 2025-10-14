from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal, List, Dict

import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, filtfilt, get_window
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


# -----------------------------
# Sampling analysis config
# -----------------------------
IRREGULAR_RATIO_THRESHOLD: float = 1.5  # max(max_dt / min_dt) prima del warning
IRREGULAR_STD_THRESHOLD: float = 0.2   # std_rel = std(dt) / median(dt)


# -----------------------------
# Sampling info dataclass
# -----------------------------
@dataclass
class FsInfo:
    value: Optional[float]
    source: Literal["manual", "datetime", "index", "none"]
    unit: Literal["seconds", "index", "manual", "unknown"] = "unknown"
    is_uniform: bool = True
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, float] = field(default_factory=dict)

    def __iter__(self):
        # Consente ancora: fs, source = resolve_fs(...)
        yield self.value
        yield self.source


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
    Mantiene compatibilità con l'API precedente restituendo solo il valore di fs.
    Per ottenere warning e dettagli usare resolve_fs().
    """
    info = resolve_fs(x, manual_fs=None)
    return info.value


def resolve_fs(
    x_values: pd.Index | pd.Series | None,
    manual_fs: Optional[float],
) -> FsInfo:
    """
    Determina la frequenza di campionamento applicando la priorità:
      1. manual_fs positivo (source="manual")
      2. stima automatica su datetime -> source="datetime"
      3. stima su indice numerico -> source="index"
      4. assenza di stima -> source="none"
    Restituisce un FsInfo che include warning e metriche di irregolarità.
    """
    if manual_fs is not None:
        try:
            manual_val = float(manual_fs)
        except (TypeError, ValueError):
            manual_val = None
        if manual_val is not None and manual_val > 0:
            return FsInfo(
                value=float(manual_val),
                source="manual",
                unit="manual",
                is_uniform=True,
                warnings=[],
                details={"manual_fs": float(manual_val)},
            )

    return _analyze_sampling(x_values)


def _analyze_sampling(x: pd.Index | pd.Series | None) -> FsInfo:
    if x is None:
        return FsInfo(
            value=None,
            source="none",
            unit="unknown",
            is_uniform=False,
            warnings=["Nessun asse X disponibile per stimare la frequenza di campionamento."],
        )

    if isinstance(x, pd.Series):
        series = x.dropna()
    else:
        try:
            series = pd.Series(x.dropna())
        except Exception:
            series = pd.Series([])

    if len(series) < 2:
        return FsInfo(
            value=None,
            source="none",
            unit="unknown",
            is_uniform=False,
            warnings=["Campioni X insufficienti per stimare la frequenza di campionamento."],
        )

    is_datetime = _is_datetime_like(series)

    try:
        if is_datetime:
            values = pd.to_datetime(series).astype("int64") / 1e9  # ns -> s
            unit = "seconds"
            source = "datetime"
        else:
            values = pd.to_numeric(series, errors="coerce").astype(float)
            values = values[np.isfinite(values)]
            unit = "index"
            source = "index"
    except Exception:
        return FsInfo(
            value=None,
            source="none",
            unit="unknown",
            is_uniform=False,
            warnings=["Impossibile interpretare i valori X per la stima di fs."],
        )

    if len(values) < 2:
        return FsInfo(
            value=None,
            source="none",
            unit="unknown",
            is_uniform=False,
            warnings=["Campioni X insufficienti per stimare la frequenza di campionamento."],
        )

    dx = np.diff(values)
    dx = dx[np.isfinite(dx)]
    dx = dx[dx > 0]

    if dx.size == 0:
        return FsInfo(
            value=None,
            source="none",
            unit="unknown",
            is_uniform=False,
            warnings=["Differenze non positive sull'asse X; campionamento non valido."],
        )

    med = float(np.median(dx))
    if med <= 0:
        return FsInfo(
            value=None,
            source="none",
            unit="unknown",
            is_uniform=False,
            warnings=["Mediana delle differenze nulla o negativa; campionamento non valido."],
        )

    fs_value = float(1.0 / med)
    min_dt = float(np.min(dx))
    max_dt = float(np.max(dx))
    std_dt = float(np.std(dx))
    rel_std = std_dt / med if med > 0 else 0.0
    ratio = (max_dt / min_dt) if min_dt > 0 else float("inf")

    warnings: List[str] = []
    is_uniform = True
    if ratio > IRREGULAR_RATIO_THRESHOLD or rel_std > IRREGULAR_STD_THRESHOLD:
        is_uniform = False
        parts = []
        if ratio > IRREGULAR_RATIO_THRESHOLD:
            parts.append(f"Δt max/min = {ratio:.2f}")
        if rel_std > IRREGULAR_STD_THRESHOLD:
            parts.append(f"std_rel = {rel_std:.2f}")
        warnings.append("Campionamento irregolare: " + ", ".join(parts))

    return FsInfo(
        value=fs_value,
        source=source,
        unit=unit,
        is_uniform=is_uniform,
        warnings=warnings,
        details={
            "median_dt": med,
            "min_dt": min_dt,
            "max_dt": max_dt,
            "std_dt": std_dt,
            "rel_std": rel_std,
            "ratio_max_min": ratio,
        },
    )


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

    fs_info = resolve_fs(x_values, fs_override)
    fs = fs_info.value

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
