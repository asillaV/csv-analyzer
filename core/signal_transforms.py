"""
Trasformazioni non distruttive su segnali (pd.Series).
Segue il pattern Spec → Validate → Apply di signal_tools.py.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from core.signal_tools import FsInfo

try:
    from scipy.interpolate import interp1d as _scipy_interp1d
    _SCIPY_INTERP_OK = True
except Exception:
    _SCIPY_INTERP_OK = False

try:
    from scipy.integrate import cumulative_trapezoid as _scipy_cumtrapz
    _SCIPY_INTEG_OK = True
except Exception:
    _SCIPY_INTEG_OK = False


TransformKind = Literal[
    "offset", "scale", "minmax_norm", "zscore_norm",
    "shift_samples", "shift_time",
    "resample", "derivative", "integral",
]

TRANSFORM_LABELS: Dict[str, str] = {
    "offset": "Offset",
    "scale": "Scala (gain)",
    "minmax_norm": "Norm. Min-Max [0,1]",
    "zscore_norm": "Norm. Z-score",
    "shift_samples": "Shift campioni",
    "shift_time": "Shift temporale",
    "resample": "Ricampionamento",
    "derivative": "Derivata (dy/dx)",
    "integral": "Integrale (∫y dx)",
}

TRANSFORM_GUIDE: Dict[str, Dict[str, str]] = {
    "offset":        {"formula": "y = y + c",          "note": "Sposta il segnale verticalmente.",           "x": "No",  "len": "No"},
    "scale":         {"formula": "y = y × c",          "note": "Cambia ampiezza / unità di misura.",         "x": "No",  "len": "No"},
    "minmax_norm":   {"formula": "y → [0, 1]",         "note": "Normalizza tra 0 e 1.",                      "x": "No",  "len": "No"},
    "zscore_norm":   {"formula": "y → μ=0, σ=1",       "note": "Centra e scala a deviazione standard 1.",    "x": "No",  "len": "No"},
    "shift_samples": {"formula": "shift(y, N)",         "note": "Ritarda (N>0) o anticipa (N<0) il segnale.", "x": "No",  "len": "No"},
    "shift_time":    {"formula": "x = x + Δt",         "note": "Sposta l'asse temporale.",                   "x": "Sì",  "len": "No"},
    "resample":      {"formula": "interp(y, x, fs_new)","note": "Ricampiona a nuova frequenza.",              "x": "Sì",  "len": "Sì"},
    "derivative":    {"formula": "dy/dx (N−1 camp.)",  "note": "Velocità / tasso di variazione.",            "x": "Sì",  "len": "Sì"},
    "integral":      {"formula": "∫y dx (trapezi)",    "note": "Area sotto la curva cumulativa.",            "x": "Sì",  "len": "No"},
}

_X_REQUIRED: frozenset = frozenset({"shift_time", "resample", "derivative", "integral"})
_CHANGES_LENGTH: frozenset = frozenset({"derivative", "resample"})


@dataclass
class TransformSpec:
    """Parameters for a single signal transformation step."""
    kind: TransformKind
    enabled: bool = True
    constant: float = 0.0           # additive (offset) or multiplicative (scale) value
    shift_samples: int = 0           # for shift_samples kind
    shift_time: float = 0.0          # for shift_time kind (X-axis units; seconds for datetime)
    target_fs: Optional[float] = None  # for resample kind (Hz)
    interp_method: str = "linear"    # for resample: "linear" | "cubic" | "nearest"
    per_column: bool = False         # use per_column_overrides[col] instead of constant
    per_column_overrides: Dict[str, float] = field(default_factory=dict)


@dataclass
class TransformResult:
    """Result of applying one transformation."""
    y: pd.Series
    x: Optional[pd.Series]
    kind: TransformKind
    description: str
    changed_length: bool = False     # True for derivative (N-1) and resample (N')
    fs_out: Optional[float] = None   # updated fs after resampling


def transform_spec_cache_key(spec: TransformSpec) -> str:
    """Return a JSON string usable as a hashable dict key (handles the dict field)."""
    return json.dumps(asdict(spec), sort_keys=True)


def validate_transform_spec(
    spec: TransformSpec,
    fs_info: Optional[FsInfo] = None,
    x_available: bool = True,
) -> Tuple[bool, str]:
    """Return (ok, message). Always ok=True when spec.enabled is False."""
    if not spec.enabled:
        return True, "Trasformazione disabilitata."

    if spec.kind in _X_REQUIRED and not x_available:
        names = {
            "shift_time": "Shift temporale",
            "resample": "Ricampionamento",
            "derivative": "Derivata",
            "integral": "Integrale",
        }
        return False, f"{names.get(spec.kind, spec.kind)} richiede un asse X."

    if spec.kind == "offset":
        return True, "Offset ok."

    if spec.kind == "scale":
        return True, "Scala ok."

    if spec.kind in ("minmax_norm", "zscore_norm"):
        return True, f"{TRANSFORM_LABELS[spec.kind]} ok."

    if spec.kind == "shift_samples":
        return True, "Shift campioni ok."

    if spec.kind == "shift_time":
        return True, "Shift temporale ok."

    if spec.kind == "resample":
        if spec.target_fs is None or spec.target_fs <= 0:
            return False, "Ricampionamento: target_fs deve essere > 0 Hz."
        if spec.interp_method not in ("linear", "cubic", "nearest"):
            return False, f"Metodo di interpolazione non valido: '{spec.interp_method}'."
        if spec.interp_method in ("cubic", "nearest") and not _SCIPY_INTERP_OK:
            return False, "SciPy non disponibile: usa il metodo 'linear'."
        fs_val = fs_info.value if fs_info and fs_info.value else None
        if fs_val and spec.target_fs > fs_val / 2:
            return True, (
                f"Avviso: target_fs ({spec.target_fs} Hz) > fs/2 "
                f"({fs_val / 2:.3g} Hz) — possibile aliasing."
            )
        return True, "Ricampionamento ok."

    if spec.kind == "derivative":
        return True, "Derivata ok."

    if spec.kind == "integral":
        if not _SCIPY_INTEG_OK:
            return False, "SciPy non disponibile: integrale non calcolabile."
        return True, "Integrale ok."

    return True, "Trasformazione ok."


def apply_transform(
    y: pd.Series,
    x: Optional[pd.Series],
    spec: TransformSpec,
    fs_info: Optional[FsInfo] = None,
    col_name: Optional[str] = None,
) -> TransformResult:
    """Apply a single transformation. Raises ValueError for invalid specs or data."""
    if not spec.enabled:
        return TransformResult(
            y=y.copy(),
            x=x.copy() if x is not None else None,
            kind=spec.kind,
            description="(disabilitata)",
        )

    x_available = x is not None and len(x) > 0
    ok, msg = validate_transform_spec(spec, fs_info, x_available)
    if not ok:
        raise ValueError(msg)

    if spec.kind == "offset":
        return _apply_offset(y, x, spec, col_name)
    if spec.kind == "scale":
        return _apply_scale(y, x, spec, col_name)
    if spec.kind == "minmax_norm":
        return _apply_minmax_norm(y, x)
    if spec.kind == "zscore_norm":
        return _apply_zscore_norm(y, x)
    if spec.kind == "shift_samples":
        return _apply_shift_samples(y, x, spec)
    if spec.kind == "shift_time":
        return _apply_shift_time(y, x, spec)
    if spec.kind == "resample":
        return _apply_resample(y, x, spec)
    if spec.kind == "derivative":
        return _apply_derivative(y, x)
    if spec.kind == "integral":
        return _apply_integral(y, x)

    return TransformResult(
        y=y.copy(),
        x=x.copy() if x is not None else None,
        kind=spec.kind,
        description="kind sconosciuto",
    )


def apply_transform_pipeline(
    y: pd.Series,
    x: Optional[pd.Series],
    specs: List[TransformSpec],
    fs_info: Optional[FsInfo] = None,
    col_name: Optional[str] = None,
) -> Tuple[pd.Series, Optional[pd.Series], List[str], bool]:
    """
    Apply a list of TransformSpecs in order.
    Returns (y_out, x_out, descriptions, changed_length).
    Raises ValueError if any enabled step fails validation.
    """
    descriptions: List[str] = []
    y_cur = y
    x_cur = x
    changed_length = False

    for spec in specs:
        if not spec.enabled:
            continue
        result = apply_transform(y_cur, x_cur, spec, fs_info, col_name)
        y_cur = result.y
        x_cur = result.x
        if result.changed_length:
            changed_length = True
        descriptions.append(result.description)

    return y_cur, x_cur, descriptions, changed_length


# ---- Private helpers ----

def _get_constant(spec: TransformSpec, col_name: Optional[str]) -> float:
    if spec.per_column and col_name is not None:
        return float(spec.per_column_overrides.get(col_name, spec.constant))
    return float(spec.constant)


def _apply_offset(
    y: pd.Series, x: Optional[pd.Series], spec: TransformSpec, col_name: Optional[str]
) -> TransformResult:
    c = _get_constant(spec, col_name)
    return TransformResult(
        y=pd.Series(y.to_numpy(dtype=float) + c, index=y.index, name=y.name),
        x=x.copy() if x is not None else None,
        kind="offset",
        description=f"offset {c:+g}",
    )


def _apply_scale(
    y: pd.Series, x: Optional[pd.Series], spec: TransformSpec, col_name: Optional[str]
) -> TransformResult:
    c = _get_constant(spec, col_name)
    return TransformResult(
        y=pd.Series(y.to_numpy(dtype=float) * c, index=y.index, name=y.name),
        x=x.copy() if x is not None else None,
        kind="scale",
        description=f"scala ×{c:g}",
    )


def _apply_minmax_norm(y: pd.Series, x: Optional[pd.Series]) -> TransformResult:
    y_num = pd.to_numeric(y, errors="coerce").dropna()
    y_min = float(y_num.min())
    y_max = float(y_num.max())
    if y_max == y_min:
        raise ValueError("Normalizzazione Min-Max: segnale costante (max == min).")
    result = (y.to_numpy(dtype=float) - y_min) / (y_max - y_min)
    return TransformResult(
        y=pd.Series(result, index=y.index, name=y.name),
        x=x.copy() if x is not None else None,
        kind="minmax_norm",
        description=f"norm. min-max [{y_min:g}, {y_max:g}] → [0, 1]",
    )


def _apply_zscore_norm(y: pd.Series, x: Optional[pd.Series]) -> TransformResult:
    y_num = pd.to_numeric(y, errors="coerce").dropna()
    mean = float(y_num.mean())
    std = float(y_num.std())
    if std == 0.0:
        raise ValueError("Normalizzazione Z-score: deviazione standard nulla (segnale costante).")
    result = (y.to_numpy(dtype=float) - mean) / std
    return TransformResult(
        y=pd.Series(result, index=y.index, name=y.name),
        x=x.copy() if x is not None else None,
        kind="zscore_norm",
        description=f"z-score (µ={mean:g}, σ={std:g})",
    )


def _apply_shift_samples(
    y: pd.Series, x: Optional[pd.Series], spec: TransformSpec
) -> TransformResult:
    n = int(spec.shift_samples)
    shifted = y.shift(n).to_numpy(dtype=float)
    return TransformResult(
        y=pd.Series(shifted, index=y.index, name=y.name),
        x=x.copy() if x is not None else None,
        kind="shift_samples",
        description=f"shift {n:+d} campioni",
    )


def _apply_shift_time(
    y: pd.Series, x: Optional[pd.Series], spec: TransformSpec
) -> TransformResult:
    dt = spec.shift_time
    if pd.api.types.is_datetime64_any_dtype(x):
        x_shifted = x + pd.Timedelta(seconds=dt)
    else:
        x_shifted = x + dt
    return TransformResult(
        y=y.copy(),
        x=pd.Series(x_shifted.to_numpy(), index=x.index, name=x.name),
        kind="shift_time",
        description=f"shift X {dt:+g}",
    )


def _apply_resample(
    y: pd.Series, x: Optional[pd.Series], spec: TransformSpec
) -> TransformResult:
    x_np = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_np = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(x_np) & np.isfinite(y_np)
    x_v = x_np[valid]
    y_v = y_np[valid]

    if len(x_v) < 2:
        raise ValueError("Ricampionamento: dati insufficienti (< 2 campioni validi).")

    x_start, x_end = float(x_v[0]), float(x_v[-1])
    if x_end <= x_start:
        raise ValueError("Ricampionamento: asse X non monotono.")

    n_new = max(2, int(round((x_end - x_start) * spec.target_fs)))
    x_new = np.linspace(x_start, x_end, n_new)

    if spec.interp_method == "linear" or not _SCIPY_INTERP_OK:
        y_new = np.interp(x_new, x_v, y_v)
    else:
        f_interp = _scipy_interp1d(
            x_v, y_v,
            kind=spec.interp_method,
            bounds_error=False,
            fill_value=(float(y_v[0]), float(y_v[-1])),
        )
        y_new = f_interp(x_new)

    x_name = x.name if x is not None else "x"
    return TransformResult(
        y=pd.Series(y_new, name=y.name),
        x=pd.Series(x_new, name=x_name),
        kind="resample",
        description=f"ricampionato a {spec.target_fs:g} Hz ({len(y_v)} → {n_new} campioni)",
        changed_length=True,
        fs_out=float(spec.target_fs),
    )


def _apply_derivative(y: pd.Series, x: Optional[pd.Series]) -> TransformResult:
    x_np = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_np = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)

    if len(y_np) < 2:
        raise ValueError("Derivata: segnale troppo corto (< 2 campioni).")

    dy = np.diff(y_np)
    dx = np.diff(x_np)
    dx_safe = np.where(dx == 0, np.nan, dx)
    deriv = dy / dx_safe
    x_mid = (x_np[:-1] + x_np[1:]) / 2.0

    x_name = x.name if x is not None else "x"
    return TransformResult(
        y=pd.Series(deriv, name=y.name),
        x=pd.Series(x_mid, name=x_name),
        kind="derivative",
        description=f"derivata dy/dx ({len(y_np)} → {len(deriv)} campioni)",
        changed_length=True,
    )


def _apply_integral(y: pd.Series, x: Optional[pd.Series]) -> TransformResult:
    x_np = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_np = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)

    if len(y_np) < 2:
        raise ValueError("Integrale: segnale troppo corto (< 2 campioni).")

    integral = _scipy_cumtrapz(y_np, x_np, initial=0)

    return TransformResult(
        y=pd.Series(integral, index=y.index, name=y.name),
        x=x.copy() if x is not None else None,
        kind="integral",
        description="integrale ∫y dx (trapezi)",
    )
