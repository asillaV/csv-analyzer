from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Cache module-level per LTTB: sopravvive a tutti i rerender Streamlit nello stesso processo.
_LTTB_CACHE: Dict[Tuple, np.ndarray] = {}
_LTTB_CACHE_MAX = 64

DownsampleMethod = Literal["lttb", "minmax", "identity"]


@dataclass(slots=True)
class DownsampleResult:
    y: pd.Series
    x: Optional[pd.Series]
    indices: np.ndarray
    method: DownsampleMethod
    original_count: int
    sampled_count: int

    @property
    def reduction_ratio(self) -> float:
        if self.sampled_count == 0:
            return np.inf
        return self.original_count / self.sampled_count

    def summary(self) -> str:
        return (
            f"{self.original_count:,} → {self.sampled_count:,} "
            f"({self.reduction_ratio:.1f}x, {self.method})"
        )


def _as_numeric_x(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy()
    if np.issubdtype(arr.dtype, np.datetime64) or np.issubdtype(arr.dtype, np.timedelta64):
        return arr.astype("int64").astype("float64")
    try:
        return arr.astype("float64")
    except (TypeError, ValueError):
        cast = pd.to_numeric(values, errors="coerce")
        return cast.to_numpy(dtype=np.float64)


def _lttb_fingerprint(x: np.ndarray, y: np.ndarray) -> str:
    """Fingerprint veloce basato su campionamento di 100 punti. Collision rate trascurabile."""
    n = len(y)
    idx = np.linspace(0, n - 1, min(100, n), dtype=int)
    fp = np.concatenate([[n, x[0], x[-1], y[0], y[-1]], x[idx], y[idx]])
    return hashlib.md5(fp.tobytes()).hexdigest()


def _lttb_indices_cached(x: np.ndarray, y: np.ndarray, target_count: int) -> np.ndarray:
    """Wrapper cachato di _lttb_indices: evita ricalcolo O(n×target) su dati invariati."""
    key = (_lttb_fingerprint(x, y), target_count)
    if key in _LTTB_CACHE:
        return _LTTB_CACHE[key]
    result = _lttb_indices(x, y, target_count)
    if len(_LTTB_CACHE) >= _LTTB_CACHE_MAX:
        del _LTTB_CACHE[next(iter(_LTTB_CACHE))]
    _LTTB_CACHE[key] = result
    return result


def _lttb_indices(x: np.ndarray, y: np.ndarray, target_count: int) -> np.ndarray:
    n = y.shape[0]
    if target_count >= n or target_count < 3:
        return np.arange(n, dtype=int)

    result = np.empty(target_count, dtype=int)
    result[0] = 0
    result[-1] = n - 1

    bucket = (n - 2) / (target_count - 2)
    a = 0
    for i in range(1, target_count - 1):
        start = int(np.floor((i - 1) * bucket)) + 1
        end = int(np.floor(i * bucket)) + 1
        if end >= n:
            end = n - 1
        if end <= start:
            end = min(start + 1, n - 1)

        bucket_x = x[start:end]
        bucket_y = y[start:end]
        if bucket_x.size == 0:
            bucket_x = x[start : end + 1]
            bucket_y = y[start : end + 1]

        avg_start = int(np.floor(i * bucket)) + 1
        avg_end = int(np.floor((i + 1) * bucket)) + 1
        if avg_start >= n:
            avg_start = n - 1
        if avg_end <= avg_start:
            avg_end = min(avg_start + 1, n)

        avg_x = np.mean(x[avg_start:avg_end])
        avg_y = np.mean(y[avg_start:avg_end])

        ax = x[a]
        ay = y[a]
        dx = bucket_x - ax
        dy = bucket_y - ay
        area = np.abs(dx * (avg_y - ay) - (bucket_x - avg_x) * dy)
        chosen = start + int(np.argmax(area))
        result[i] = min(max(chosen, start), end - 1)
        a = result[i]

    return np.unique(result)


def _minmax_indices(y: np.ndarray, target_count: int) -> np.ndarray:
    n = y.shape[0]
    if target_count >= n or target_count < 3:
        return np.arange(n, dtype=int)

    step = max(1, int(np.floor(n / target_count)))
    picks = {0, n - 1}
    for start in range(0, n, step):
        end = min(start + step, n)
        if end - start <= 1:
            picks.add(start)
            continue
        segment = y[start:end]
        picks.add(start + int(np.argmin(segment)))
        picks.add(start + int(np.argmax(segment)))

    ordered = np.array(sorted(picks))
    if ordered.size <= target_count:
        return ordered
    idx = np.linspace(0, ordered.size - 1, num=target_count, dtype=int)
    return np.unique(ordered[idx])


def downsample_series(
    y: pd.Series,
    x: Optional[pd.Series],
    *,
    max_points: int,
    method: DownsampleMethod = "lttb",
) -> DownsampleResult:
    if max_points <= 0:
        raise ValueError("max_points deve essere positivo.")

    original_count = len(y)
    if original_count <= max_points:
        return DownsampleResult(
            y=y,
            x=x,
            indices=y.index.to_numpy(),
            method="identity",
            original_count=original_count,
            sampled_count=original_count,
        )

    df = pd.DataFrame({"y": y})
    if x is not None:
        df["x"] = x
    df = df.dropna(subset=["y"])
    if x is not None:
        df = df.dropna(subset=["x"])

    if df.empty or len(df) <= max_points:
        subset = df.index.to_numpy()
        return DownsampleResult(
            y=y.loc[subset],
            x=x.loc[subset] if x is not None else None,
            indices=subset,
            method="identity",
            original_count=original_count,
            sampled_count=len(subset),
        )

    work_y = df["y"].to_numpy(dtype=np.float64)
    work_x = _as_numeric_x(df["x"]) if x is not None else np.arange(len(df), dtype=np.float64)

    target = min(max_points, len(df))
    if method == "lttb":
        idx_positions = _lttb_indices_cached(work_x, work_y, target)
    else:
        idx_positions = _minmax_indices(work_y, target)

    if idx_positions[0] != 0:
        idx_positions = np.insert(idx_positions, 0, 0)
    if idx_positions[-1] != len(df) - 1:
        idx_positions = np.append(idx_positions, len(df) - 1)

    idx_positions = np.unique(np.clip(idx_positions, 0, len(df) - 1))
    if idx_positions.size > target:
        idx_positions = np.unique(
            idx_positions[np.linspace(0, idx_positions.size - 1, num=target, dtype=int)]
        )

    selected_index = df.index.to_numpy()[idx_positions]
    y_down = y.loc[selected_index]
    x_down = x.loc[selected_index] if x is not None else None

    return DownsampleResult(
        y=y_down,
        x=x_down,
        indices=selected_index,
        method=method,
        original_count=original_count,
        sampled_count=len(y_down),
    )


def bulk_downsample(
    series: Sequence[Tuple[pd.Series, Optional[pd.Series]]],
    *,
    max_points: int,
    method: DownsampleMethod = "lttb",
) -> Tuple[DownsampleResult, ...]:
    results = []
    for y, x in series:
        results.append(downsample_series(y, x, max_points=max_points, method=method))
    return tuple(results)
