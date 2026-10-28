"""
Strumenti di parità per il caricamento CSV (Parte 1, F0).

- `load_reference()` riproduce il percorso di caricamento usato oggi da
  web_app.py (`analyze_csv` + `loader_optimized.load_csv`): è l'oracolo.
- `assert_load_equivalent()` confronta due esiti di caricamento secondo il
  contratto della specifica: stesse colonne, stessi dtype, stessi valori (con
  tolleranza), stessi NA, stesso formato numerico e stesse colonne numeriche.

Usato da tests/test_ingest_parity.py e da scripts/bench_load.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from core.csv_cleaner import CleaningReport

VALUE_RTOL = 1e-12


@dataclass
class LoadOutcome:
    """Esito di un caricamento: risultato oppure eccezione."""

    df: Optional[pd.DataFrame] = None
    report: Optional[CleaningReport] = None
    meta: Optional[Dict[str, Any]] = None
    error: Optional[BaseException] = None


Loader = Callable[[Path], LoadOutcome]


def load_reference(path: Path, apply_cleaning: bool = True) -> LoadOutcome:
    """Caricamento con il ramo standard del loader storico (oracolo di parità)."""
    from core.analyzer import analyze_csv
    from core.loader import load_csv

    try:
        meta = analyze_csv(str(path))
        df, report = load_csv(
            str(path),
            encoding=meta.get("encoding"),
            delimiter=meta.get("delimiter"),
            header=meta.get("header"),
            apply_cleaning=apply_cleaning,
            return_details=True,
        )
        return LoadOutcome(df=df, report=report, meta=dict(meta))
    except Exception as exc:
        return LoadOutcome(error=exc)


MSVC_SPECIAL_RE = r"^\s*[+-]?\d+\.#"


def apply_msvc_rule(ref: LoadOutcome, path: Path) -> LoadOutcome:
    """Applica al riferimento la regola voluta "token MSVC = NA".

    Il loader storico toglie i caratteri non numerici da "1.#IO"/"-1.#R" e, se i
    token superano ~5% della colonna, li trasforma in ±1.0. Qui quelle celle del
    riferimento diventano NA, così il confronto resta esatto su tutto il resto.
    """
    if ref.error is not None or ref.df is None or ref.report is None or ref.meta is None:
        return ref
    raw = pd.read_csv(
        path,
        sep=ref.meta.get("delimiter") or ",",
        encoding=ref.meta.get("encoding") or "utf-8",
        header=ref.meta.get("header") if ref.meta.get("header") is not None else "infer",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="skip",
    )
    df = ref.df.copy()
    for position, name in enumerate(df.columns):
        if name not in ref.report.numeric_columns:
            continue
        tokens = raw.iloc[:, position].astype("string").str.match(MSVC_SPECIAL_RE, na=False).to_numpy()
        if tokens.any():
            column = df.iloc[:, position].copy()
            column[tokens] = pd.NA
            df.iloc[:, position] = column
    return LoadOutcome(df=df, report=ref.report, meta=ref.meta)


def load_reference_chunked(path: Path, chunk_size: int = 1000) -> LoadOutcome:
    """Ramo chunked del loader attuale (usato oggi oltre le soglie di dimensione)."""
    from core.analyzer import analyze_csv
    from core.loader_optimized import load_csv_chunked

    try:
        meta = analyze_csv(str(path))
        df, report = load_csv_chunked(
            str(path),
            encoding=meta.get("encoding"),
            delimiter=meta.get("delimiter"),
            header=meta.get("header"),
            chunk_size=chunk_size,
        )
        return LoadOutcome(df=df, report=report, meta=dict(meta))
    except Exception as exc:
        return LoadOutcome(error=exc)


def _column_values_equal(name: str, ref: pd.Series, new: pd.Series) -> Optional[str]:
    if pd.api.types.is_numeric_dtype(ref.dtype):
        ref_na = ref.isna().to_numpy()
        new_na = new.isna().to_numpy()
        if not np.array_equal(ref_na, new_na):
            first = int(np.argmax(ref_na != new_na))
            return f"colonna {name!r}: NA diversi (prima differenza riga {first}: {ref.iloc[first]!r} vs {new.iloc[first]!r})"
        a = ref.to_numpy(dtype="float64", na_value=np.nan)
        b = new.to_numpy(dtype="float64", na_value=np.nan)
        close = np.isclose(a, b, rtol=VALUE_RTOL, atol=0.0, equal_nan=True)
        if not close.all():
            first = int(np.argmin(close))
            return f"colonna {name!r}: valori diversi (riga {first}: {a[first]!r} vs {b[first]!r})"
        return None
    a = ref.astype("string").fillna("<NA>").to_numpy()
    b = new.astype("string").fillna("<NA>").to_numpy()
    if not np.array_equal(a, b):
        first = int(np.argmax(a != b))
        return f"colonna {name!r}: testo diverso (riga {first}: {a[first]!r} vs {b[first]!r})"
    return None


def compare_outcomes(
    ref: LoadOutcome, new: LoadOutcome, ignore_columns: Iterable[str] = ()
) -> list[str]:
    """Restituisce l'elenco delle differenze (vuoto = equivalenti).

    `ignore_columns`: colonne in cui il riferimento ha un difetto noto; ne
    vengono confrontati solo nome e posizione.
    """
    ignored = set(ignore_columns)
    if ref.error is not None or new.error is not None:
        if ref.error is not None and new.error is not None:
            return []
        if ref.error is not None:
            return [f"il riferimento fallisce ({ref.error!r}) ma il nuovo no"]
        return [f"il nuovo fallisce: {new.error!r}"]

    assert ref.df is not None and new.df is not None
    diffs: list[str] = []
    if list(ref.df.columns) != list(new.df.columns):
        return [f"colonne diverse: {list(ref.df.columns)} vs {list(new.df.columns)}"]
    if ref.df.shape != new.df.shape:
        return [f"forma diversa: {ref.df.shape} vs {new.df.shape}"]
    for name in ref.df.columns:
        if name in ignored:
            continue
        if str(ref.df[name].dtype) != str(new.df[name].dtype):
            diffs.append(f"colonna {name!r}: dtype {ref.df[name].dtype} vs {new.df[name].dtype}")
            continue
        msg = _column_values_equal(name, ref.df[name], new.df[name])
        if msg:
            diffs.append(msg)

    if ref.report is not None and new.report is not None:
        ref_numeric = [c for c in ref.report.numeric_columns if c not in ignored]
        new_numeric = [c for c in new.report.numeric_columns if c not in ignored]
        if ref_numeric != new_numeric:
            diffs.append(
                f"colonne numeriche diverse: {ref.report.numeric_columns} vs {new.report.numeric_columns}"
            )
        rs, ns = ref.report.suggestion, new.report.suggestion
        if (rs.decimal, rs.thousands) != (ns.decimal, ns.thousands):
            diffs.append(f"formato numerico diverso: {(rs.decimal, rs.thousands)} vs {(ns.decimal, ns.thousands)}")
    return diffs


def assert_load_equivalent(
    ref: LoadOutcome, new: LoadOutcome, ignore_columns: Iterable[str] = ()
) -> None:
    diffs = compare_outcomes(ref, new, ignore_columns)
    assert not diffs, "Caricamento non equivalente:\n  - " + "\n  - ".join(diffs)
