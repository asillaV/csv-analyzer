from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .csv_cleaner import CleaningReport, clean_dataframe
from .logger import LogManager

log = LogManager("loader").get_logger()

# Limits for float32 conversion (±3.4e38)
FLOAT32_MAX = 3.4e38
FLOAT32_MIN = -3.4e38


def load_csv(
    path_str: str,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
    header: Optional[int] = 0,
    usecols: Optional[Iterable[str]] = None,
    apply_cleaning: bool = True,
    return_details: bool = False,
    decimal: Optional[str] = None,
    thousands: Optional[str] = None,
) -> pd.DataFrame | Tuple[pd.DataFrame, CleaningReport]:
    """
    Carica il CSV e applica la sanificazione opzionale dei numeri.

    - encoding/delimiter possono essere None (pandas tenterà l'inferenza).
    - header: 0, None (nessuna riga header) o indice della riga header.
    - usecols: elenco colonne da leggere (stringhe) o None per tutte.
    - apply_cleaning: abilita la correzione automatica di decimal/thousands.
    - return_details: se True restituisce anche il report di cleaning.
    - decimal/thousands: consentono di forzare i separatori rilevati.
    """
    path = Path(path_str)
    if not path.exists():
        msg = f"File non trovato: {path}"
        log.error(msg)
        raise FileNotFoundError(msg)

    read_kwargs = {
        "sep": delimiter if delimiter else None,
        "engine": "python",
        "encoding": encoding if encoding else None,
        "header": header if header is not None else "infer",
        "usecols": list(usecols) if usecols is not None else None,
        "dtype": "string",
        "keep_default_na": False,
        "na_filter": False,
        "on_bad_lines": "skip",
    }

    try:
        df_raw = pd.read_csv(path, **read_kwargs)
    except pd.errors.EmptyDataError as ede:
        log.error("File CSV vuoto o non leggibile (%s): %s", path.name, ede, exc_info=True)
        raise
    except ValueError as ve:
        log.error("Valore non valido in load_csv (usecols o header): %s", ve, exc_info=True)
        raise
    except Exception as e:
        log.error("Errore in lettura CSV: %s", e, exc_info=True)
        raise

    result = clean_dataframe(
        df_raw,
        apply=apply_cleaning,
        decimal=decimal,
        thousands=thousands,
    )

    df_clean = result.df
    log.info(
        "CSV caricato: %s (righe=%d, colonne=%d, col_numeric=%d, cleaning=%s)",
        path.name,
        len(df_clean),
        len(df_clean.columns),
        len(result.report.numeric_columns),
        "on" if apply_cleaning else "off",
    )

    if result.report.rows_all_nan_after_clean:
        log.debug(
            "Righe con soli NaN dopo cleaning: %s",
            result.report.rows_all_nan_after_clean,
        )

    if return_details:
        return df_clean, result.report
    return df_clean


def optimize_dtypes(
    df: pd.DataFrame,
    enabled: bool = True,
    aggressive: bool = False
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Ottimizza i dtype delle colonne DataFrame per ridurre memoria.

    Issue #53: Converte float64 → float32 quando i valori rientrano nei limiti
    di float32 (±3.4e38), risparmiando ~50% memoria su colonne numeriche.

    Args:
        df: DataFrame da ottimizzare
        enabled: Se False, ritorna df invariato (per configurabilità)
        aggressive: Se True, tenta anche ottimizzazioni int64→int32 (non implementato)

    Returns:
        (df_optimized, conversions_map) dove conversions_map è un dict
        {column_name: "float64→float32"} per le colonne convertite.

    Example:
        >>> df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})  # float64 default
        >>> df_opt, conversions = optimize_dtypes(df)
        >>> df_opt["a"].dtype  # float32
        dtype('float32')
        >>> conversions
        {'a': 'float64→float32'}
    """
    if not enabled:
        return df, {}

    df_optimized = df.copy()
    conversions: Dict[str, str] = {}

    # Ottimizza colonne float64 → float32
    float64_cols = df_optimized.select_dtypes(include=["float64"]).columns

    for col in float64_cols:
        series = df_optimized[col]

        # Controlla se tutti i valori (non-NaN) rientrano nei limiti float32
        if series.isna().all():
            # Colonna tutta NaN → safe da convertire
            df_optimized[col] = series.astype("float32")
            conversions[col] = "float64→float32"
            continue

        # Calcola min/max solo su valori finiti (esclude NaN e Inf)
        finite_mask = np.isfinite(series)
        if not finite_mask.any():
            # Nessun valore finito (solo NaN/Inf) → conversione safe
            df_optimized[col] = series.astype("float32")
            conversions[col] = "float64→float32"
            continue

        finite_values = series[finite_mask]
        abs_max = float(finite_values.abs().max())

        # Verifica se rientra nei limiti float32
        if abs_max < FLOAT32_MAX:
            df_optimized[col] = series.astype("float32")
            conversions[col] = "float64→float32"
        else:
            log.debug(
                "Colonna '%s' non convertita a float32: abs_max=%.2e >= %.2e",
                col,
                abs_max,
                FLOAT32_MAX,
            )

    if conversions:
        log.info(
            "Ottimizzazione dtype completata: %d colonne convertite float64→float32",
            len(conversions),
        )
        for col, conversion in conversions.items():
            log.debug("  - %s: %s", col, conversion)

    # Placeholder per future ottimizzazioni (int64→int32, ecc.)
    if aggressive:
        log.warning("Aggressive dtype optimization non ancora implementata")

    return df_optimized, conversions
