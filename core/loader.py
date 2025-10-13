from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

from .csv_cleaner import CleaningReport, clean_dataframe
from .logger import LogManager

log = LogManager("loader").get_logger()


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
