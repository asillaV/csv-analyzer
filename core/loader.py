from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .logger import LogManager

log = LogManager("loader").get_logger()


def load_csv(
    path_str: str,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
    header: Optional[int] = 0,
    usecols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Carica il CSV con opzioni:
    - encoding/delimiter possono essere None (pandas tenterà inferenza).
    - header: 0, None (per nessuna riga header).
    - usecols: elenco colonne da leggere (stringhe) o None per tutte.
    """
    path = Path(path_str)
    if not path.exists():
        msg = f"File non trovato: {path}"
        log.error(msg)
        raise FileNotFoundError(msg)

    try:
        df = pd.read_csv(
            path,
            sep=delimiter if delimiter else None,
            engine="python",
            encoding=encoding if encoding else None,
            header=header if header is not None else "infer",
            usecols=list(usecols) if usecols is not None else None,
        )
        log.info("CSV caricato: %s (righe=%d, colonne=%d)", path.name, len(df), len(df.columns))
        return df
    except ValueError as ve:
        log.error("Valore non valido in load_csv (usecols o header): %s", ve, exc_info=True)
        raise
    except Exception as e:
        log.error("Errore in lettura CSV: %s", e, exc_info=True)
        raise
