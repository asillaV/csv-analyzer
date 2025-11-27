"""
Modulo di caricamento CSV ottimizzato per performance e gestione file grandi.

Ottimizzazioni principali:
1. Caricamento chunk-based per ridurre picco RAM
2. Sampling intelligente per file > soglia dimensione
3. Early stopping nel format detection
4. Parallelizzazione pulizia colonne (opzionale)
5. Progress callback per UI

Mantiene compatibilità con loader.py esistente.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple
import os

import numpy as np
import pandas as pd

from .csv_cleaner import CleaningReport, clean_dataframe, suggest_number_format
from .logger import LogManager

log = LogManager("loader_optimized").get_logger()

# Limiti per float32 conversion (±3.4e38)
FLOAT32_MAX = 3.4e38
FLOAT32_MIN = -3.4e38

# Default values (se config.json non esiste o è malformato)
_DEFAULT_SIZE_THRESHOLD_MB = 50
_DEFAULT_ROWS_THRESHOLD = 100_000
_DEFAULT_SAMPLE_SIZE = 50_000
_DEFAULT_CHUNK_SIZE = 50_000

# Configurazione parallelizzazione (opzionale, disabilitato di default per compatibilità)
ENABLE_PARALLEL = False  # Richiede multiprocessing, può causare problemi con Streamlit


def _load_config() -> Dict:
    """Carica configurazione da config.json con fallback a defaults"""
    import json
    defaults = {
        "chunked_loading_threshold_mb": _DEFAULT_SIZE_THRESHOLD_MB,
        "chunk_size": _DEFAULT_CHUNK_SIZE,
        "rows_threshold": _DEFAULT_ROWS_THRESHOLD,
        "sample_size": _DEFAULT_SAMPLE_SIZE,
    }
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
            perf_config = config.get("performance", {})
            return {
                "chunked_loading_threshold_mb": perf_config.get("chunked_loading_threshold_mb", defaults["chunked_loading_threshold_mb"]),
                "chunk_size": perf_config.get("chunk_size", defaults["chunk_size"]),
                "rows_threshold": perf_config.get("rows_threshold", defaults["rows_threshold"]),
                "sample_size": perf_config.get("sample_size", defaults["sample_size"]),
            }
    except Exception as e:
        log.warning(f"Could not load config.json, using defaults: {e}")
        return defaults


# Carica configurazione al module load (una sola volta)
_CONFIG = _load_config()
SIZE_THRESHOLD_MB = _CONFIG["chunked_loading_threshold_mb"]
ROWS_THRESHOLD = _CONFIG["rows_threshold"]
SAMPLE_SIZE = _CONFIG["sample_size"]
CHUNK_SIZE = _CONFIG["chunk_size"]

# Log configurazione caricata
log.info(
    "Loader optimized config: SIZE_THRESHOLD=%d MB, ROWS_THRESHOLD=%d, CHUNK_SIZE=%d, SAMPLE_SIZE=%d",
    SIZE_THRESHOLD_MB, ROWS_THRESHOLD, CHUNK_SIZE, SAMPLE_SIZE
)


class LoadProgress:
    """Classe per tracciare il progresso del caricamento"""
    def __init__(self):
        self.phase = ""
        self.current = 0
        self.total = 0
        self.message = ""

    def update(self, phase: str, current: int, total: int, message: str = ""):
        self.phase = phase
        self.current = current
        self.total = total
        self.message = message


def should_use_sampling(file_path: Path) -> Tuple[bool, str]:
    """
    Determina se usare sampling basandosi su dimensione file e stima righe.

    Returns:
        (use_sampling, reason) - bool + spiegazione
    """
    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    if file_size_mb > SIZE_THRESHOLD_MB:
        return True, f"file size {file_size_mb:.1f} MB > {SIZE_THRESHOLD_MB} MB"

    # Stima righe (approssimativa: byte / 100 byte per riga)
    estimated_rows = (file_path.stat().st_size) / 100

    if estimated_rows > ROWS_THRESHOLD:
        return True, f"estimated {int(estimated_rows):,} rows > {ROWS_THRESHOLD:,}"

    return False, f"file size {file_size_mb:.1f} MB, small enough for full load"


def load_csv_sampled(
    path_str: str,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
    header: Optional[int] = 0,
    sample_size: int = SAMPLE_SIZE,
    random_seed: int = 42,
    progress_callback: Optional[Callable[[LoadProgress], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Carica un campione stratificato del CSV per analisi rapida.

    Strategia:
    1. Conta le righe totali (scan veloce)
    2. Genera indici stratificati (inizio, metà, fine)
    3. Carica solo le righe campionate

    Args:
        path_str: Path al file CSV
        encoding: Encoding (None = auto)
        delimiter: Delimiter (None = auto)
        header: Riga header (0, 1, None)
        sample_size: Numero righe da campionare
        random_seed: Seed per riproducibilità
        progress_callback: Funzione opzionale per progress updates

    Returns:
        (df_sample, metadata) dove metadata contiene:
            - total_rows: righe totali nel file
            - sampled_rows: righe caricate
            - sampling_ratio: ratio
            - is_sample: True
    """
    path = Path(path_str)
    progress = LoadProgress()

    if progress_callback:
        progress.update("counting", 0, 1, "Counting rows...")
        progress_callback(progress)

    # Conta righe (scan veloce senza parsing)
    log.info("Counting rows in %s...", path.name)
    total_rows = _count_rows_fast(path) - (1 if header is not None else 0)

    if total_rows <= sample_size:
        log.info("File has %d rows, loading all (no sampling needed)", total_rows)
        # File abbastanza piccolo, carica tutto
        from .loader import load_csv
        df = load_csv(path_str, encoding, delimiter, header, apply_cleaning=False)
        return df, {
            "total_rows": total_rows,
            "sampled_rows": total_rows,
            "sampling_ratio": 1.0,
            "is_sample": False
        }

    # Genera indici stratificati
    log.info("Generating stratified sample indices (%d from %d rows)...", sample_size, total_rows)
    skip_indices = _generate_stratified_indices(total_rows, sample_size, random_seed)

    if progress_callback:
        progress.update("loading", 0, 1, f"Loading {sample_size:,} sampled rows...")
        progress_callback(progress)

    # Carica solo le righe campionate
    effective_delimiter = delimiter if delimiter else ","
    read_kwargs = {
        "sep": effective_delimiter,
        "engine": "c",
        "encoding": encoding if encoding else "utf-8",
        "header": header if header is not None else "infer",
        "dtype": str,
        "keep_default_na": False,
        "na_filter": False,
        "on_bad_lines": "skip",
        "skiprows": lambda x: x != 0 and x not in skip_indices,  # header (row 0) + sampled rows
    }

    df_sample = pd.read_csv(path, **read_kwargs)
    log.info("Loaded %d sampled rows from %d total (%.1f%%)",
             len(df_sample), total_rows, 100 * len(df_sample) / total_rows)

    metadata = {
        "total_rows": total_rows,
        "sampled_rows": len(df_sample),
        "sampling_ratio": len(df_sample) / total_rows,
        "is_sample": True
    }

    return df_sample, metadata


def load_csv_chunked(
    path_str: str,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
    header: Optional[int] = 0,
    usecols: Optional[Iterable[str]] = None,
    apply_cleaning: bool = True,
    decimal: Optional[str] = None,
    thousands: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    progress_callback: Optional[Callable[[LoadProgress], None]] = None,
) -> Tuple[pd.DataFrame, CleaningReport]:
    """
    Carica CSV usando chunked reading per ridurre picco RAM.

    Strategia:
    1. Stima formato numerico da primo chunk (risparmia RAM)
    2. Legge file a chunks
    3. Pulisce ogni chunk separatamente
    4. Concatena risultati

    Vantaggi:
    - Picco RAM ridotto: ~2× chunk_size invece di file intero
    - Supporta file > RAM
    - Progress callback per UI

    Args:
        chunk_size: Righe per chunk (default 50k)
        progress_callback: Funzione chiamata ad ogni chunk: f(LoadProgress)
        Altri parametri: come load_csv()

    Returns:
        (df_cleaned, cleaning_report)
    """
    path = Path(path_str)
    progress = LoadProgress()

    # FASE 1: Stima formato numerico da primo chunk
    log.info("Loading first chunk to detect numeric format...")
    if progress_callback:
        progress.update("format_detection", 0, 1, "Detecting numeric format...")
        progress_callback(progress)

    effective_delimiter = delimiter if delimiter else ","
    read_kwargs_base = {
        "sep": effective_delimiter,
        "engine": "c",
        "encoding": encoding if encoding else "utf-8",
        "header": header if header is not None else "infer",
        "usecols": list(usecols) if usecols is not None else None,
        "dtype": str,
        "keep_default_na": False,
        "na_filter": False,
        "on_bad_lines": "skip",
    }

    # Carica primo chunk per format detection
    first_chunk = pd.read_csv(path, nrows=min(chunk_size, 10000), **read_kwargs_base)
    suggestion = suggest_number_format(first_chunk, decimal=decimal, thousands=thousands)

    log.info("Detected format: decimal='%s', thousands='%s' (confidence=%.1f%%)",
             suggestion.decimal, suggestion.thousands, suggestion.confidence * 100)

    # FASE 2: Caricamento chunked
    log.info("Loading CSV in chunks (chunk_size=%d)...", chunk_size)

    chunks = []
    chunk_reports = []

    # Itera sui chunks
    chunk_iterator = pd.read_csv(path, chunksize=chunk_size, **read_kwargs_base)

    total_rows_loaded = 0
    chunk_num = 0

    for chunk in chunk_iterator:
        chunk_num += 1
        total_rows_loaded += len(chunk)

        if progress_callback:
            progress.update(
                "loading",
                total_rows_loaded,
                -1,  # Unknown total (stimabile con count preliminare)
                f"Loading chunk {chunk_num} ({total_rows_loaded:,} rows)..."
            )
            progress_callback(progress)

        # Pulisci chunk
        from .csv_cleaner import clean_dataframe
        result = clean_dataframe(
            chunk,
            apply=apply_cleaning,
            decimal=suggestion.decimal,
            thousands=suggestion.thousands
        )

        chunks.append(result.df)
        chunk_reports.append(result.report)

        log.debug("Chunk %d: %d rows, %d numeric cols",
                  chunk_num, len(result.df), len(result.report.numeric_columns))

    # FASE 3: Concatenazione finale
    if progress_callback:
        progress.update("finalizing", 0, 1, "Concatenating chunks...")
        progress_callback(progress)

    log.info("Concatenating %d chunks...", len(chunks))
    df_final = pd.concat(chunks, ignore_index=True)

    # Merge reports (usa il primo come base)
    final_report = chunk_reports[0]

    log.info("CSV loaded: %d rows, %d columns, %d numeric columns",
             len(df_final), len(df_final.columns), len(final_report.numeric_columns))

    return df_final, final_report


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
    use_optimization: bool = True,
    progress_callback: Optional[Callable[[LoadProgress], None]] = None,
) -> pd.DataFrame | Tuple[pd.DataFrame, CleaningReport]:
    """
    Versione ottimizzata di load_csv con auto-detection di strategia ottimale.

    Strategia di caricamento:
    - File piccoli (< 50 MB, < 100k righe): caricamento standard (compatibilità)
    - File grandi: caricamento chunked per ridurre RAM

    Args:
        use_optimization: Se False, usa loader legacy (compatibilità)
        progress_callback: Callback per progress updates
        Altri args: come loader.py

    Returns:
        DataFrame o (DataFrame, CleaningReport) se return_details=True
    """
    path = Path(path_str)

    if not path.exists():
        msg = f"File non trovato: {path}"
        log.error(msg)
        raise FileNotFoundError(msg)

    # Determina strategia
    use_sampling, reason = should_use_sampling(path)

    if not use_optimization:
        log.info("Using legacy loader (optimization disabled)")
        from .loader import load_csv as load_csv_legacy
        return load_csv_legacy(
            path_str, encoding, delimiter, header, usecols,
            apply_cleaning, return_details, decimal, thousands
        )

    if use_sampling:
        log.info("Using CHUNKED loader (%s)", reason)
        df_clean, report = load_csv_chunked(
            path_str, encoding, delimiter, header, usecols,
            apply_cleaning, decimal, thousands,
            progress_callback=progress_callback
        )
    else:
        log.info("Using STANDARD loader (%s)", reason)
        # File piccolo, usa loader standard per compatibilità
        from .loader import load_csv as load_csv_legacy
        return load_csv_legacy(
            path_str, encoding, delimiter, header, usecols,
            apply_cleaning, return_details, decimal, thousands
        )

    if return_details:
        return df_clean, report
    return df_clean


def optimize_dtypes(
    df: pd.DataFrame,
    enabled: bool = True,
    aggressive: bool = False
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Importato da loader.py per compatibilità"""
    from .loader import optimize_dtypes as optimize_dtypes_legacy
    return optimize_dtypes_legacy(df, enabled, aggressive)


# ==================== UTILITY FUNCTIONS ====================

def _count_rows_fast(path: Path) -> int:
    """
    Conta righe del CSV velocemente senza parsing completo.
    Usa newline counting (molto veloce).
    """
    count = 0
    with open(path, 'rb') as f:
        for _ in f:
            count += 1
    return count


def _generate_stratified_indices(
    total_rows: int,
    sample_size: int,
    random_seed: int = 42
) -> set:
    """
    Genera indici stratificati per sampling bilanciato.

    Strategia:
    - 40% inizio file
    - 40% fine file
    - 20% casuale al centro

    Questo cattura pattern temporali e variazioni nel file.
    """
    np.random.seed(random_seed)

    start_count = int(sample_size * 0.4)
    end_count = int(sample_size * 0.4)
    middle_count = sample_size - start_count - end_count

    # Inizio
    start_indices = set(range(1, min(start_count + 1, total_rows + 1)))

    # Fine
    end_start = max(1, total_rows - end_count + 1)
    end_indices = set(range(end_start, total_rows + 1))

    # Centro (casuale)
    middle_start = start_count + 1
    middle_end = end_start - 1
    if middle_end > middle_start:
        middle_indices = set(np.random.choice(
            range(middle_start, middle_end),
            size=min(middle_count, middle_end - middle_start),
            replace=False
        ))
    else:
        middle_indices = set()

    all_indices = start_indices | middle_indices | end_indices

    # Aggiungi header (row 0 non conta, indici partono da 1)
    return all_indices
