"""
Caricamento CSV veloce basato su pyarrow (Parte 1, F1).

Contratto di uscita identico al loader standard (`core.loader.load_csv`):
colonne numeriche `Float64` nullable, colonne non numeriche come stringhe
originali, nomi duplicati rinominati come pandas, `CleaningReport` con gli
stessi campi. La parità è verificata da tests/test_ingest_parity.py.

Strategia:
  1. `analyze_csv` rileva encoding, separatore e riga di intestazione.
  2. pyarrow legge tutto il file come stringhe (colonnare, multi-thread,
     senza creare oggetti Python per cella).
  3. Il formato numerico è stimato con `suggest_number_format` sullo stesso
     campione che usa il loader standard.
  4. Ogni colonna viene convertita a float con operazioni Arrow vettoriali;
     solo le colonne che non si convertono in modo pulito (migliaia, valute,
     simboli...) passano a `csv_cleaner.clean_column`, che è la logica storica.
  5. Casi che Arrow non gestisce come pandas (righe con meno campi, newline nei
     campi quotati, errori di parsing) ricadono sul loader standard.

Differenze volute rispetto al loader standard (difetti del vecchio):
  - file non UTF-8 senza BOM (es. cp1252): riletti con cp1252 invece di crash;
  - file senza intestazione: nessuna riga di dati usata come intestazione;
  - token speciali MSVC (`1.#INF`, `1.#QNAN`, `1.#IO`, `-1.#R`...): sempre NA,
    mai convertiti in ±1 dalla rimozione dei caratteri non numerici.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .analyzer import analyze_csv
from .csv_cleaner import (
    MAX_SAMPLE_VALUES,
    CleaningReport,
    ColumnReport,
    FormatSuggestion,
    clean_column,
    suggest_number_format,
)
from .logger import LogManager

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv

    _ARROW_OK = True
except Exception:  # pragma: no cover - pyarrow è nei requirements
    _ARROW_OK = False

log = LogManager("ingest").get_logger()

# Da incrementare a ogni modifica che cambia il risultato (invalida la cache F2).
INGEST_VERSION = 1

FALLBACK_ENCODING = "cp1252"
BLOCK_SIZE = 64 << 20
MSVC_SPECIAL_RE = r"^[+-]?\d+\.#"
# Oltre questa quota di righe scartate per troppi campi, meglio il loader standard.
MAX_SKIPPED_ROWS_RATIO = 0.5
MAX_CONVERT_WORKERS = 8
ROWS_ALL_NAN_BATCH = 4096

ProgressCallback = Callable[[Any], None]


@dataclass
class IngestResult:
    df: pd.DataFrame
    report: CleaningReport
    meta: Dict[str, Any]
    timings: Dict[str, float] = field(default_factory=dict)
    engine: str = "arrow"


class FallbackToLegacy(Exception):
    """Il file va letto con il loader standard."""


def load(
    path: str | Path,
    *,
    meta: Optional[Dict[str, Any]] = None,
    apply_cleaning: bool = True,
    decimal: Optional[str] = None,
    thousands: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> IngestResult:
    """Carica un CSV e restituisce DataFrame pulito, report, metadati e tempi per fase.

    `meta`: risultato di `analyze_csv` già calcolato dal chiamante (encoding,
    delimiter, header, columns); se assente viene calcolato qui.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    if meta is None:
        _progress(progress_callback, "analyze", 0, 1, "Analisi formato...")
        meta = analyze_csv(str(path))
    meta = dict(meta)
    timings["analyze_s"] = time.perf_counter() - t0

    if not _ARROW_OK or not apply_cleaning:
        return _load_legacy(path, meta, apply_cleaning, decimal, thousands, timings)

    try:
        return _load_arrow(path, meta, decimal, thousands, timings, progress_callback)
    except FallbackToLegacy as reason:
        log.info("Ingest veloce non applicabile a %s (%s): uso il loader standard", path.name, reason)
        return _load_legacy(path, meta, apply_cleaning, decimal, thousands, timings)


# ---------------------------------------------------------------------------
# Percorso Arrow
# ---------------------------------------------------------------------------
def _load_arrow(
    path: Path,
    meta: Dict[str, Any],
    decimal: Optional[str],
    thousands: Optional[str],
    timings: Dict[str, float],
    progress_callback: Optional[ProgressCallback],
) -> IngestResult:
    delimiter = meta.get("delimiter") or ","
    header = meta.get("header")
    encoding = meta.get("encoding") or "utf-8"

    t0 = time.perf_counter()
    _progress(progress_callback, "loading", 0, 1, "Lettura file...")
    try:
        names = column_names(path, encoding, delimiter, header, meta)
        table, skipped = _read_strings(path, encoding, delimiter, header, len(names))
    except (UnicodeDecodeError, InvalidText):
        if encoding.lower().replace("-", "") not in ("utf8", "utf8sig"):
            raise FallbackToLegacy(f"decodifica fallita con {encoding}")
        log.info("%s non è UTF-8 valido: rilettura con %s", path.name, FALLBACK_ENCODING)
        encoding = FALLBACK_ENCODING
        meta["encoding"] = encoding
        names = column_names(path, encoding, delimiter, header, meta)
        table, skipped = _read_strings(path, encoding, delimiter, header, len(names))
    timings["read_s"] = time.perf_counter() - t0

    if skipped and skipped >= MAX_SKIPPED_ROWS_RATIO * (table.num_rows + skipped):
        raise FallbackToLegacy(f"{skipped} righe con troppi campi")

    t1 = time.perf_counter()
    _progress(progress_callback, "cleaning", 0, len(names), "Conversione colonne...")
    suggestion = suggest_format(table, names, decimal, thousands)
    columns, report = _convert_columns(table, names, suggestion, progress_callback)
    timings["convert_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    _progress(progress_callback, "finalizing", 0, 1, "Creazione tabella...")
    df = pd.DataFrame(dict(enumerate(columns)), copy=False)
    df.columns = names
    timings["frame_s"] = time.perf_counter() - t2
    timings["total_s"] = sum(timings.values())

    log.info(
        "CSV caricato (arrow): %s righe=%d colonne=%d numeriche=%d in %.2fs",
        path.name, len(df), len(df.columns), len(report.numeric_columns), timings["total_s"],
    )
    return IngestResult(df=df, report=report, meta=meta, timings=timings, engine="arrow")


class InvalidText(Exception):
    """Byte non validi per l'encoding dichiarato."""


def column_names(
    path: Path, encoding: str, delimiter: str, header: Optional[int], meta: Dict[str, Any]
) -> List[str]:
    """Nomi colonna esattamente come li produrrebbe pandas (duplicati inclusi)."""
    if header is None:
        names = list(meta.get("columns") or [])
        if names:
            return names
        first = pd.read_csv(path, sep=delimiter, encoding=encoding, header=None, nrows=1, dtype=str)
        return [f"Column_{i}" for i in range(first.shape[1])]
    try:
        head = pd.read_csv(
            path, sep=delimiter, encoding=encoding, header=header, nrows=0, dtype=str, engine="c"
        )
    except UnicodeDecodeError:
        raise
    except Exception as exc:
        raise FallbackToLegacy(f"intestazione non leggibile: {exc}") from exc
    return [str(c) for c in head.columns]


class InvalidRowCounter:
    """Handler Arrow per le righe malformate: le scarta e le conta come fa pandas."""

    def __init__(self) -> None:
        self.too_many = 0
        self.too_few = 0

    def __call__(self, row) -> str:
        if row.actual_columns > row.expected_columns:
            self.too_many += 1
        else:
            self.too_few += 1
        return "skip"

    def check(self) -> None:
        if self.too_few:
            raise FallbackToLegacy(f"{self.too_few} righe con meno campi dell'intestazione")


def csv_options(
    encoding: str,
    delimiter: str,
    header: Optional[int],
    n_columns: int,
    counter: InvalidRowCounter,
    include: Optional[List[int]] = None,
    block_size: Optional[int] = None,
    use_threads: bool = True,
) -> Dict[str, Any]:
    """Opzioni pyarrow per leggere tutte le celle come stringhe originali.

    `include`: posizioni delle colonne da materializzare (le altre vengono solo
    tokenizzate). Le colonne interne si chiamano c0..cN.
    """
    if n_columns == 0:
        raise FallbackToLegacy("nessuna colonna rilevata")
    internal = [f"c{i}" for i in range(n_columns)]
    convert = dict(
        column_types={name: pa.string() for name in internal},
        strings_can_be_null=False,
        quoted_strings_can_be_null=False,
    )
    if include is not None:
        convert["include_columns"] = [internal[i] for i in include]
    return dict(
        read_options=pacsv.ReadOptions(
            encoding="utf8" if is_plain_utf8(encoding) else encoding,
            skip_rows=0 if header is None else header + 1,
            column_names=internal,
            block_size=block_size or BLOCK_SIZE,
            use_threads=use_threads,
        ),
        parse_options=pacsv.ParseOptions(delimiter=delimiter, invalid_row_handler=counter),
        convert_options=pacsv.ConvertOptions(**convert),
    )


def is_plain_utf8(encoding: str) -> bool:
    return encoding.lower().replace("-", "").replace("_", "") == "utf8"


def arrow_errors(exc: Exception) -> Exception:
    """Traduce gli errori Arrow in testo non valido (ritenta con cp1252) o ripiego."""
    if isinstance(exc, UnicodeDecodeError):
        return exc
    text = str(exc).lower()
    if "utf8" in text or "utf-8" in text:
        return InvalidText(str(exc))
    return FallbackToLegacy(f"parsing arrow: {exc}")


def _read_strings(
    path: Path, encoding: str, delimiter: str, header: Optional[int], n_columns: int
) -> Tuple["pa.Table", int]:
    """Legge tutte le celle come stringhe; restituisce la tabella e le righe scartate."""
    counter = InvalidRowCounter()
    options = csv_options(encoding, delimiter, header, n_columns, counter)
    try:
        table = pacsv.read_csv(str(path), **options)
    except (UnicodeDecodeError, pa.ArrowInvalid) as exc:
        raise arrow_errors(exc) from exc
    counter.check()
    return table, counter.too_many


def suggest_format(
    table: "pa.Table", names: List[str], decimal: Optional[str], thousands: Optional[str]
) -> FormatSuggestion:
    """Stessa stima del loader standard: usa solo le prime righe di ogni colonna."""
    rows_per_column = max(10, (MAX_SAMPLE_VALUES * 4) // max(1, len(names)))
    head = table.slice(0, rows_per_column).to_pandas()
    head.columns = names
    return suggest_number_format(head, decimal=decimal, thousands=thousands)


def _convert_columns(
    table: "pa.Table",
    names: List[str],
    suggestion: FormatSuggestion,
    progress_callback: Optional[ProgressCallback],
) -> Tuple[List[Any], CleaningReport]:
    columns: List[Any] = []
    reports: Dict[str, ColumnReport] = {}
    numeric_columns: List[str] = []
    numeric_positions: List[int] = []
    all_nan: Optional[np.ndarray] = None
    warnings = format_warnings(suggestion)

    # Le funzioni di pyarrow.compute rilasciano il GIL: le colonne si convertono in parallelo.
    workers = min(len(names), os.cpu_count() or 4, MAX_CONVERT_WORKERS)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = pool.map(lambda i: _convert_column(table.column(i), suggestion), range(len(names)))
        for i, (name, (converted, column_report)) in enumerate(zip(names, results)):
            reports[name] = column_report
            if converted is None:
                columns.append(table.column(i).to_pandas())
            else:
                numeric_columns.append(name)
                numeric_positions.append(i)
                columns.append(converted)
                na = converted.isna()
                all_nan = na if all_nan is None else (all_nan & na)
            _progress(progress_callback, "cleaning", i + 1, len(names), f"Colonna {i + 1}/{len(names)}")

    rows_all_nan: List[int] = []
    if all_nan is not None and all_nan.any():
        rows_all_nan = _rows_with_lost_digits(table, numeric_positions, np.flatnonzero(all_nan))

    report = CleaningReport(
        suggestion=suggestion,
        applied_cleaning=True,
        columns=reports,
        numeric_columns=numeric_columns,
        rows_all_nan_after_clean=rows_all_nan,
        warnings=warnings,
    )
    return columns, report


def format_warnings(suggestion: FormatSuggestion) -> List[str]:
    """Avvisi sul formato numerico, come in csv_cleaner.clean_dataframe."""
    warnings: List[str] = []
    if suggestion.sample_size == 0:
        warnings.append("Impossibile stimare il formato: nessun valore numerico trovato nel campione.")
    if suggestion.confidence < 0.25:
        warnings.append(f"Bassa confidenza nel formato numerico stimato ({suggestion.confidence:.0%}).")
    return warnings


@dataclass
class FastCast:
    """Esito della conversione vettoriale di una colonna (o di un blocco di righe)."""

    numbers: Optional[Any]  # pa.Array/ChunkedArray float64, None se il cast fallisce o non ci sono valori
    candidate_count: int  # celle non vuote dopo strip
    msvc: Optional[Any]  # maschera booleana dei token MSVC, None se assenti


def fast_cast(raw: Any, suggestion: FormatSuggestion) -> FastCast:
    """Conversione Arrow a float64 con le regole del percorso veloce del loader storico.

    Celle vuote e token MSVC diventano null, NaN diventa null. Se anche una sola
    cella non è convertibile restituisce `numbers=None`: la colonna va trattata
    con la logica storica (`clean_column`).
    """
    stripped = pc.utf8_trim_whitespace(raw)
    non_empty = pc.not_equal(stripped, "")
    candidate_count = int(pc.sum(non_empty).as_py() or 0)
    if candidate_count == 0:
        return FastCast(None, 0, None)

    msvc = None
    if pc.any(pc.match_substring(stripped, "#")).as_py():
        msvc = pc.match_substring_regex(stripped, MSVC_SPECIAL_RE)

    candidate = stripped
    if suggestion.decimal and suggestion.decimal != ".":
        candidate = pc.replace_substring(candidate, suggestion.decimal, ".")
    null_mask = pc.invert(non_empty)
    if msvc is not None:
        null_mask = pc.or_(null_mask, msvc)
    candidate = pc.if_else(null_mask, pa.scalar(None, pa.string()), candidate)

    try:
        numbers = pc.cast(candidate, pa.float64())
    except pa.ArrowInvalid:
        return FastCast(None, candidate_count, msvc)
    numbers = pc.if_else(pc.is_nan(numbers), pa.scalar(None, pa.float64()), numbers)
    return FastCast(numbers, candidate_count, msvc)


def empty_column_report(total: int) -> ColumnReport:
    return ColumnReport(
        total=total, candidate_numeric=0, converted=0, non_numeric=0,
        conversion_rate=0.0, dtype="object", applied=False, examples_non_numeric=[],
    )


def fast_column_report(total: int, candidate_count: int) -> ColumnReport:
    return ColumnReport(
        total=total,
        candidate_numeric=candidate_count,
        converted=candidate_count,
        non_numeric=0,
        conversion_rate=1.0,
        dtype="Float64",
        applied=True,
        examples_non_numeric=[],
    )


def _convert_column(
    raw: "pa.ChunkedArray", suggestion: FormatSuggestion
) -> Tuple[Optional[pd.arrays.FloatingArray], ColumnReport]:
    total = len(raw)
    cast = fast_cast(raw, suggestion)
    if cast.candidate_count == 0:
        return None, empty_column_report(total)
    if cast.numbers is None:
        return convert_column_legacy(raw, cast.msvc, suggestion)
    return to_float_array(cast.numbers), fast_column_report(total, cast.candidate_count)


def convert_column_legacy(
    raw: Any, msvc: Optional[Any], suggestion: FormatSuggestion
) -> Tuple[Optional[pd.arrays.FloatingArray], ColumnReport]:
    """Colonne non convertibili in modo pulito: logica storica di csv_cleaner."""
    series = raw.to_pandas()
    if msvc is not None:
        # I token MSVC valgono come valori speciali (NA), come "nan"/"inf".
        series = series.where(~msvc.to_numpy(zero_copy_only=False), "nan")
    converted, column_report = clean_column(series, suggestion, apply=True)
    if converted is None:
        return None, column_report
    return converted.array, column_report


def to_float_array(numbers: "pa.ChunkedArray") -> pd.arrays.FloatingArray:
    combined = numbers.combine_chunks() if isinstance(numbers, pa.ChunkedArray) else numbers
    mask = np.array(combined.is_null().to_numpy(zero_copy_only=False), dtype=bool)
    # NB: niente fill_null(): con pyarrow 21 `fill_null(...).to_numpy()` restituisce
    # valori errati. to_numpy() mette NaN sui null, poi la maschera li rende NA.
    values = np.array(combined.to_numpy(zero_copy_only=False), dtype="float64")
    values[mask] = 0.0
    return pd.arrays.FloatingArray(values, mask)


def _rows_with_lost_digits(
    table: "pa.Table", numeric_positions: List[int], candidate_rows: np.ndarray
) -> List[int]:
    """Righe (1-based, max 50) che avevano cifre ma sono diventate tutte NA.

    Stesso risultato di csv_cleaner._detect_all_nan_rows, ma controlla le cifre
    solo sulle righe già tutte NA invece che sull'intero file.
    """
    found: List[int] = []
    for start in range(0, len(candidate_rows), ROWS_ALL_NAN_BATCH):
        batch = candidate_rows[start:start + ROWS_ALL_NAN_BATCH]
        indices = pa.array(batch)
        had_digit = np.zeros(len(batch), dtype=bool)
        for pos in numeric_positions:
            values = pc.take(table.column(pos), indices)
            had_digit |= pc.match_substring_regex(values, r"\d").to_numpy(zero_copy_only=False)
        found.extend(int(i) + 1 for i in batch[had_digit])
        if len(found) >= 50:
            break
    return found[:50]


# ---------------------------------------------------------------------------
# Fallback e utilità
# ---------------------------------------------------------------------------
def _load_legacy(
    path: Path,
    meta: Dict[str, Any],
    apply_cleaning: bool,
    decimal: Optional[str],
    thousands: Optional[str],
    timings: Dict[str, float],
) -> IngestResult:
    from .loader import load_csv as load_csv_standard

    t0 = time.perf_counter()
    df, report = load_csv_standard(
        str(path),
        encoding=meta.get("encoding"),
        delimiter=meta.get("delimiter"),
        header=meta.get("header"),
        apply_cleaning=apply_cleaning,
        return_details=True,
        decimal=decimal,
        thousands=thousands,
    )
    timings["read_s"] = time.perf_counter() - t0
    timings["total_s"] = sum(timings.values())
    return IngestResult(df=df, report=report, meta=meta, timings=timings, engine="legacy")


def _progress(
    callback: Optional[ProgressCallback], phase: str, current: int, total: int, message: str
) -> None:
    if callback is None:
        return
    from .loader_optimized import LoadProgress

    progress = LoadProgress()
    progress.update(phase, current, total, message)
    callback(progress)
