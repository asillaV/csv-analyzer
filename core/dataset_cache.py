"""
Cache persistente dei CSV convertiti in Parquet (Parte 1, F2).

Un CSV viene convertito una sola volta, in streaming a blocchi (in RAM resta un
blocco alla volta, non l'intero file), e salvato in `user_data_dir()/cache`.
Le aperture successive leggono dal Parquet solo le colonne richieste.

Il risultato di `CachedDataset.load()` è identico a `core.ingest.load()`:
stesse colonne, dtype, valori, NA e `CleaningReport`.

Struttura di una voce:
    cache/<chiave>/main.parquet   colonne convertite dal percorso veloce (float64)
    cache/<chiave>/extra.parquet  colonne rilette e pulite con la logica storica
                                  (migliaia, valute, testo, colonne vuote)
    cache/<chiave>/meta.json      nomi, posizione di ogni colonna, meta, report

La chiave dipende da percorso, dimensione, data di modifica e hash veloce del
file (oppure dall'hash del contenuto per i file caricati via upload), dalle
opzioni di pulizia e da `ingest.INGEST_VERSION`.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from . import ingest
from .analyzer import analyze_csv
from .csv_cleaner import CleaningReport, ColumnReport, FormatSuggestion, clean_column
from .logger import LogManager
from .paths import user_data_dir

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    _ARROW_OK = True
except Exception:  # pragma: no cover - pyarrow è nei requirements
    _ARROW_OK = False

log = LogManager("dataset_cache").get_logger()

CACHE_FORMAT = 1
DEFAULT_MAX_GB = 5.0
HASH_SPAN = 4 << 20
# Blocchi piccoli in streaming: picco di RAM basso senza perdere velocità.
STREAM_BLOCK_SIZE = 16 << 20
# Stime per decidere se un DataFrame completo sta in RAM.
FLOAT_BYTES_PER_VALUE = 9  # Float64 nullable: 8 byte + maschera
STRING_BYTES_PER_VALUE = 60  # stringa Python media di un CSV di misure
MAIN_FILE = "main.parquet"
EXTRA_FILE = "extra.parquet"
META_FILE = "meta.json"

ProgressCallback = Callable[[Any], None]


@dataclass
class CachedDataset:
    """Un CSV convertito e salvato in cache: si caricano solo le colonne che servono."""

    directory: Path
    columns: List[str]
    meta: Dict[str, Any]
    report: CleaningReport
    num_rows: int
    from_cache: bool = False
    timings: Dict[str, float] = field(default_factory=dict)
    _storage: Dict[str, str] = field(default_factory=dict, repr=False)
    _kinds: Dict[str, str] = field(default_factory=dict, repr=False)
    _internal: Dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def numeric_columns(self) -> List[str]:
        return list(self.report.numeric_columns)

    def estimated_memory_bytes(self, columns: Optional[Sequence[str]] = None) -> int:
        """Stima della RAM del DataFrame con queste colonne (tutte se None)."""
        wanted = self.columns if columns is None else columns
        per_row = sum(
            FLOAT_BYTES_PER_VALUE if self._kinds[c] == "float" else STRING_BYTES_PER_VALUE for c in wanted
        )
        return int(per_row * self.num_rows)

    def load(self, columns: Optional[Sequence[str]] = None, rows: Optional[int] = None) -> pd.DataFrame:
        """DataFrame con le colonne richieste (tutte se None), nell'ordine richiesto.

        `rows`: solo le prime N righe (anteprima), senza leggere il resto del file.
        """
        wanted = list(self.columns if columns is None else columns)
        unknown = [c for c in wanted if c not in self._storage]
        if unknown:
            raise KeyError(f"Colonne non presenti: {unknown}")
        values: Dict[str, Any] = {}
        for storage, filename in (("main", MAIN_FILE), ("extra", EXTRA_FILE)):
            names = [c for c in wanted if self._storage[c] == storage]
            if not names:
                continue
            internal = [self._internal[c] for c in names]
            if rows is None:
                table = pq.read_table(self.directory / filename, columns=internal)
            else:
                table = _read_head(self.directory / filename, internal, rows)
            for name in names:
                column = table.column(self._internal[name])
                if self._kinds[name] == "float":
                    values[name] = ingest.to_float_array(column)
                else:
                    values[name] = column.to_pandas()
        df = pd.DataFrame({i: values[name] for i, name in enumerate(wanted)}, copy=False)
        df.columns = wanted
        if len(wanted) == 0:
            n = self.num_rows if rows is None else min(rows, self.num_rows)
            df = pd.DataFrame(index=pd.RangeIndex(n))
        return df


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def cache_root(directory: Optional[Path] = None) -> Path:
    root = Path(directory) if directory is not None else user_data_dir() / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def open_dataset(
    path: str | Path,
    *,
    apply_cleaning: bool = True,
    decimal: Optional[str] = None,
    thousands: Optional[str] = None,
    content_key: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    directory: Optional[Path] = None,
    max_bytes: Optional[int] = None,
) -> CachedDataset:
    """Apre un CSV dalla cache, convertendolo prima se non è ancora presente.

    `content_key`: identificativo del contenuto (es. SHA-1 dei byte caricati via
    upload) da usare al posto di percorso e data di modifica.
    `max_bytes`: limite della cache; dopo una nuova conversione vengono
    eliminate le voci usate meno di recente.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    root = cache_root(directory)
    key = dataset_key(path, apply_cleaning, decimal, thousands, content_key)
    entry = root / key

    cached = _read_entry(entry)
    if cached is not None:
        _touch(entry)
        log.info("Cache hit per %s (%s)", path.name, key)
        return cached
    if entry.exists():
        # Voce incompleta o illeggibile (es. conversione interrotta): si rifà.
        shutil.rmtree(entry, ignore_errors=True)

    t0 = time.perf_counter()
    tmp = root / f"{key}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    tmp.mkdir(parents=True)
    try:
        _build_entry(path, tmp, apply_cleaning, decimal, thousands, meta, progress_callback)
        _write_source(tmp, path, content_key, time.perf_counter() - t0)
        if entry.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            _rename_with_retry(tmp, entry)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    if max_bytes is not None:
        enforce_limit(max_bytes, directory=root, keep={key})
    dataset = _read_entry(entry)
    assert dataset is not None
    dataset.from_cache = False
    dataset.timings["build_s"] = time.perf_counter() - t0
    return dataset


def load_for_app(
    path: str | Path,
    apply_cleaning: bool = True,
    content_key: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> tuple[pd.DataFrame, CleaningReport, Dict[str, Any]]:
    """Caricamento completo tramite cache, con il limite di spazio dalle impostazioni.

    Restituisce (df, report, meta) come `analyze_csv` + `loader.load_csv`.
    """
    dataset = open_dataset(
        path,
        apply_cleaning=apply_cleaning,
        content_key=content_key,
        progress_callback=progress_callback,
        max_bytes=max_bytes_from_config(),
    )
    return dataset.load(), dataset.report, dict(dataset.meta)


def dataset_key(
    path: Path,
    apply_cleaning: bool = True,
    decimal: Optional[str] = None,
    thousands: Optional[str] = None,
    content_key: Optional[str] = None,
) -> str:
    stat = path.stat()
    if content_key:
        source = {"content": content_key, "size": stat.st_size}
    else:
        source = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "fast_hash": _fast_hash(path, stat.st_size),
        }
    payload = {
        "source": source,
        "cache_format": CACHE_FORMAT,
        "ingest_version": ingest.INGEST_VERSION,
        "apply_cleaning": apply_cleaning,
        "decimal": decimal,
        "thousands": thousands,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def cache_size(directory: Optional[Path] = None) -> int:
    return sum(f.stat().st_size for f in cache_root(directory).rglob("*") if f.is_file())


def clear_cache(directory: Optional[Path] = None) -> int:
    """Svuota la cache; restituisce i byte liberati."""
    root = cache_root(directory)
    freed = cache_size(root)
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
    return freed


def enforce_limit(max_bytes: int, directory: Optional[Path] = None, keep: Iterable[str] = ()) -> int:
    """Elimina le voci usate meno di recente finché la cache sta sotto `max_bytes`."""
    root = cache_root(directory)
    keep = set(keep)
    entries = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
        meta_file = child / META_FILE
        last_used = meta_file.stat().st_mtime if meta_file.exists() else 0.0
        entries.append((last_used, child, size))
    total = sum(size for _, _, size in entries)
    freed = 0
    for _, child, size in sorted(entries, key=lambda e: e[0]):
        if total <= max_bytes:
            break
        if child.name in keep:
            continue
        shutil.rmtree(child, ignore_errors=True)
        total -= size
        freed += size
    return freed


def max_bytes_from_config(config: Optional[Dict[str, Any]] = None) -> int:
    """Limite della cache da `performance.cache_max_gb` (default 5 GB)."""
    gb = DEFAULT_MAX_GB
    try:
        if config is None:
            from .settings import effective_config

            config = effective_config()
        gb = float(config.get("performance", {}).get("cache_max_gb", DEFAULT_MAX_GB))
    except Exception:
        pass
    return int(max(gb, 0.0) * 1024**3)


# ---------------------------------------------------------------------------
# Conversione in streaming
# ---------------------------------------------------------------------------
def _build_entry(
    path: Path,
    target: Path,
    apply_cleaning: bool,
    decimal: Optional[str],
    thousands: Optional[str],
    meta: Optional[Dict[str, Any]],
    progress_callback: Optional[ProgressCallback],
) -> None:
    meta = dict(meta) if meta is not None else dict(analyze_csv(str(path)))
    if _ARROW_OK:
        try:
            with _system_memory_pool():
                _build_streaming(path, target, meta, decimal, thousands, progress_callback, apply_cleaning)
            return
        except ingest.InvalidText:
            encoding = (meta.get("encoding") or "utf-8").lower().replace("-", "")
            if encoding not in ("utf8", "utf8sig"):
                log.info("Decodifica fallita per %s: conversione in memoria", path.name)
            else:
                meta["encoding"] = ingest.FALLBACK_ENCODING
                _clear_dir(target)
                try:
                    with _system_memory_pool():
                        _build_streaming(path, target, meta, decimal, thousands, progress_callback, apply_cleaning)
                    return
                except (ingest.FallbackToLegacy, ingest.InvalidText):
                    pass
        except ingest.FallbackToLegacy as reason:
            log.info("Conversione in streaming non applicabile a %s (%s)", path.name, reason)
        _clear_dir(target)
    result = ingest.load(
        path,
        meta=meta,
        apply_cleaning=apply_cleaning,
        decimal=decimal,
        thousands=thousands,
        progress_callback=progress_callback,
    )
    _write_from_frame(target, result.df, result.report, result.meta)


def _build_streaming(
    path: Path,
    target: Path,
    meta: Dict[str, Any],
    decimal: Optional[str],
    thousands: Optional[str],
    progress_callback: Optional[ProgressCallback],
    apply_cleaning: bool = True,
) -> None:
    """Conversione a blocchi. Con `apply_cleaning=False` tutte le colonne restano
    testo originale (come il loader standard senza correzione) e serve solo il report."""
    delimiter = meta.get("delimiter") or ","
    header = meta.get("header")
    encoding = meta.get("encoding") or "utf-8"
    try:
        names = ingest.column_names(path, encoding, delimiter, header, meta)
    except UnicodeDecodeError as exc:
        raise ingest.InvalidText(str(exc)) from exc
    n = len(names)
    internal = [f"c{i}" for i in range(n)]

    suggestion: Optional[FormatSuggestion] = None
    candidates = np.zeros(n, dtype=np.int64)
    deferred: Set[int] = set()
    main_positions: Optional[List[int]] = None
    msvc_rows: Dict[int, List[tuple]] = {}  # colonna -> [(prima riga, righe, maschera compressa)]
    writer = None
    reader = None
    num_rows = 0
    file_size = max(1, path.stat().st_size)
    counter = ingest.InvalidRowCounter()
    workers = max(1, min(n, os.cpu_count() or 4, ingest.MAX_CONVERT_WORKERS))

    try:
        reader = _ChunkedCsvReader(path, encoding, delimiter, header, n, counter)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for batch, suggestion in _batches_with_suggestion(reader, names, decimal, thousands):
                casts = list(pool.map(lambda i: ingest.fast_cast(batch.column(i), suggestion), range(n)))
                if main_positions is None:
                    if apply_cleaning:
                        main_positions = [
                            i for i, c in enumerate(casts) if not (c.candidate_count and c.numbers is None)
                        ]
                        column_type = pa.float64()
                    else:
                        main_positions = list(range(n))
                        column_type = pa.string()
                    if main_positions:
                        schema = pa.schema([(internal[i], column_type) for i in main_positions])
                        writer = pq.ParquetWriter(target / MAIN_FILE, schema, compression="zstd")
                arrays = []
                for i in main_positions:
                    cast = casts[i]
                    if cast.candidate_count and cast.numbers is None:
                        deferred.add(i)
                    if not apply_cleaning:
                        arrays.append(batch.column(i))
                    elif i in deferred or cast.numbers is None:
                        arrays.append(pa.nulls(batch.num_rows, pa.float64()))
                    else:
                        arrays.append(cast.numbers)
                for i, cast in enumerate(casts):
                    candidates[i] += cast.candidate_count
                    if cast.candidate_count and cast.numbers is None:
                        deferred.add(i)
                    if cast.msvc is not None and cast.numbers is not None:
                        mask = cast.msvc.to_numpy(zero_copy_only=False)
                        if mask.any():
                            # 1 bit per riga: resta piccolo anche con milioni di token.
                            msvc_rows.setdefault(i, []).append((num_rows, len(mask), np.packbits(mask)))
                if writer is not None:
                    writer.write_table(pa.Table.from_arrays(arrays, schema=writer.schema))
                del casts, arrays
                pa.default_memory_pool().release_unused()
                num_rows += batch.num_rows
                done = min(100, int(100 * reader.position() / file_size))
                ingest._progress(progress_callback, "loading", done, 100, f"{num_rows:,} righe convertite")
    except (UnicodeDecodeError, pa.ArrowInvalid) as exc:
        raise ingest.arrow_errors(exc) from exc
    finally:
        if writer is not None:
            writer.close()
        if reader is not None:
            reader.close()

    counter.check()
    if counter.too_many and counter.too_many >= ingest.MAX_SKIPPED_ROWS_RATIO * (num_rows + counter.too_many):
        raise ingest.FallbackToLegacy(f"{counter.too_many} righe con troppi campi")
    if suggestion is None:
        raise ingest.FallbackToLegacy("file senza righe di dati")
    if not apply_cleaning:
        _finish_raw(target, names, internal, meta, encoding, suggestion, candidates, deferred, num_rows)
        return

    # Colonne da rileggere: non convertibili in modo pulito o senza alcun valore.
    reread = sorted(deferred | {i for i in range(n) if candidates[i] == 0})
    reports: Dict[str, ColumnReport] = {}
    kinds: Dict[str, str] = {}
    storage: Dict[str, str] = {}
    numeric: List[str] = []
    extra_had_digit = np.zeros(num_rows, dtype=bool)
    extra_arrays: List[Any] = []
    extra_names: List[str] = []

    if reread:
        ingest._progress(progress_callback, "cleaning", 0, len(reread), "Pulizia colonne particolari...")
        raw_columns = _read_columns(path, encoding, delimiter, header, n, reread)
        for i in reread:
            raw = raw_columns[i]
            name = names[i]
            if candidates[i] == 0:
                reports[name] = ingest.empty_column_report(num_rows)
                extra_arrays.append(raw)
                kinds[name] = "string"
            else:
                stripped = pc.utf8_trim_whitespace(raw)
                msvc = None
                if pc.any(pc.match_substring(stripped, "#")).as_py():
                    msvc = pc.match_substring_regex(stripped, ingest.MSVC_SPECIAL_RE)
                converted, column_report = ingest.convert_column_legacy(raw, msvc, suggestion)
                reports[name] = column_report
                if converted is None:
                    extra_arrays.append(raw)
                    kinds[name] = "string"
                else:
                    extra_arrays.append(pa.array(converted, type=pa.float64()))
                    kinds[name] = "float"
                    extra_had_digit |= pc.match_substring_regex(raw, r"\d").to_numpy(zero_copy_only=False)
            storage[name] = "extra"
            extra_names.append(internal[i])
        pq.write_table(
            pa.table([_combine(a) for a in extra_arrays], names=extra_names),
            target / EXTRA_FILE,
            compression="zstd",
        )

    for i, name in enumerate(names):
        if name in storage:
            if kinds[name] == "float":
                numeric.append(name)
            continue
        storage[name] = "main"
        kinds[name] = "float"
        numeric.append(name)
        reports[name] = ingest.fast_column_report(num_rows, int(candidates[i]))

    # Le cifre contano solo nelle colonne numeriche finali (come nel loader standard);
    # per quelle rilette, extra_had_digit è già calcolato sul testo completo.
    had_digit = extra_had_digit
    for i, chunks in msvc_rows.items():
        if storage[names[i]] != "main":
            continue
        for offset, length, packed in chunks:
            had_digit[offset:offset + length] |= np.unpackbits(packed, count=length).astype(bool)
    rows_all_nan = _rows_all_nan(target, names, internal, storage, kinds, num_rows, had_digit)

    report = CleaningReport(
        suggestion=suggestion,
        applied_cleaning=True,
        columns={name: reports[name] for name in names},
        numeric_columns=numeric,
        rows_all_nan_after_clean=rows_all_nan,
        warnings=ingest.format_warnings(suggestion),
    )
    meta = dict(meta)
    meta["encoding"] = encoding
    _write_meta(target, names, internal, storage, kinds, meta, report, num_rows, engine="arrow-stream")


def _finish_raw(
    target: Path,
    names: List[str],
    internal: List[str],
    meta: Dict[str, Any],
    encoding: str,
    suggestion: FormatSuggestion,
    candidates: np.ndarray,
    deferred: Set[int],
    num_rows: int,
) -> None:
    """Report della modalità senza correzione: tutte le colonne restano testo."""
    reports: Dict[str, ColumnReport] = {}
    raw_table = None
    if deferred:
        raw_table = pq.read_table(target / MAIN_FILE, columns=[internal[i] for i in sorted(deferred)])
    for i, name in enumerate(names):
        if candidates[i] == 0:
            reports[name] = ingest.empty_column_report(num_rows)
        elif i in deferred:
            raw = raw_table.column(internal[i])
            stripped = pc.utf8_trim_whitespace(raw)
            series = raw.to_pandas()
            if pc.any(pc.match_substring(stripped, "#")).as_py():
                msvc = pc.match_substring_regex(stripped, ingest.MSVC_SPECIAL_RE)
                series = series.where(~msvc.to_numpy(zero_copy_only=False), "nan")
            _, column_report = clean_column(series, suggestion, apply=False)
            column_report.dtype = "object"
            reports[name] = column_report
        else:
            column_report = ingest.fast_column_report(num_rows, int(candidates[i]))
            column_report.dtype = "object"
            column_report.applied = False
            reports[name] = column_report
    report = CleaningReport(
        suggestion=suggestion,
        applied_cleaning=False,
        columns=reports,
        numeric_columns=[],
        rows_all_nan_after_clean=[],
        warnings=ingest.format_warnings(suggestion),
    )
    storage = {name: "main" for name in names}
    kinds = {name: "string" for name in names}
    meta = dict(meta)
    meta["encoding"] = encoding
    _write_meta(target, names, internal, storage, kinds, meta, report, num_rows, engine="arrow-stream-raw")


class _ChunkedCsvReader:
    """Legge il CSV a blocchi di testo UTF-8 allineati a fine riga, uno alla volta.

    Né `pacsv.open_csv` su un percorso con `encoding` (la transcodifica interna
    accumulava 1,3-1,9 GB sul file reale UTF-16) né su un file Python (Arrow lo
    legge in anticipo quasi per intero) tengono la memoria sotto controllo: qui
    in RAM c'è sempre un solo blocco, e l'avanzamento è esatto.

    Limite: un "a capo" dentro un campo tra virgolette spezzato tra due blocchi
    produce una riga malformata, e la conversione ripiega sul percorso in memoria.
    """

    def __init__(
        self,
        path: Path,
        encoding: str,
        delimiter: str,
        header: Optional[int],
        n_columns: int,
        counter,
        chunk_bytes: Optional[int] = None,
    ) -> None:
        self._file = open(path, "rb")
        plain = ingest.is_plain_utf8(encoding)
        self._decoder = None if plain else codecs.getincrementaldecoder(encoding)("strict")
        self._chunk_bytes = chunk_bytes or STREAM_BLOCK_SIZE
        self._first_options = ingest.csv_options("utf8", delimiter, header, n_columns, counter, use_threads=False)
        self._next_options = ingest.csv_options("utf8", delimiter, None, n_columns, counter, use_threads=False)

    def position(self) -> int:
        try:
            return self._file.tell()
        except (OSError, ValueError):
            return 0

    def close(self) -> None:
        self._file.close()

    def _text_blocks(self):
        carry = b""
        eof = False
        while not eof:
            raw = self._file.read(self._chunk_bytes)
            eof = not raw
            if self._decoder is not None:
                raw = self._decoder.decode(raw, final=eof).encode("utf-8")
            data = carry + raw
            if eof:
                carry = b""
                if data:
                    yield data
                break
            cut = data.rfind(b"\n")
            if cut < 0:
                carry = data
                continue
            carry = data[cut + 1:]
            yield data[:cut + 1]

    def __iter__(self):
        first = True
        for block in self._text_blocks():
            options = self._first_options if first else self._next_options
            first = False
            table = pacsv.read_csv(pa.BufferReader(block), **options)
            for batch in table.to_batches():
                if batch.num_rows:
                    yield batch


def _batches_with_suggestion(batches, names: List[str], decimal: Optional[str], thousands: Optional[str]):
    """Restituisce (batch, formato stimato) dopo aver letto abbastanza righe per la stima.

    La stima usa le prime `rows_per_column` righe come `ingest.suggest_format`:
    se i primi blocchi sono più corti vengono accumulati prima di decidere.
    """
    rows_needed = max(10, (ingest.MAX_SAMPLE_VALUES * 4) // max(1, len(names)))
    pending: List[Any] = []
    suggestion: Optional[FormatSuggestion] = None
    for batch in batches:
        if suggestion is not None:
            yield batch, suggestion
            continue
        pending.append(batch)
        if sum(b.num_rows for b in pending) >= rows_needed:
            suggestion = ingest.suggest_format(pa.Table.from_batches(pending), names, decimal, thousands)
            yield from ((b, suggestion) for b in pending)
            pending = []
    if pending:
        suggestion = ingest.suggest_format(pa.Table.from_batches(pending), names, decimal, thousands)
        yield from ((b, suggestion) for b in pending)


def _read_columns(
    path: Path, encoding: str, delimiter: str, header: Optional[int], n: int, positions: List[int]
) -> Dict[int, Any]:
    counter = ingest.InvalidRowCounter()
    options = ingest.csv_options(encoding, delimiter, header, n, counter, include=positions)
    try:
        table = pacsv.read_csv(str(path), **options)
    except (UnicodeDecodeError, pa.ArrowInvalid) as exc:
        raise ingest.arrow_errors(exc) from exc
    return {i: table.column(f"c{i}") for i in positions}


def _rows_all_nan(
    target: Path,
    names: List[str],
    internal: List[str],
    storage: Dict[str, str],
    kinds: Dict[str, str],
    num_rows: int,
    had_digit: np.ndarray,
) -> List[int]:
    """Righe (1-based, max 50) con cifre nel CSV ma tutte NA dopo la pulizia."""
    numeric = [i for i, name in enumerate(names) if kinds[name] == "float"]
    if not numeric or not had_digit.any():
        return []
    all_nan = np.ones(num_rows, dtype=bool)
    for storage_name, filename in (("main", MAIN_FILE), ("extra", EXTRA_FILE)):
        cols = [internal[i] for i in numeric if storage[names[i]] == storage_name]
        if not cols:
            continue
        offset = 0
        # File chiuso subito: su Windows una cartella con file aperti non si rinomina.
        with pq.ParquetFile(target / filename) as parquet:
            for batch in parquet.iter_batches(columns=cols):
                block = np.ones(batch.num_rows, dtype=bool)
                for column in batch.columns:
                    block &= column.is_null().to_numpy(zero_copy_only=False)
                all_nan[offset:offset + batch.num_rows] &= block
                offset += batch.num_rows
    return [int(i) + 1 for i in np.flatnonzero(all_nan & had_digit)[:50]]


# ---------------------------------------------------------------------------
# Scrittura/lettura di una voce
# ---------------------------------------------------------------------------
def _write_from_frame(target: Path, df: pd.DataFrame, report: CleaningReport, meta: Dict[str, Any]) -> None:
    """Salva un DataFrame già caricato in memoria (percorsi di ripiego)."""
    names = [str(c) for c in df.columns]
    internal = [f"c{i}" for i in range(len(names))]
    numeric = set(report.numeric_columns)
    arrays, kinds, storage = [], {}, {}
    for i, name in enumerate(names):
        series = df.iloc[:, i]
        if name in numeric:
            arrays.append(pa.array(series.astype("Float64").array, type=pa.float64()))
            kinds[name] = "float"
        else:
            arrays.append(pa.array(series.astype("string").array, type=pa.string()))
            kinds[name] = "string"
        storage[name] = "main"
    pq.write_table(pa.table(arrays, names=internal), target / MAIN_FILE, compression="zstd")
    _write_meta(target, names, internal, storage, kinds, meta, report, len(df), engine="memory")


def _write_meta(
    target: Path,
    names: List[str],
    internal: List[str],
    storage: Dict[str, str],
    kinds: Dict[str, str],
    meta: Dict[str, Any],
    report: CleaningReport,
    num_rows: int,
    engine: str,
) -> None:
    payload = {
        "cache_format": CACHE_FORMAT,
        "ingest_version": ingest.INGEST_VERSION,
        "engine": engine,
        "names": names,
        "internal": internal,
        "storage": [storage[n] for n in names],
        "kinds": [kinds[n] for n in names],
        "num_rows": num_rows,
        "meta": _jsonable(meta),
        "report": report.to_dict(),
    }
    (target / META_FILE).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_source(target: Path, path: Path, content_key: Optional[str], build_s: float) -> None:
    meta_file = target / META_FILE
    payload = json.loads(meta_file.read_text(encoding="utf-8"))
    payload["source"] = {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "content_key": content_key,
        "created": time.time(),
        "build_s": build_s,
    }
    meta_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _read_entry(entry: Path) -> Optional[CachedDataset]:
    meta_file = entry / META_FILE
    if not meta_file.exists():
        return None
    try:
        payload = json.loads(meta_file.read_text(encoding="utf-8"))
        if payload.get("cache_format") != CACHE_FORMAT or payload.get("ingest_version") != ingest.INGEST_VERSION:
            return None
        names = payload["names"]
        return CachedDataset(
            directory=entry,
            columns=list(names),
            meta=payload["meta"],
            report=report_from_dict(payload["report"]),
            num_rows=int(payload["num_rows"]),
            from_cache=True,
            _storage=dict(zip(names, payload["storage"])),
            _kinds=dict(zip(names, payload["kinds"])),
            _internal=dict(zip(names, payload["internal"])),
        )
    except Exception as exc:
        log.warning("Voce di cache non leggibile %s: %s", entry.name, exc)
        return None


def report_from_dict(data: Dict[str, Any]) -> CleaningReport:
    return CleaningReport(
        suggestion=FormatSuggestion(**data["suggestion"]),
        applied_cleaning=bool(data["applied_cleaning"]),
        columns={name: ColumnReport(**col) for name, col in data["columns"].items()},
        numeric_columns=list(data["numeric_columns"]),
        rows_all_nan_after_clean=list(data["rows_all_nan_after_clean"]),
        warnings=list(data["warnings"]),
    )


# ---------------------------------------------------------------------------
# Utilità
# ---------------------------------------------------------------------------
@contextmanager
def _system_memory_pool():
    """Allocatore di sistema durante la conversione.

    L'allocatore predefinito di Arrow (mimalloc su Windows) trattiene la memoria
    dei blocchi già scritti: con quello di sistema il picco scende di ~2,5 volte
    (ipraw25k: 1,67 GB -> 0,65 GB) a parità di tempo.
    """
    previous = pa.default_memory_pool()
    pa.set_memory_pool(pa.system_memory_pool())
    try:
        yield
    finally:
        pa.set_memory_pool(previous)


def _fast_hash(path: Path, size: int) -> str:
    digest = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        digest.update(f.read(HASH_SPAN))
        if size > HASH_SPAN:
            f.seek(max(HASH_SPAN, size - HASH_SPAN))
            digest.update(f.read(HASH_SPAN))
    return digest.hexdigest()


def _read_head(filename: Path, columns: List[str], rows: int) -> "pa.Table":
    batches = []
    remaining = rows
    with pq.ParquetFile(filename) as parquet:
        for batch in parquet.iter_batches(batch_size=max(1, rows), columns=columns):
            batches.append(batch.slice(0, remaining))
            remaining -= min(remaining, batch.num_rows)
            if remaining <= 0:
                break
    if not batches:
        return pq.read_table(filename, columns=columns).slice(0, 0)
    return pa.Table.from_batches(batches)


def _combine(array: Any) -> Any:
    return array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array


def _rename_with_retry(source: Path, target: Path, attempts: int = 5) -> None:
    """Rinomina la cartella temporanea; su Windows un antivirus può bloccarla per un attimo."""
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.2 * (attempt + 1))


def _touch(entry: Path) -> None:
    try:
        os.utime(entry / META_FILE, None)
    except OSError:
        pass


def _clear_dir(target: Path) -> None:
    for child in target.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        return str(value)
