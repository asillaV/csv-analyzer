"""Test della cache Parquet (Parte 1, F2): identità con core.ingest e gestione della cache."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

from core import dataset_cache, ingest
from tests.fixtures.csv_corpus import CORPUS, write_corpus
from tests.parity import LoadOutcome, compare_outcomes

LOCAL_CSV_DIR = Path(__file__).resolve().parents[1] / "tests_csv"
LOCAL_MAX_MB = 50


@pytest.fixture(scope="module")
def corpus_paths(tmp_path_factory) -> Dict[str, Path]:
    return write_corpus(tmp_path_factory.mktemp("cache_corpus"))


def _assert_same_as_ingest(path: Path, cache_dir: Path) -> dataset_cache.CachedDataset:
    expected = ingest.load(path)
    dataset = dataset_cache.open_dataset(path, directory=cache_dir)
    df = dataset.load()
    diffs = compare_outcomes(
        LoadOutcome(df=expected.df, report=expected.report),
        LoadOutcome(df=df, report=dataset.report),
    )
    assert not diffs, f"{path.name}: " + "; ".join(diffs)
    assert dataset.report.to_dict() == expected.report.to_dict()
    assert dataset.num_rows == len(expected.df)
    assert dataset.meta.get("encoding") == expected.meta.get("encoding")
    return dataset


@pytest.mark.integration
@pytest.mark.parametrize("name", list(CORPUS))
def test_cache_matches_ingest_on_corpus(corpus_paths, tmp_path, name):
    _assert_same_as_ingest(corpus_paths[name], tmp_path / "cache")


@pytest.mark.integration
@pytest.mark.parametrize("name", list(CORPUS))
def test_cache_matches_ingest_with_many_batches(corpus_paths, tmp_path, monkeypatch, name):
    """Blocchi minuscoli: ogni file viene diviso in molti batch."""
    monkeypatch.setattr(dataset_cache, "STREAM_BLOCK_SIZE", 2048)
    _assert_same_as_ingest(corpus_paths[name], tmp_path / "cache")


def _local_csvs() -> list[Path]:
    if not LOCAL_CSV_DIR.is_dir():
        return []
    return sorted(p for p in LOCAL_CSV_DIR.glob("*.csv") if p.stat().st_size <= LOCAL_MAX_MB * 1024 * 1024)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("path", _local_csvs(), ids=lambda p: p.name)
def test_cache_matches_ingest_on_local_files(path, tmp_path):
    _assert_same_as_ingest(path, tmp_path / "cache")


# ---------------------------------------------------------------------------
# Casi specifici della conversione in streaming
# ---------------------------------------------------------------------------
def test_column_becoming_irregular_late_is_reread(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_cache, "STREAM_BLOCK_SIZE", 1024)
    rows = [f"{i};{i * 0.5:.2f}".replace(".", ",") for i in range(400)]
    rows += [f'{i};"€ 1.234,{i % 100:02d}"' for i in range(400, 420)]
    path = tmp_path / "late.csv"
    path.write_text("n;v\n" + "\n".join(rows) + "\n", encoding="utf-8")
    dataset = _assert_same_as_ingest(path, tmp_path / "cache")
    assert dataset._storage["v"] == "extra"
    assert dataset._storage["n"] == "main"


def test_text_and_empty_columns_kept_as_strings(tmp_path):
    rows = [f"Linea {i % 3},{i},,{i * 1.5}" for i in range(50)]
    path = tmp_path / "text.csv"
    path.write_text("Line,n,vuota,v\n" + "\n".join(rows) + "\n", encoding="utf-8")
    dataset = _assert_same_as_ingest(path, tmp_path / "cache")
    df = dataset.load()
    assert df["Line"].iloc[1] == "Linea 1"
    assert dataset.numeric_columns == ["n", "v"]


def test_cp1252_streaming_retry(corpus_paths, tmp_path):
    dataset = _assert_same_as_ingest(corpus_paths["cp1252_accents"], tmp_path / "cache")
    assert dataset.meta["encoding"] == "cp1252"


def test_fallback_rows_with_fewer_fields(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("a,b,c\n1,2,3\n4,5\n6,7,8\n", encoding="utf-8")
    dataset = _assert_same_as_ingest(path, tmp_path / "cache")
    assert json.loads((dataset.directory / dataset_cache.META_FILE).read_text("utf-8"))["engine"] == "memory"


def test_apply_cleaning_false(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    dataset = dataset_cache.open_dataset(path, apply_cleaning=False, directory=tmp_path / "cache")
    expected = ingest.load(path, apply_cleaning=False)
    assert dataset.report.numeric_columns == expected.report.numeric_columns == []
    assert dataset.load()["a"].tolist() == ["1", "3"]


# ---------------------------------------------------------------------------
# Gestione della cache
# ---------------------------------------------------------------------------
def _csv(tmp_path: Path, name: str = "data.csv", rows: int = 100) -> Path:
    path = tmp_path / name
    path.write_text("t,a,b\n" + "\n".join(f"{i},{i * 0.1:.3f},{i * 2}" for i in range(rows)) + "\n", encoding="utf-8")
    return path


def test_second_open_hits_cache(tmp_path, monkeypatch):
    path = _csv(tmp_path)
    first = dataset_cache.open_dataset(path, directory=tmp_path / "cache")
    assert not first.from_cache

    def no_build(*_args, **_kwargs):
        raise AssertionError("non deve riconvertire")

    monkeypatch.setattr(dataset_cache, "_build_entry", no_build)
    second = dataset_cache.open_dataset(path, directory=tmp_path / "cache")
    assert second.from_cache
    pd.testing.assert_frame_equal(first.load(), second.load())


def test_modified_file_is_reconverted(tmp_path):
    path = _csv(tmp_path)
    cache = tmp_path / "cache"
    first = dataset_cache.open_dataset(path, directory=cache)
    path.write_text("t,a\n0,9.5\n1,8.5\n", encoding="utf-8")
    os.utime(path, (1_000_000_000, 1_000_000_000))
    second = dataset_cache.open_dataset(path, directory=cache)
    assert second.directory != first.directory
    assert second.load()["a"].tolist() == [9.5, 8.5]


def test_content_key_ignores_path(tmp_path):
    a = _csv(tmp_path, "a.csv")
    b = tmp_path / "b.csv"
    b.write_bytes(a.read_bytes())
    key_a = dataset_cache.dataset_key(a, content_key="abc")
    key_b = dataset_cache.dataset_key(b, content_key="abc")
    assert key_a == key_b
    assert dataset_cache.dataset_key(a) != dataset_cache.dataset_key(b)


def test_options_change_key(tmp_path):
    path = _csv(tmp_path)
    assert dataset_cache.dataset_key(path) != dataset_cache.dataset_key(path, decimal=",")
    assert dataset_cache.dataset_key(path) != dataset_cache.dataset_key(path, apply_cleaning=False)


def test_load_column_subset_in_requested_order(tmp_path):
    dataset = dataset_cache.open_dataset(_csv(tmp_path), directory=tmp_path / "cache")
    df = dataset.load(["b", "t"])
    assert list(df.columns) == ["b", "t"]
    assert df["b"].iloc[3] == 6.0
    assert len(dataset.load([])) == dataset.num_rows
    with pytest.raises(KeyError):
        dataset.load(["manca"])


def test_corrupted_entry_is_rebuilt(tmp_path):
    path = _csv(tmp_path)
    cache = tmp_path / "cache"
    first = dataset_cache.open_dataset(path, directory=cache)
    (first.directory / dataset_cache.META_FILE).write_text("{non json", encoding="utf-8")
    again = dataset_cache.open_dataset(path, directory=cache)
    assert not again.from_cache
    assert again.directory == first.directory
    assert again.load()["b"].iloc[2] == 4.0


def test_unreadable_meta_is_ignored(tmp_path):
    entry = tmp_path / "entry"
    entry.mkdir()
    (entry / dataset_cache.META_FILE).write_text("{non json", encoding="utf-8")
    assert dataset_cache._read_entry(entry) is None


def test_old_ingest_version_is_ignored(tmp_path, monkeypatch):
    path = _csv(tmp_path)
    cache = tmp_path / "cache"
    first = dataset_cache.open_dataset(path, directory=cache)
    monkeypatch.setattr(ingest, "INGEST_VERSION", ingest.INGEST_VERSION + 1)
    assert dataset_cache._read_entry(first.directory) is None


def test_limit_evicts_least_recently_used(tmp_path):
    cache = tmp_path / "cache"
    old = dataset_cache.open_dataset(_csv(tmp_path, "old.csv", 3000), directory=cache)
    os.utime(old.directory / dataset_cache.META_FILE, (1_000_000_000, 1_000_000_000))
    new = dataset_cache.open_dataset(_csv(tmp_path, "new.csv", 3000), directory=cache)
    size_new = sum(f.stat().st_size for f in new.directory.rglob("*") if f.is_file())
    freed = dataset_cache.enforce_limit(size_new, directory=cache, keep={new.directory.name})
    assert freed > 0
    assert not old.directory.exists() and new.directory.exists()


def test_open_with_limit_keeps_new_entry(tmp_path):
    dataset = dataset_cache.open_dataset(_csv(tmp_path), directory=tmp_path / "cache", max_bytes=1)
    assert dataset.directory.exists()


def test_clear_cache_and_size(tmp_path):
    cache = tmp_path / "cache"
    dataset_cache.open_dataset(_csv(tmp_path), directory=cache)
    size = dataset_cache.cache_size(cache)
    assert size > 0
    assert dataset_cache.clear_cache(cache) == size
    assert dataset_cache.cache_size(cache) == 0


def test_max_bytes_from_config():
    assert dataset_cache.max_bytes_from_config({"performance": {"cache_max_gb": 2}}) == 2 * 1024**3
    assert dataset_cache.max_bytes_from_config({}) == int(dataset_cache.DEFAULT_MAX_GB * 1024**3)
    assert dataset_cache.max_bytes_from_config({"performance": {"cache_max_gb": "x"}}) == int(
        dataset_cache.DEFAULT_MAX_GB * 1024**3
    )


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset_cache.open_dataset(tmp_path / "manca.csv", directory=tmp_path / "cache")


def test_progress_callback(tmp_path):
    phases = []
    dataset_cache.open_dataset(_csv(tmp_path), directory=tmp_path / "cache", progress_callback=lambda p: phases.append(p.phase))
    assert "loading" in phases


# ---------------------------------------------------------------------------
# F3: modalità senza correzione, anteprima, stima memoria, rinomina
# ---------------------------------------------------------------------------
RAW_REFERENCE_BUGS = {"cp1252_accents", "no_header"}  # vedi test_ingest_parity.KNOWN_REFERENCE_BUGS


@pytest.mark.integration
@pytest.mark.parametrize("name", [n for n in CORPUS if n not in RAW_REFERENCE_BUGS])
def test_raw_mode_matches_standard_loader(corpus_paths, tmp_path, monkeypatch, name):
    """Senza correzione: stesso testo originale e nessuna colonna numerica, come il loader standard."""
    monkeypatch.setattr(dataset_cache, "STREAM_BLOCK_SIZE", 4096)
    path = corpus_paths[name]
    expected = ingest.load(path, apply_cleaning=False)
    dataset = dataset_cache.open_dataset(path, apply_cleaning=False, directory=tmp_path / "cache")
    diffs = compare_outcomes(
        LoadOutcome(df=expected.df, report=expected.report),
        LoadOutcome(df=dataset.load(), report=dataset.report),
    )
    assert not diffs, f"{name}: " + "; ".join(diffs)
    assert dataset.report.numeric_columns == []
    assert dataset.report.applied_cleaning is False


def test_raw_mode_fixes_reference_bugs(corpus_paths, tmp_path):
    cp = dataset_cache.open_dataset(corpus_paths["cp1252_accents"], apply_cleaning=False, directory=tmp_path / "a")
    assert cp.columns == ["misura", "pressione àèé"] and cp.num_rows == 60
    assert cp.load(["pressione àèé"]).iloc[0, 0].count(",") == 1
    nh = dataset_cache.open_dataset(corpus_paths["no_header"], apply_cleaning=False, directory=tmp_path / "b")
    assert nh.num_rows == 100
    assert nh.load(rows=1).iloc[0, 0] == "0.00"


def test_raw_mode_column_report(tmp_path):
    rows = [f'{i};"€ 1.234,{i:02d}";{i * 0.5}' for i in range(30)]
    path = tmp_path / "raw.csv"
    path.write_text("n;prezzo;v\n" + "\n".join(rows) + "\n", encoding="utf-8")
    dataset = dataset_cache.open_dataset(path, apply_cleaning=False, directory=tmp_path / "cache")
    report = dataset.report.columns
    assert report["prezzo"].candidate_numeric == 30 and report["prezzo"].applied is False
    assert report["v"].converted == 30 and report["v"].dtype == "object"
    assert dataset.load(["prezzo"])["prezzo"].iloc[3] == "€ 1.234,03"


def test_head_rows_preview(tmp_path):
    dataset = dataset_cache.open_dataset(_csv(tmp_path, rows=500), directory=tmp_path / "cache")
    head = dataset.load(rows=7)
    assert len(head) == 7
    pd.testing.assert_frame_equal(head, dataset.load().head(7))
    assert len(dataset.load([], rows=3)) == 3
    assert len(dataset.load(rows=10_000)) == 500


def test_head_rows_with_extra_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_cache, "STREAM_BLOCK_SIZE", 1024)
    rows = [f"Linea {i % 3};{i}" for i in range(200)]
    path = tmp_path / "mixed.csv"
    path.write_text("Line;n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    dataset = dataset_cache.open_dataset(path, directory=tmp_path / "cache")
    pd.testing.assert_frame_equal(dataset.load(rows=5), dataset.load().head(5))


def test_estimated_memory(tmp_path):
    path = tmp_path / "est.csv"
    words = ["alfa", "beta", "gamma"]
    path.write_text("txt,a\n" + "\n".join(f"{words[i % 3]},{i}" for i in range(100)) + "\n", encoding="utf-8")
    dataset = dataset_cache.open_dataset(path, directory=tmp_path / "cache")
    float_only = dataset.estimated_memory_bytes(["a"])
    assert float_only == 100 * dataset_cache.FLOAT_BYTES_PER_VALUE
    assert dataset.estimated_memory_bytes() == float_only + 100 * dataset_cache.STRING_BYTES_PER_VALUE


def test_rename_retries_on_permission_error(tmp_path, monkeypatch):
    source = tmp_path / "a"
    source.mkdir()
    target = tmp_path / "b"
    calls = []
    original = Path.replace

    def flaky(self, other):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("bloccato")
        return original(self, other)

    monkeypatch.setattr(Path, "replace", flaky)
    monkeypatch.setattr(dataset_cache.time, "sleep", lambda _s: None)
    dataset_cache._rename_with_retry(source, target)
    assert target.exists() and len(calls) == 3

    def always_locked(self, other):
        raise PermissionError("sempre")

    other = tmp_path / "c"
    other.mkdir()
    monkeypatch.setattr(Path, "replace", always_locked)
    with pytest.raises(PermissionError):
        dataset_cache._rename_with_retry(other, tmp_path / "d", attempts=2)


def test_msvc_rows_all_nan_entry_is_renamed(tmp_path):
    """Regressione Windows: i Parquet letti per rows_all_nan vanno chiusi prima di rinominare."""
    rows = [f"{i},1.#IO,1.#QO" if i % 5 == 0 else f"{i},{i * 0.5},{i * 0.25}" for i in range(100)]
    path = tmp_path / "msvc.csv"
    path.write_text("n,a,b\n" + "\n".join(rows) + "\n", encoding="utf-8")
    dataset = _assert_same_as_ingest(path, tmp_path / "cache")
    assert dataset.directory.exists()
    assert not [p for p in (tmp_path / "cache").iterdir() if ".tmp-" in p.name]


def test_progress_reaches_end(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_cache, "STREAM_BLOCK_SIZE", 512)
    seen = []
    dataset_cache.open_dataset(
        _csv(tmp_path, rows=400),
        directory=tmp_path / "cache",
        progress_callback=lambda p: seen.append(p.current) if p.phase == "loading" else None,
    )
    assert len(seen) > 3
    assert seen == sorted(seen) and seen[-1] == 100
