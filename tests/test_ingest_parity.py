"""
Test di parità del caricamento CSV (Parte 1, F0).

Il nuovo percorso di ingest (`core.ingest.load`, introdotto in F1) deve dare
lo stesso risultato del percorso attuale (`tests.parity.load_reference`) su
tutto il corpus generato, tranne i casi in KNOWN_REFERENCE_BUGS, dove il
loader attuale è difettoso e il nuovo deve invece rispettare le aspettative
esplicite di `test_ingest_fixes_reference_bugs`.

Finché `core.ingest` non esiste i test sul nuovo percorso vengono saltati;
quelli sul riferimento girano subito e fissano il comportamento di partenza.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.csv_corpus import CORPUS, write_corpus
from tests.parity import LoadOutcome, assert_load_equivalent, load_reference

pytestmark = pytest.mark.integration

INGEST_AVAILABLE = importlib.util.find_spec("core.ingest") is not None
needs_ingest = pytest.mark.skipif(not INGEST_AVAILABLE, reason="core.ingest non ancora implementato (F1)")

# Casi in cui il loader attuale sbaglia: il nuovo NON deve replicarli.
KNOWN_REFERENCE_BUGS: Dict[str, str] = {
    "cp1252_accents": "UnicodeDecodeError: i file non UTF non vengono aperti",
    "no_header": "la prima riga di dati viene usata come intestazione (riga persa)",
}

PARITY_CASES = [name for name in CORPUS if name not in KNOWN_REFERENCE_BUGS]

LOCAL_CSV_DIR = Path(__file__).resolve().parents[1] / "tests_csv"
LOCAL_MAX_MB = 50


@pytest.fixture(scope="module")
def corpus_paths(tmp_path_factory) -> Dict[str, Path]:
    return write_corpus(tmp_path_factory.mktemp("csv_corpus"))


def _load_new(path: Path) -> LoadOutcome:
    from core import ingest

    try:
        result = ingest.load(path)
        return LoadOutcome(df=result.df, report=result.report, meta=dict(result.meta))
    except Exception as exc:
        return LoadOutcome(error=exc)


# ---------------------------------------------------------------------------
# Comportamento di partenza del riferimento
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", PARITY_CASES)
def test_reference_loads_corpus(corpus_paths, name):
    outcome = load_reference(corpus_paths[name])
    assert outcome.error is None, f"il riferimento non carica {name}: {outcome.error!r}"
    assert len(outcome.df) > 0
    for col in outcome.report.numeric_columns:
        assert str(outcome.df[col].dtype) == "Float64", f"{name}:{col} non è Float64"


# ---------------------------------------------------------------------------
# Parità nuovo vs riferimento
# ---------------------------------------------------------------------------
@needs_ingest
@pytest.mark.parametrize("name", PARITY_CASES)
def test_ingest_parity(corpus_paths, name):
    path = corpus_paths[name]
    assert_load_equivalent(load_reference(path), _load_new(path))


@needs_ingest
def test_ingest_fixes_reference_bugs(corpus_paths):
    cp = _load_new(corpus_paths["cp1252_accents"])
    assert cp.error is None, f"cp1252 non caricato: {cp.error!r}"
    assert list(cp.df.columns) == ["misura", "pressione àèé"]
    assert len(cp.df) == 60
    assert str(cp.df["pressione àèé"].dtype) == "Float64"
    assert cp.df["pressione àèé"].notna().all()

    nh = _load_new(corpus_paths["no_header"])
    assert nh.error is None, f"no_header non caricato: {nh.error!r}"
    assert len(nh.df) == 100
    assert nh.df.iloc[0, 0] == pytest.approx(0.0)
    assert nh.df.iloc[1, 0] == pytest.approx(0.01)


@needs_ingest
def test_ingest_msvc_tokens_become_na(corpus_paths):
    outcome = _load_new(corpus_paths["utf16le_excel_like"])
    assert outcome.error is None
    offset = outcome.df["Offset [V]"]
    assert str(offset.dtype) == "Float64"
    assert offset.isna().all()
    assert list(outcome.df.columns[:3]) == ["SS Counter", "Undeclared []", "Undeclared [].1"]
    assert np.isfinite(outcome.df["ACC11 [m/s2]"].to_numpy(dtype="float64", na_value=np.nan)).all()


# ---------------------------------------------------------------------------
# File locali (tests_csv/ non è in git): solo sulla macchina di sviluppo
# ---------------------------------------------------------------------------
def _local_csvs() -> list[Path]:
    if not LOCAL_CSV_DIR.is_dir():
        return []
    return sorted(
        p for p in LOCAL_CSV_DIR.glob("*.csv") if p.stat().st_size <= LOCAL_MAX_MB * 1024 * 1024
    )


@needs_ingest
@pytest.mark.slow
@pytest.mark.parametrize("path", _local_csvs(), ids=lambda p: p.name)
def test_ingest_parity_local_files(path):
    assert_load_equivalent(load_reference(path), _load_new(path))


def test_corpus_is_deterministic(tmp_path):
    first = write_corpus(tmp_path / "a")
    second = write_corpus(tmp_path / "b")
    for name in CORPUS:
        assert first[name].read_bytes() == second[name].read_bytes(), name


def test_reference_bug_cases_still_documented(corpus_paths):
    """Se il riferimento venisse corretto, togliere il caso da KNOWN_REFERENCE_BUGS."""
    cp = load_reference(corpus_paths["cp1252_accents"])
    nh = load_reference(corpus_paths["no_header"])
    assert cp.error is not None
    assert isinstance(nh.df, pd.DataFrame) and len(nh.df) == 99
