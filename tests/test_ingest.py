"""Test unitari di core.ingest: percorsi di ripiego, report e regressioni."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from core import ingest
from core.analyzer import analyze_csv
from core.loader import load_csv as load_csv_standard

pytestmark = pytest.mark.unit


def _write(tmp_path: Path, text: str, name: str = "data.csv", encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest.load(tmp_path / "manca.csv")


def test_arrow_engine_and_timings(tmp_path):
    path = _write(tmp_path, "t,v\n0,1.5\n1,2.5\n2,3.5\n")
    result = ingest.load(path)
    assert result.engine == "arrow"
    assert {"analyze_s", "read_s", "convert_s", "frame_s", "total_s"} <= set(result.timings)
    assert result.df["v"].tolist() == [1.5, 2.5, 3.5]
    assert str(result.df["v"].dtype) == "Float64"


def test_reuses_given_meta(tmp_path, monkeypatch):
    path = _write(tmp_path, "t;v\n0;1,5\n1;2,5\n2;3,5\n")
    meta = analyze_csv(str(path))

    def fail(*_args, **_kwargs):
        raise AssertionError("analyze_csv non deve essere richiamato")

    monkeypatch.setattr(ingest, "analyze_csv", fail)
    result = ingest.load(path, meta=meta)
    assert result.df["v"].tolist() == [1.5, 2.5, 3.5]


def test_nulls_do_not_corrupt_values(tmp_path):
    """Regressione: con pyarrow 21 fill_null().to_numpy() azzerava i valori."""
    values = [f"{i * 0.2040816:.7f}" for i in range(10)] + ["", "", "inf", "-inf"]
    path = _write(tmp_path, "n,s\n" + "\n".join(f"{i},{v}" for i, v in enumerate(values)) + "\n")
    df = ingest.load(path).df
    assert df["s"].iloc[5] == pytest.approx(5 * 0.2040816)
    assert df["s"].iloc[9] == pytest.approx(9 * 0.2040816)
    assert df["s"].iloc[10:12].isna().all()
    assert df["s"].iloc[12] == np.inf and df["s"].iloc[13] == -np.inf


def test_to_float_array_keeps_mask():
    arr = ingest._to_float_array(pa.chunked_array([pa.array([1.0, None, 3.0]), pa.array([None, 5.0])]))
    assert arr.tolist() == [1.0, pd.NA, 3.0, pd.NA, 5.0]


def test_apply_cleaning_false_uses_standard_loader(tmp_path):
    path = _write(tmp_path, "t,v\n0,1.5\n1,2.5\n")
    result = ingest.load(path, apply_cleaning=False)
    assert result.engine == "legacy"
    assert result.report.numeric_columns == []


def test_rows_with_fewer_fields_fall_back(tmp_path):
    path = _write(tmp_path, "a,b,c\n1,2,3\n4,5\n6,7,8\n")
    result = ingest.load(path)
    expected, _ = load_csv_standard(str(path), "utf-8", ",", 0, return_details=True)
    assert result.engine == "legacy"
    assert result.df.shape == expected.shape


def test_mostly_bad_rows_fall_back(tmp_path):
    path = _write(tmp_path, "a,b\n" + "\n".join(f"{i},{i},{i}" for i in range(10)) + "\n")
    assert ingest.load(path).engine == "legacy"


def test_unreadable_header_falls_back(tmp_path, monkeypatch):
    path = _write(tmp_path, "a,b\n1,2\n3,4\n")

    def broken(*_args, **_kwargs):
        raise ValueError("intestazione rotta")

    monkeypatch.setattr(ingest.pd, "read_csv", broken)
    with pytest.raises(ValueError):
        # anche il loader standard usa pd.read_csv: l'errore deve emergere, non essere nascosto
        ingest.load(path)


def test_cp1252_fallback_updates_meta(tmp_path):
    path = _write(tmp_path, "nome;città\n1;2,5\n2;3,5\n3;4,5\n", encoding="cp1252")
    result = ingest.load(path)
    assert result.meta["encoding"] == "cp1252"
    assert list(result.df.columns) == ["nome", "città"]


def test_thousands_column_uses_legacy_cleaning(tmp_path):
    path = _write(tmp_path, "n;v\n" + "\n".join(f"{i};1.234,{i:02d}" for i in range(20)) + "\n")
    result = ingest.load(path)
    assert result.report.suggestion.thousands == "."
    assert result.df["v"].iloc[3] == pytest.approx(1234.03)


def test_rows_all_nan_matches_standard_loader(tmp_path):
    rows = [f"{i}.5,{i}.25" for i in range(9)]
    rows.insert(4, "1e5e5,2e3e3")
    path = _write(tmp_path, "a,b\n" + "\n".join(rows) + "\n")
    result = ingest.load(path)
    _, expected = load_csv_standard(str(path), "utf-8", ",", 0, return_details=True)
    assert expected.rows_all_nan_after_clean == [5]
    assert result.report.rows_all_nan_after_clean == expected.rows_all_nan_after_clean


def test_msvc_tokens_in_mixed_column_are_na(tmp_path):
    values = ["1.5", "1.#IO", "2.5", "-1.#R", "3.5", "1.#QO"]
    path = _write(tmp_path, "v\n" + "\n".join(values) + "\n")
    df = ingest.load(path).df
    assert df["v"].tolist()[0::2] == [1.5, 2.5, 3.5]
    assert df["v"].iloc[1::2].isna().all()


def test_msvc_tokens_in_legacy_column_are_na(tmp_path):
    values = ["€ 1.234,50", "1.#IO", "€ 2.000,00", "€ 3.100,25", "-1.#R"]
    path = _write(tmp_path, "v\n" + "\n".join(f'"{v}"' for v in values) + "\n")
    df = ingest.load(path).df
    assert df["v"].iloc[1] is pd.NA and df["v"].iloc[4] is pd.NA
    assert df["v"].iloc[0] == pytest.approx(1234.5)


def test_progress_callback_receives_phases(tmp_path):
    path = _write(tmp_path, "t,v\n0,1\n1,2\n")
    phases = []
    ingest.load(path, progress_callback=lambda p: phases.append(p.phase))
    assert phases[0] == "analyze"
    assert {"loading", "cleaning", "finalizing"} <= set(phases)


def test_loader_optimized_delegates_to_ingest(tmp_path, monkeypatch):
    from core import loader_optimized

    path = _write(tmp_path, "t,v\n0,1\n1,2\n")
    calls = []
    original = ingest.load
    monkeypatch.setattr(ingest, "load", lambda *a, **k: calls.append(k) or original(*a, **k))
    df, report = loader_optimized.load_csv(str(path), "utf-8", ",", 0, return_details=True)
    assert calls and calls[0]["meta"] == {"encoding": "utf-8", "delimiter": ",", "header": 0}
    assert report.numeric_columns == ["t", "v"]

    calls.clear()
    loader_optimized.load_csv(str(path), "utf-8", ",", 0, use_fast_ingest=False)
    assert not calls
