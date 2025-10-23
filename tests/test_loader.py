from __future__ import annotations

from pathlib import Path

import pytest

from core.loader import load_csv


def test_load_csv_applies_cleaning(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "tempo;valore\n"
        "0;1.234,56\n"
        "1;7.890,12\n",
        encoding="utf-8",
    )

    df, report = load_csv(
        str(csv_path),
        delimiter=";",
        header=0,
        return_details=True,
    )

    assert "valore" in report.numeric_columns
    assert df["valore"].iloc[0] == pytest.approx(1234.56)
    assert df["valore"].iloc[1] == pytest.approx(7890.12)


def test_load_csv_without_cleaning(tmp_path: Path) -> None:
    csv_path = tmp_path / "raw.csv"
    csv_path.write_text(
        "value;other\n"
        "1,234.56;100\n",
        encoding="utf-8",
    )

    df = load_csv(str(csv_path), delimiter=";", header=0, apply_cleaning=False)

    # FASE 1: Con engine C, dtype=str produce dtype 'object' invece di 'string'
    assert df["value"].dtype == "object"
    assert df["value"].iloc[0] == "1,234.56"


def test_load_csv_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_csv(str(missing))


# FASE 1 OPTIMIZATION: Test for engine C with fallback
def test_load_csv_with_malformed_rows(tmp_path: Path) -> None:
    """Test engine C with fallback for malformed rows (missing columns)."""
    csv_path = tmp_path / "malformed.csv"
    csv_path.write_text(
        "a,b,c\n"
        "1,2,3\n"
        "4,5\n"        # Malformed row (missing column)
        "7,8,9,10\n"   # Malformed row (extra column)
        "11,12,13\n",
        encoding="utf-8",
    )

    # Should work thanks to on_bad_lines='skip'
    df = load_csv(str(csv_path), delimiter=",", header=0)

    # At least the valid rows should be loaded
    assert len(df) >= 2
    assert "a" in df.columns
    assert "b" in df.columns
    assert "c" in df.columns


def test_load_csv_tab_delimiter(tmp_path: Path) -> None:
    """Test tab delimiter with engine C."""
    csv_path = tmp_path / "tabs.tsv"
    csv_path.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n", encoding="utf-8")

    df = load_csv(str(csv_path), delimiter="\t", header=0)

    assert len(df.columns) == 3
    assert len(df) == 2
    # Cleaning is enabled by default, so numeric values are converted to float
    assert df["a"].iloc[0] == pytest.approx(1.0)


def test_load_csv_semicolon_delimiter(tmp_path: Path) -> None:
    """Test semicolon delimiter (common in European CSVs)."""
    csv_path = tmp_path / "semicolon.csv"
    csv_path.write_text("x;y;z\n10;20;30\n40;50;60\n", encoding="utf-8")

    df = load_csv(str(csv_path), delimiter=";", header=0)

    assert len(df.columns) == 3
    assert len(df) == 2


def test_load_csv_pipe_delimiter(tmp_path: Path) -> None:
    """Test pipe delimiter."""
    csv_path = tmp_path / "pipe.csv"
    csv_path.write_text("col1|col2\nval1|val2\nval3|val4\n", encoding="utf-8")

    df = load_csv(str(csv_path), delimiter="|", header=0)

    assert len(df.columns) == 2
    assert len(df) == 2
