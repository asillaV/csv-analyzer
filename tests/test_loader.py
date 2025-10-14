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

    assert df["value"].dtype == "string"
    assert df["value"].iloc[0] == "1,234.56"


def test_load_csv_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_csv(str(missing))
