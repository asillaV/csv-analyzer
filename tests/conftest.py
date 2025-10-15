"""Configurazione pytest e fixtures condivise."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_csv_path() -> Iterator[Path]:
    """Crea un file CSV temporaneo per i test."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def sample_csv_basic(temp_csv_path: Path) -> Path:
    """CSV base con dati numerici semplici."""
    df = pd.DataFrame({
        'time': range(10),
        'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    })
    df.to_csv(temp_csv_path, index=False)
    return temp_csv_path


@pytest.fixture
def sample_csv_european_format(temp_csv_path: Path) -> Path:
    """CSV con formato numerico europeo (virgola decimale, punto migliaia)."""
    content = """valore,descrizione
1.234,56,test1
7.890,12,test2
12.345,67,test3
"""
    temp_csv_path.write_text(content, encoding='utf-8')
    return temp_csv_path


@pytest.fixture
def sample_csv_us_format(temp_csv_path: Path) -> Path:
    """CSV con formato numerico US (punto decimale, virgola migliaia)."""
    content = """value,description
1,234.56,test1
7,890.12,test2
12,345.67,test3
"""
    temp_csv_path.write_text(content, encoding='utf-8')
    return temp_csv_path
