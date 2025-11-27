"""
Test suite per loader_optimized.py

Testa:
- Auto-detection della strategia (small vs large file)
- Caricamento chunked
- Sampling stratificato
- Progress callback
- Compatibilità con loader legacy
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from core.loader_optimized import (
    load_csv,
    load_csv_chunked,
    load_csv_sampled,
    should_use_sampling,
    LoadProgress,
    _count_rows_fast,
    _generate_stratified_indices,
)


@pytest.fixture
def small_csv(tmp_path):
    """Crea un CSV piccolo (< 50 MB)"""
    csv_path = tmp_path / "small.csv"
    df = pd.DataFrame({
        "Col_A": range(100),
        "Col_B": np.random.uniform(0, 100, 100),
        "Col_C": ["text"] * 100
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def large_csv(tmp_path):
    """Crea un CSV grande (> 50 MB simulato con molte righe)"""
    csv_path = tmp_path / "large.csv"
    # 150k righe × 10 colonne = ~15 MB, ma supera ROWS_THRESHOLD (100k)
    rows = 150_000
    df = pd.DataFrame({
        f"Col_{i}": np.random.uniform(0, 1000, rows)
        for i in range(10)
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def european_format_csv(tmp_path):
    """CSV con numeri in formato europeo (virgola decimale, punto migliaia)"""
    csv_path = tmp_path / "european.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Value,Amount\n")
        f.write("1.234,56,123.456,78\n")
        f.write("9.876,54,987.654,32\n")
        f.write("123,45,12.345,67\n")
    return csv_path


class TestShouldUseSampling:
    """Test per la logica di decisione strategia"""

    def test_small_file_no_sampling(self, small_csv):
        """File piccolo non deve usare sampling"""
        use_sampling, reason = should_use_sampling(small_csv)
        assert not use_sampling
        assert "small enough" in reason.lower()

    def test_large_file_uses_sampling(self, large_csv):
        """File grande deve usare sampling"""
        use_sampling, reason = should_use_sampling(large_csv)
        assert use_sampling
        # Può essere per size o per rows estimate
        assert ("rows" in reason.lower() or "mb" in reason.lower())


class TestCountRowsFast:
    """Test per conteggio veloce righe"""

    def test_count_rows_simple(self, small_csv):
        """Conta righe su CSV piccolo"""
        count = _count_rows_fast(small_csv)
        # 100 righe dati + 1 header = 101
        assert count == 101

    def test_count_rows_empty(self, tmp_path):
        """File vuoto dovrebbe avere 0 righe"""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")
        count = _count_rows_fast(empty_csv)
        assert count == 0


class TestGenerateStratifiedIndices:
    """Test per generazione indici stratificati"""

    def test_generates_correct_count(self):
        """Genera esattamente sample_size indici"""
        indices = _generate_stratified_indices(
            total_rows=10000,
            sample_size=1000,
            random_seed=42
        )
        assert len(indices) == 1000

    def test_indices_within_bounds(self):
        """Tutti gli indici devono essere validi"""
        indices = _generate_stratified_indices(
            total_rows=10000,
            sample_size=1000,
            random_seed=42
        )
        assert all(1 <= idx <= 10000 for idx in indices)

    def test_stratification_distribution(self):
        """Verifica distribuzione stratificata (inizio, centro, fine)"""
        total_rows = 10000
        sample_size = 1000
        indices = _generate_stratified_indices(total_rows, sample_size, random_seed=42)

        # 40% dovrebbero essere all'inizio
        start_count = sum(1 for idx in indices if idx <= 400)
        # 40% dovrebbero essere alla fine
        end_count = sum(1 for idx in indices if idx >= 9600)

        # Verifica che ci siano campioni bilanciati (tolleranza ±20%)
        assert start_count > 300  # Almeno 300 all'inizio (75% di 400)
        assert end_count > 300    # Almeno 300 alla fine

    def test_reproducibility_with_seed(self):
        """Stesso seed produce stessi indici"""
        indices1 = _generate_stratified_indices(10000, 1000, random_seed=42)
        indices2 = _generate_stratified_indices(10000, 1000, random_seed=42)
        assert indices1 == indices2


class TestLoadCsvSampled:
    """Test per caricamento con sampling"""

    def test_load_sampled_small_file(self, small_csv):
        """File piccolo con sample_size < file size viene comunque campionato"""
        df, metadata = load_csv_sampled(
            str(small_csv),
            sample_size=50
        )

        # Il file ha 100 righe, ma sample_size=50 → campiona 50
        assert len(df) == 50
        assert metadata["is_sample"] is True
        assert metadata["sampling_ratio"] == 0.5

    def test_load_sampled_large_file(self, large_csv):
        """File grande viene campionato"""
        df, metadata = load_csv_sampled(
            str(large_csv),
            sample_size=10000
        )

        assert len(df) == 10000  # Carica solo sample_size righe
        assert metadata["is_sample"] is True
        assert metadata["sampling_ratio"] < 1.0
        assert metadata["total_rows"] == 150_000

    def test_sampled_data_quality(self, large_csv):
        """Dati campionati devono essere validi"""
        df, metadata = load_csv_sampled(
            str(large_csv),
            sample_size=5000
        )

        # Verifica che le colonne siano presenti
        assert len(df.columns) == 10
        # Verifica che non ci siano NaN (i dati erano numerici)
        assert not df.isnull().all().any()


class TestLoadCsvChunked:
    """Test per caricamento chunked"""

    def test_chunked_load_small_file(self, small_csv):
        """File piccolo caricato a chunks"""
        df, report = load_csv_chunked(
            str(small_csv),
            chunk_size=30,  # 30 righe per chunk
            apply_cleaning=True
        )

        assert len(df) == 100
        assert len(df.columns) == 3
        # Verifica che la pulizia sia stata applicata
        assert report.applied_cleaning is True

    def test_chunked_load_large_file(self, large_csv):
        """File grande caricato a chunks"""
        df, report = load_csv_chunked(
            str(large_csv),
            chunk_size=50_000,
            apply_cleaning=True
        )

        assert len(df) == 150_000
        assert len(df.columns) == 10

    def test_chunked_handles_european_format(self, european_format_csv):
        """Chunked loader gestisce formato europeo"""
        df, report = load_csv_chunked(
            str(european_format_csv),
            chunk_size=2,
            apply_cleaning=True
        )

        # Verifica che i numeri siano stati convertiti
        assert report.numeric_columns  # Almeno una colonna numerica
        assert df["Value"].dtype in ["float64", "Float64"]

    def test_progress_callback(self, small_csv):
        """Progress callback viene chiamato"""
        progress_updates = []

        def capture_progress(progress: LoadProgress):
            progress_updates.append({
                "phase": progress.phase,
                "message": progress.message
            })

        df, report = load_csv_chunked(
            str(small_csv),
            chunk_size=30,
            progress_callback=capture_progress
        )

        # Verifica che il callback sia stato chiamato
        assert len(progress_updates) > 0
        # Verifica che ci sia una fase di format detection
        phases = [u["phase"] for u in progress_updates]
        assert "format_detection" in phases or "loading" in phases


class TestLoadCsvOptimized:
    """Test per funzione load_csv con auto-detection"""

    def test_auto_uses_legacy_for_small(self, small_csv):
        """File piccolo usa loader legacy"""
        df = load_csv(
            str(small_csv),
            use_optimization=True
        )

        assert len(df) == 100
        assert len(df.columns) == 3

    def test_auto_uses_chunked_for_large(self, large_csv):
        """File grande usa chunked loader"""
        df, report = load_csv(
            str(large_csv),
            use_optimization=True,
            return_details=True
        )

        assert len(df) == 150_000
        assert len(df.columns) == 10

    def test_force_legacy_loader(self, large_csv):
        """Forza uso loader legacy anche per file grandi"""
        df = load_csv(
            str(large_csv),
            use_optimization=False  # Forza legacy
        )

        assert len(df) == 150_000

    def test_return_details(self, small_csv):
        """return_details restituisce report"""
        result = load_csv(
            str(small_csv),
            return_details=True
        )

        assert isinstance(result, tuple)
        df, report = result
        assert isinstance(df, pd.DataFrame)
        assert hasattr(report, "numeric_columns")

    def test_no_return_details(self, small_csv):
        """Senza return_details restituisce solo DataFrame"""
        result = load_csv(
            str(small_csv),
            return_details=False
        )

        assert isinstance(result, pd.DataFrame)


class TestEdgeCases:
    """Test per edge cases e situazioni limite"""

    def test_file_not_found(self):
        """File non esistente solleva FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/file.csv")

    def test_empty_csv(self, tmp_path):
        """CSV vuoto viene gestito (solo header, no dati)"""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("Col_A,Col_B\n")  # Solo header

        # Il loader ora gestisce CSV vuoti senza errori (restituisce DataFrame 0 righe)
        df = load_csv(str(empty_csv))
        assert len(df) == 0
        assert len(df.columns) == 2

    def test_single_row_csv(self, tmp_path):
        """CSV con una sola riga dati"""
        single_row_csv = tmp_path / "single.csv"
        with open(single_row_csv, 'w') as f:
            f.write("A,B,C\n")
            f.write("1,2,3\n")

        df = load_csv(str(single_row_csv))
        assert len(df) == 1
        assert len(df.columns) == 3

    def test_very_wide_csv(self, tmp_path):
        """CSV con molte colonne (stress test)"""
        wide_csv = tmp_path / "wide.csv"
        num_cols = 100
        df = pd.DataFrame({
            f"Col_{i}": np.random.uniform(0, 100, 50)
            for i in range(num_cols)
        })
        df.to_csv(wide_csv, index=False)

        df_loaded = load_csv(str(wide_csv))
        assert len(df_loaded.columns) == num_cols

    def test_mixed_types_columns(self, tmp_path):
        """CSV con tipi misti per colonna"""
        mixed_csv = tmp_path / "mixed.csv"
        with open(mixed_csv, 'w') as f:
            f.write("Col_A,Col_B\n")
            f.write("123,abc\n")
            f.write("456,def\n")
            f.write("invalid,ghi\n")

        df = load_csv(str(mixed_csv), apply_cleaning=True)

        # Col_A dovrebbe rimanere string perché ha "invalid"
        # Col_B è completamente testuale
        assert len(df) == 3


class TestCompatibility:
    """Test di compatibilità con loader legacy"""

    def test_same_api_signature(self, small_csv):
        """API identica tra loader legacy e ottimizzato"""
        from core.loader import load_csv as load_csv_legacy

        df_legacy = load_csv_legacy(str(small_csv))
        df_optimized = load_csv(str(small_csv), use_optimization=False)

        # Stessi risultati
        assert len(df_legacy) == len(df_optimized)
        assert list(df_legacy.columns) == list(df_optimized.columns)

    def test_cleaning_consistency(self, european_format_csv):
        """Pulizia numerica consistente tra i due loader"""
        from core.loader import load_csv as load_csv_legacy

        df_legacy, report_legacy = load_csv_legacy(
            str(european_format_csv),
            apply_cleaning=True,
            return_details=True
        )

        df_optimized, report_optimized = load_csv(
            str(european_format_csv),
            apply_cleaning=True,
            return_details=True,
            use_optimization=False  # Usa legacy per confronto
        )

        # Stesse colonne numeriche convertite
        assert set(report_legacy.numeric_columns) == set(report_optimized.numeric_columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
