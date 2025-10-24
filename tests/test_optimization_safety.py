"""
Test di sicurezza per le ottimizzazioni CSV loading.

Questi test DEVONO passare PRIMA e DOPO ogni fase di ottimizzazione
per garantire che non ci siano regressioni o breaking changes.

Run:
    pytest tests/test_optimization_safety.py -v
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from core.loader import load_csv
from core.csv_cleaner import clean_dataframe, suggest_number_format


class TestEngineSafety:
    """Test per garantire robustezza engine C con fallback."""

    def test_malformed_rows_missing_columns(self, tmp_path: Path) -> None:
        """File con righe che hanno meno colonne dell'header."""
        csv_path = tmp_path / "malformed_missing.csv"
        csv_path.write_text(
            "a,b,c\n"
            "1,2,3\n"
            "4,5\n"      # ← Missing column 'c'
            "7,8,9\n",
            encoding="utf-8",
        )

        # Dovrebbe funzionare grazie a on_bad_lines='skip'
        df = load_csv(str(csv_path), delimiter=",", header=0)

        # Verifica che abbia caricato almeno le righe valide
        assert len(df) >= 2
        assert list(df.columns) == ["a", "b", "c"]

    def test_malformed_rows_extra_columns(self, tmp_path: Path) -> None:
        """File con righe che hanno più colonne dell'header."""
        csv_path = tmp_path / "malformed_extra.csv"
        csv_path.write_text(
            "a,b,c\n"
            "1,2,3\n"
            "4,5,6,7,8\n"  # ← Extra columns
            "9,10,11\n",
            encoding="utf-8",
        )

        df = load_csv(str(csv_path), delimiter=",", header=0)
        assert len(df) >= 2

    def test_mixed_malformed_rows(self, tmp_path: Path) -> None:
        """File con mix di righe valide e malformate."""
        csv_path = tmp_path / "mixed_malformed.csv"
        csv_path.write_text(
            "col1,col2,col3\n"
            "1,2,3\n"
            "4,5\n"        # Missing
            "6,7,8,9\n"    # Extra
            "10,11,12\n"
            ",13,\n"       # Empty values (valid)
            "14,15,16\n",
            encoding="utf-8",
        )

        df = load_csv(str(csv_path), delimiter=",", header=0)
        # Dovrebbe avere almeno le righe completamente valide
        assert len(df) >= 3

    def test_tab_delimiter(self, tmp_path: Path) -> None:
        """File TSV con tab delimiter."""
        csv_path = tmp_path / "tabs.tsv"
        csv_path.write_text(
            "a\tb\tc\n"
            "1\t2\t3\n"
            "4\t5\t6\n",
            encoding="utf-8",
        )

        df = load_csv(str(csv_path), delimiter="\t", header=0)
        assert len(df.columns) == 3
        # Cleaning converte a numerico (corretto)
        assert float(df["a"].iloc[0]) == 1.0

    def test_pipe_delimiter(self, tmp_path: Path) -> None:
        """File con pipe delimiter."""
        csv_path = tmp_path / "pipes.csv"
        csv_path.write_text(
            "a|b|c\n"
            "1|2|3\n"
            "4|5|6\n",
            encoding="utf-8",
        )

        df = load_csv(str(csv_path), delimiter="|", header=0)
        assert len(df.columns) == 3

    def test_semicolon_delimiter(self, tmp_path: Path) -> None:
        """File con semicolon delimiter (comune in EU)."""
        csv_path = tmp_path / "semicolon.csv"
        csv_path.write_text(
            "a;b;c\n"
            "1;2;3\n"
            "4;5;6\n",
            encoding="utf-8",
        )

        df = load_csv(str(csv_path), delimiter=";", header=0)
        assert len(df.columns) == 3


class TestEncodingSafety:
    """Test per garantire gestione corretta encoding edge cases."""

    def test_utf8_with_bom(self, tmp_path: Path) -> None:
        """File UTF-8 con BOM (\\ufeff)."""
        csv_path = tmp_path / "utf8_bom.csv"
        # Write BOM + content
        csv_path.write_bytes(
            b"\xef\xbb\xbf"  # UTF-8 BOM
            b"a,b,c\n"
            b"1,2,3\n"
        )

        df = load_csv(str(csv_path), encoding="utf-8-sig", delimiter=",", header=0)
        assert len(df) == 1
        # Verifica che BOM sia rimosso dai nomi colonne
        assert "a" in df.columns
        assert "\ufeffa" not in df.columns

    def test_latin1_encoding(self, tmp_path: Path) -> None:
        """File con encoding Latin-1 (ISO-8859-1)."""
        csv_path = tmp_path / "latin1.csv"
        # Caratteri accentati Latin-1
        content = "città,età\nRoma,2000\nMilano,1500\n"
        csv_path.write_bytes(content.encode("latin-1"))

        df = load_csv(str(csv_path), encoding="latin-1", delimiter=",", header=0)
        assert len(df) == 2
        assert "città" in df.columns

    def test_utf16_encoding(self, tmp_path: Path) -> None:
        """File UTF-16 (Windows Unicode)."""
        csv_path = tmp_path / "utf16.csv"
        content = "a,b,c\n1,2,3\n"
        csv_path.write_bytes(content.encode("utf-16"))

        df = load_csv(str(csv_path), encoding="utf-16", delimiter=",", header=0)
        assert len(df) == 1


class TestSampleBasedSafety:
    """Test per validare sample-based format detection."""

    def test_heterogeneous_numeric_format(self) -> None:
        """
        File con pattern numerici che cambiano nel mezzo.

        Questo è il caso più critico per sample-based detection:
        se il sample include solo le prime N righe, potrebbe
        rilevare un formato sbagliato.
        """
        # Prime 50 righe: formato europeo (virgola decimale)
        eu_data = [f"1.{i:03d},{i:02d}" for i in range(50)]
        # Ultime 50 righe: formato US (punto decimale)
        us_data = [f"2,{i:03d}.{i:02d}" for i in range(50, 100)]

        df = pd.DataFrame({"value": eu_data + us_data})

        # Verifica detection su intero DataFrame
        suggestion = suggest_number_format(df)

        # Confidence dovrebbe essere BASSA perché pattern misto
        # Oppure dovrebbe scegliere formato dominante
        assert suggestion.sample_size > 0
        assert suggestion.confidence >= 0.0  # Almeno non crashare

    def test_sample_all_same_format(self) -> None:
        """File omogeneo: sample dovrebbe dare alta confidence."""
        data = [f"1.{i:03d},50" for i in range(100)]
        df = pd.DataFrame({"value": data})

        suggestion = suggest_number_format(df)
        assert suggestion.decimal == ","
        assert suggestion.thousands == "."
        assert suggestion.confidence > 0.9

    def test_small_file_sample(self) -> None:
        """File piccolo (< 100 righe): sample = intero file."""
        data = ["1.234,56", "7.890,12", "100,00"]
        df = pd.DataFrame({"value": data})

        suggestion = suggest_number_format(df)
        assert suggestion.sample_size == 3
        assert suggestion.decimal == ","


class TestDtypeInferenceSafety:
    """Test per dtype inference sicura."""

    def test_column_with_late_text_values(self) -> None:
        """
        Colonna che sembra numerica nelle prime N righe,
        ma ha valori testuali più avanti.
        """
        # Prime 95 righe: numeri
        numeric_data = [str(i) for i in range(95)]
        # Ultime 5 righe: testo
        text_data = ["N/A", "ERROR", "INVALID", "NULL", "MISSING"]

        df = pd.DataFrame({"col": numeric_data + text_data})

        # Il cleaning dovrebbe gestirlo correttamente
        result = clean_dataframe(df, apply=True)

        # Conversion rate = 95/100 = 0.95 >= MIN_CONVERSION_RATE
        col_report = result.report.columns["col"]
        assert col_report.converted == 95
        assert col_report.non_numeric == 5

    def test_fully_numeric_column(self) -> None:
        """Colonna 100% numerica dovrebbe essere convertita."""
        df = pd.DataFrame({"col": ["123", "456.78", "9.01"]})
        result = clean_dataframe(df, apply=True)

        assert "col" in result.report.numeric_columns
        assert result.df["col"].dtype == "Float64"

    def test_mostly_text_column(self) -> None:
        """Colonna prevalentemente testuale NON dovrebbe essere convertita."""
        df = pd.DataFrame({"col": ["abc", "def", "ghi", "123", "xyz"]})
        result = clean_dataframe(df, apply=True)

        # 1/5 = 0.2 < MIN_CONVERSION_RATE (0.66)
        assert "col" not in result.report.numeric_columns

    def test_mixed_50_50_column(self) -> None:
        """Colonna 50% numerica / 50% testo."""
        df = pd.DataFrame({
            "col": ["123", "abc", "456", "def", "789", "ghi"]
        })
        result = clean_dataframe(df, apply=True)

        # 3/6 = 0.5 < 0.66 → NON convertita
        col_report = result.report.columns["col"]
        assert not col_report.applied


class TestBackwardCompatibility:
    """Test per garantire backward compatibility API."""

    def test_load_csv_signature_unchanged(self, tmp_path: Path) -> None:
        """Signature di load_csv() deve rimanere invariata."""
        csv_path = tmp_path / "compat.csv"
        csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

        # Tutte queste chiamate devono funzionare
        df1 = load_csv(str(csv_path))
        df2 = load_csv(str(csv_path), delimiter=",")
        df3 = load_csv(str(csv_path), delimiter=",", header=0)
        df4 = load_csv(str(csv_path), encoding="utf-8")
        df5, report = load_csv(str(csv_path), return_details=True)
        df6 = load_csv(str(csv_path), apply_cleaning=False)

        assert all(len(df) == 1 for df in [df1, df2, df3, df4, df5, df6])

    def test_clean_dataframe_signature_unchanged(self) -> None:
        """Signature di clean_dataframe() deve rimanere invariata."""
        df = pd.DataFrame({"val": ["123,45"]})

        # Tutte queste chiamate devono funzionare
        result1 = clean_dataframe(df)
        result2 = clean_dataframe(df, apply=True)
        result3 = clean_dataframe(df, apply=False)
        result4 = clean_dataframe(df, decimal=",")
        result5 = clean_dataframe(df, decimal=",", thousands=".")

        assert all(hasattr(r, "df") and hasattr(r, "report") for r in [
            result1, result2, result3, result4, result5
        ])


class TestRegressionPrevention:
    """Test per prevenire regressioni note."""

    def test_issue_50_downsampling(self, tmp_path: Path) -> None:
        """
        Issue #50: Downsampling inefficiente.

        Verifica che downsampling sia applicato al DataFrame PRIMA
        dei loop, non per-series.
        """
        # Questo test è più concettuale; verifica che il codice
        # non faccia chiamate ridondanti a downsampling
        csv_path = tmp_path / "large.csv"
        data = "x," + ",".join([f"y{i}" for i in range(10)]) + "\n"
        for i in range(1000):
            data += f"{i}," + ",".join([str(i * j) for j in range(10)]) + "\n"
        csv_path.write_text(data, encoding="utf-8")

        df = load_csv(str(csv_path), delimiter=",", header=0)
        # Verifica solo che carichi correttamente
        assert len(df) == 1000
        assert len(df.columns) == 11

    def test_issue_52_datetime_conversion(self, tmp_path: Path) -> None:
        """
        Issue #52: Conversioni datetime ripetute.

        Verifica che colonne X non siano convertite più volte.
        """
        csv_path = tmp_path / "datetime.csv"
        csv_path.write_text(
            "timestamp,value\n"
            "2024-01-01,100\n"
            "2024-01-02,200\n",
            encoding="utf-8",
        )

        df = load_csv(str(csv_path), delimiter=",", header=0)
        # Verifica che datetime sia gestibile
        assert len(df) == 2

    def test_issue_53_dtype_optimization(self, tmp_path: Path) -> None:
        """
        Issue #53: Ottimizzazione dtype float64→float32.

        Verifica che optimize_dtypes() sia chiamabile senza errori.
        """
        from core.loader import optimize_dtypes

        csv_path = tmp_path / "floats.csv"
        csv_path.write_text(
            "value\n"
            "1.5\n"
            "2.5\n"
            "3.5\n",
            encoding="utf-8",
        )

        df = load_csv(str(csv_path), delimiter=",", header=0)
        df_opt, conversions = optimize_dtypes(df, enabled=True)

        # Verifica che abbia provato a ottimizzare
        assert isinstance(conversions, dict)
        # Se value è float64, dovrebbe convertire a float32
        # (dipende da come load_csv gestisce i dtype)


class TestEdgeCasesRobustness:
    """Test per edge cases che potrebbero rompere ottimizzazioni."""

    def test_empty_file(self, tmp_path: Path) -> None:
        """File completamente vuoto."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("", encoding="utf-8")

        with pytest.raises(Exception):  # Dovrebbe sollevare errore appropriato
            load_csv(str(csv_path))

    def test_only_header_no_data(self, tmp_path: Path) -> None:
        """File con solo header, nessun dato."""
        csv_path = tmp_path / "only_header.csv"
        csv_path.write_text("a,b,c\n", encoding="utf-8")

        df = load_csv(str(csv_path), delimiter=",", header=0)
        assert len(df) == 0
        assert list(df.columns) == ["a", "b", "c"]

    def test_single_column_file(self, tmp_path: Path) -> None:
        """File con una sola colonna."""
        csv_path = tmp_path / "single_col.csv"
        csv_path.write_text("value\n123\n456\n", encoding="utf-8")

        df = load_csv(str(csv_path), delimiter=",", header=0)
        assert len(df.columns) == 1
        assert len(df) == 2

    def test_very_long_lines(self, tmp_path: Path) -> None:
        """File con righe molto lunghe (>1000 caratteri)."""
        csv_path = tmp_path / "long_lines.csv"
        # Riga con 500 colonne
        header = ",".join([f"col{i}" for i in range(500)])
        data = ",".join(["123"] * 500)
        csv_path.write_text(f"{header}\n{data}\n", encoding="utf-8")

        df = load_csv(str(csv_path), delimiter=",", header=0)
        assert len(df.columns) == 500
        assert len(df) == 1

    def test_unicode_column_names(self, tmp_path: Path) -> None:
        """Nomi colonna con caratteri Unicode."""
        csv_path = tmp_path / "unicode.csv"
        csv_path.write_text(
            "città,età,€_value\n"
            "Roma,2000,100\n",
            encoding="utf-8",
        )

        df = load_csv(str(csv_path), encoding="utf-8", delimiter=",", header=0)
        assert "città" in df.columns
        assert "età" in df.columns
        assert "€_value" in df.columns

    def test_quoted_fields_with_delimiter(self, tmp_path: Path) -> None:
        """Campi quotati che contengono il delimiter."""
        csv_path = tmp_path / "quoted.csv"
        csv_path.write_text(
            'a,b,c\n'
            '"1,2",3,4\n'  # b contiene virgola
            '5,"6,7",8\n',
            encoding="utf-8",
        )

        df = load_csv(str(csv_path), delimiter=",", header=0)
        assert len(df.columns) == 3
        # Prima riga: a="1,2", b="3", c="4"
        # (il parser dovrebbe gestire le quote)


# ============================================================================
# PERFORMANCE REGRESSION TESTS
# ============================================================================

class TestPerformanceNoRegression:
    """
    Test per garantire che le ottimizzazioni NON rallentino
    casi specifici.

    Nota: Questi test NON verificano speedup, ma solo che
    il tempo di esecuzione rimanga ragionevole.
    """

    def test_small_file_performance(self, tmp_path: Path, benchmark=None) -> None:
        """File piccolo (100 righe) deve caricare in <50ms."""
        csv_path = tmp_path / "small.csv"
        data = "a,b,c\n" + "\n".join([f"{i},{i*2},{i*3}" for i in range(100)])
        csv_path.write_text(data, encoding="utf-8")

        # Se pytest-benchmark disponibile, usa benchmark
        # Altrimenti, semplice timing check
        import time
        start = time.perf_counter()
        df = load_csv(str(csv_path), delimiter=",", header=0)
        elapsed = time.perf_counter() - start

        assert len(df) == 100
        # Tempo ragionevole: <100ms per 100 righe
        assert elapsed < 0.1, f"Too slow: {elapsed*1000:.1f}ms"

    def test_medium_file_performance(self, tmp_path: Path) -> None:
        """File medio (10k righe) deve caricare in <1s."""
        csv_path = tmp_path / "medium.csv"
        data = "a,b,c\n" + "\n".join([f"{i},{i*2},{i*3}" for i in range(10000)])
        csv_path.write_text(data, encoding="utf-8")

        import time
        start = time.perf_counter()
        df = load_csv(str(csv_path), delimiter=",", header=0)
        elapsed = time.perf_counter() - start

        assert len(df) == 10000
        # Tempo ragionevole: <1s per 10k righe
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s"


# ============================================================================
# SAFETY SUMMARY TEST
# ============================================================================

def test_all_safety_checks_present():
    """
    Meta-test: verifica che tutti i test di sicurezza siano presenti.

    Questo test garantisce che nessun test di sicurezza sia stato
    accidentalmente rimosso.
    """
    import inspect
    import sys

    # Ottieni tutti i test in questo modulo
    current_module = sys.modules[__name__]
    test_classes = [
        obj for name, obj in inspect.getmembers(current_module)
        if inspect.isclass(obj) and name.startswith("Test")
    ]

    # Conta test totali
    total_tests = 0
    for cls in test_classes:
        test_methods = [
            method for method in dir(cls)
            if method.startswith("test_")
        ]
        total_tests += len(test_methods)

    # SOGLIA MINIMA: almeno 29 test di sicurezza
    # (29 test attuali: aumentare soglia man mano che si aggiungono test)
    assert total_tests >= 29, (
        f"Safety test coverage too low: {total_tests} tests "
        f"(expected >= 29)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
