"""Test per core/analyzer.py - rilevamento metadati CSV."""
from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from core.analyzer import CsvAnalyzer, analyze_csv


class TestBOMDetection:
    """Test per detect_bom()."""

    def test_detect_utf8(self, tmp_path: Path):
        """UTF-8 senza BOM."""
        file = tmp_path / "test.csv"
        file.write_bytes(b"col1,col2\n1,2\n")

        analyzer = CsvAnalyzer(str(file))
        encoding = analyzer.detect_bom()
        assert encoding == "utf-8"

    def test_detect_utf8_sig(self, tmp_path: Path):
        """UTF-8 con BOM."""
        file = tmp_path / "test.csv"
        file.write_bytes(b"\xef\xbb\xbfcol1,col2\n1,2\n")

        analyzer = CsvAnalyzer(str(file))
        encoding = analyzer.detect_bom()
        assert encoding == "utf-8-sig"

    def test_detect_utf16_le(self, tmp_path: Path):
        """UTF-16 Little Endian."""
        file = tmp_path / "test.csv"
        file.write_bytes(b"\xff\xfec\x00o\x00l\x00")  # BOM UTF-16 LE

        analyzer = CsvAnalyzer(str(file))
        encoding = analyzer.detect_bom()
        assert encoding == "utf-16"

    def test_detect_utf16_be(self, tmp_path: Path):
        """UTF-16 Big Endian."""
        file = tmp_path / "test.csv"
        file.write_bytes(b"\xfe\xff\x00c\x00o\x00l")  # BOM UTF-16 BE

        analyzer = CsvAnalyzer(str(file))
        encoding = analyzer.detect_bom()
        assert encoding == "utf-16-be"

    def test_file_not_found(self):
        """File inesistente."""
        analyzer = CsvAnalyzer("nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            analyzer.detect_bom()


class TestDelimiterDetection:
    """Test per detect_delimiter_and_header()."""

    def test_comma_delimiter(self, tmp_path: Path):
        """CSV con virgola come delimitatore."""
        file = tmp_path / "test.csv"
        file.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n", encoding="utf-8")

        analyzer = CsvAnalyzer(str(file))
        info = analyzer.detect_delimiter_and_header("utf-8")

        assert info["delimiter"] == ","
        assert info["header"] == 0  # Prima riga è header

    def test_semicolon_delimiter(self, tmp_path: Path):
        """CSV con punto e virgola."""
        file = tmp_path / "test.csv"
        file.write_text("name;age;city\nAlice;30;NYC\n", encoding="utf-8")

        analyzer = CsvAnalyzer(str(file))
        info = analyzer.detect_delimiter_and_header("utf-8")

        assert info["delimiter"] == ";"

    def test_tab_delimiter(self, tmp_path: Path):
        """CSV con tabulazione."""
        file = tmp_path / "test.csv"
        file.write_text("name\tage\tcity\nAlice\t30\tNYC\n", encoding="utf-8")

        analyzer = CsvAnalyzer(str(file))
        info = analyzer.detect_delimiter_and_header("utf-8")

        assert info["delimiter"] == "\t"

    def test_pipe_delimiter(self, tmp_path: Path):
        """CSV con pipe."""
        file = tmp_path / "test.csv"
        file.write_text("name|age|city\nAlice|30|NYC\n", encoding="utf-8")

        analyzer = CsvAnalyzer(str(file))
        info = analyzer.detect_delimiter_and_header("utf-8")

        assert info["delimiter"] == "|"

    def test_header_detection_numeric_data(self, tmp_path: Path):
        """Header con dati numerici sotto."""
        file = tmp_path / "test.csv"
        file.write_text("temperature,humidity\n25.5,60\n26.0,55\n", encoding="utf-8")

        analyzer = CsvAnalyzer(str(file))
        info = analyzer.detect_delimiter_and_header("utf-8")

        # Prima riga ha testo, seconda numeri → header = 0
        assert info["header"] == 0

    def test_no_header_all_numeric(self, tmp_path: Path):
        """Nessun header, solo dati numerici."""
        file = tmp_path / "test.csv"
        file.write_text("1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")

        analyzer = CsvAnalyzer(str(file))
        info = analyzer.detect_delimiter_and_header("utf-8")

        # Tutte le righe numeriche → nessun header
        assert info["header"] is None

    def test_header_on_second_row(self, tmp_path: Path):
        """Header alla seconda riga (euristica)."""
        file = tmp_path / "test.csv"
        # Prima riga: numeri, seconda: testo, terza: numeri
        file.write_text("1,2,3\nname,age,city\n10,20,30\n", encoding="utf-8")

        analyzer = CsvAnalyzer(str(file))
        info = analyzer.detect_delimiter_and_header("utf-8")

        # Seconda riga ha più testo della prima → header = 1
        assert info["header"] == 1


class TestColumnExtraction:
    """Test per extract_columns()."""

    def test_extract_columns_with_header(self, tmp_path: Path):
        """Estrazione colonne con header."""
        file = tmp_path / "test.csv"
        file.write_text("name,age,city\nAlice,30,NYC\n", encoding="utf-8")

        analyzer = CsvAnalyzer(str(file))
        columns = analyzer.extract_columns("utf-8", ",", header=0)

        assert columns == ["name", "age", "city"]

    def test_extract_columns_without_header(self, tmp_path: Path):
        """Generazione colonne senza header."""
        file = tmp_path / "test.csv"
        file.write_text("1,2,3\n4,5,6\n", encoding="utf-8")

        analyzer = CsvAnalyzer(str(file))
        columns = analyzer.extract_columns("utf-8", ",", header=None)

        assert columns == ["Column_0", "Column_1", "Column_2"]

    def test_extract_columns_clean_bom(self, tmp_path: Path):
        """Pulizia BOM nei nomi colonne."""
        file = tmp_path / "test.csv"
        file.write_bytes(b"\xef\xbb\xbfname,age\n1,2\n")

        analyzer = CsvAnalyzer(str(file))
        columns = analyzer.extract_columns("utf-8-sig", ",", header=0)

        # BOM dovrebbe essere rimosso
        assert columns[0] == "name"
        assert "\ufeff" not in columns[0]

    def test_extract_columns_with_spaces(self, tmp_path: Path):
        """Colonne con spazi iniziali/finali."""
        file = tmp_path / "test.csv"
        file.write_text("  name  , age ,city\n1,2,3\n", encoding="utf-8")

        analyzer = CsvAnalyzer(str(file))
        columns = analyzer.extract_columns("utf-8", ",", header=0)

        # Spazi dovrebbero essere rimossi
        assert columns == ["name", "age", "city"]


class TestAnalyzeIntegration:
    """Test per analyze() e analyze_csv()."""

    def test_analyze_complete_workflow(self, tmp_path: Path):
        """Workflow completo di analisi."""
        file = tmp_path / "test.csv"
        file.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n", encoding="utf-8")

        analyzer = CsvAnalyzer(str(file))
        result = analyzer.analyze()

        assert result["encoding"] == "utf-8"
        assert result["delimiter"] == ","
        assert result["header"] == 0
        assert result["columns"] == ["name", "age", "city"]

    def test_analyze_csv_function(self, tmp_path: Path):
        """Funzione analyze_csv() helper."""
        file = tmp_path / "test.csv"
        file.write_text("col1;col2\n123;456\n", encoding="utf-8")

        result = analyze_csv(str(file))

        assert "encoding" in result
        assert "delimiter" in result
        assert "header" in result
        assert "columns" in result
        assert result["delimiter"] == ";"

    def test_analyze_empty_file(self, tmp_path: Path):
        """File vuoto."""
        file = tmp_path / "empty.csv"
        file.write_text("", encoding="utf-8")

        analyzer = CsvAnalyzer(str(file))
        with pytest.raises(ValueError, match="vuoto"):
            analyzer.detect_delimiter_and_header("utf-8")

    def test_analyze_single_row(self, tmp_path: Path):
        """File con una sola riga."""
        file = tmp_path / "single.csv"
        file.write_text("name,age,city\n", encoding="utf-8")

        analyzer = CsvAnalyzer(str(file))
        result = analyzer.analyze()

        # Con solo header, dovrebbe rilevarlo
        assert result["header"] == 0
        assert result["columns"] == ["name", "age", "city"]

    def test_analyze_complex_csv(self, tmp_path: Path):
        """CSV complesso con UTF-8 BOM, semicolon, header."""
        file = tmp_path / "complex.csv"
        content = "\ufeffProduct;Price;Quantity\nApple;1,50;100\nBanana;0,80;200\n"
        file.write_text(content, encoding="utf-8-sig")

        result = analyze_csv(str(file))

        assert result["encoding"] == "utf-8-sig"
        assert result["delimiter"] == ";"
        assert result["header"] == 0
        assert "Product" in result["columns"]
        # BOM rimosso
        assert result["columns"][0] == "Product"


class TestEdgeCases:
    """Test per casi limite."""

    def test_very_long_lines(self, tmp_path: Path):
        """Righe molto lunghe."""
        file = tmp_path / "long.csv"
        long_row = ",".join([f"col{i}" for i in range(1000)])
        file.write_text(f"{long_row}\n1,2,3\n", encoding="utf-8")

        analyzer = CsvAnalyzer(str(file))
        result = analyzer.analyze()

        assert len(result["columns"]) == 1000

    def test_special_characters_in_columns(self, tmp_path: Path):
        """Caratteri speciali nei nomi colonne."""
        file = tmp_path / "special.csv"
        file.write_text("name (full),age [years],city/state\nAlice,30,NYC\n", encoding="utf-8")

        result = analyze_csv(str(file))

        assert "name (full)" in result["columns"]
        assert "age [years]" in result["columns"]

    def test_quoted_fields(self, tmp_path: Path):
        """Campi quotati."""
        file = tmp_path / "quoted.csv"
        file.write_text('"name","age","address"\n"Alice","30","123 Main St, NYC"\n', encoding="utf-8")

        result = analyze_csv(str(file))

        assert result["delimiter"] == ","
        # Quotes dovrebbero essere gestite
        assert "name" in result["columns"]

    def test_mixed_delimiters_fallback(self, tmp_path: Path):
        """Mix di delimitatori, usa fallback."""
        file = tmp_path / "mixed.csv"
        # Prima riga con virgole, seconda con tab (inconsistente)
        file.write_text("a,b,c\n1\t2\t3\n", encoding="utf-8")

        analyzer = CsvAnalyzer(str(file))
        result = analyzer.analyze()

        # Dovrebbe scegliere il più comune nel sample
        assert result["delimiter"] in [",", "\t"]
