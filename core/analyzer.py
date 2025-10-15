from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

from .logger import LogManager

log = LogManager("analyzer").get_logger()


class CsvAnalyzer:
    """Analizza un file CSV e ne rileva encoding, delimitatore, header e nomi colonne."""

    def __init__(self, file_path: str):
        self.path = file_path

    # ---------- BOM / encoding ----------
    def detect_bom(self) -> str:
        """Rileva il tipo di encoding in base al BOM (Byte Order Mark)."""
        if not os.path.isfile(self.path):
            msg = f"File not found: {self.path}"
            log.error(msg)
            raise FileNotFoundError(msg)

        log.info("Detecting BOM for file: %s", self.path)
        with open(self.path, "rb") as f:
            start = f.read(4)

        if start.startswith(b"\xff\xfe"):
            return "utf-16"  # LE; open(..., 'utf-16') va bene
        if start.startswith(b"\xfe\xff"):
            return "utf-16-be"
        if start.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        return "utf-8"

    # ---------- Delimiter & header ----------
    def detect_delimiter_and_header(self, encoding: str) -> Dict[str, Optional[str]]:
        """Rileva il delimitatore e decide se il file ha un header."""
        delimiters = [",", ";", "\t", "|"]
        # Legge poche righe evitando problemi con caratteri non validi
        with open(self.path, "r", encoding=encoding, errors="ignore") as f:
            lines = [line.strip() for _, line in zip(range(10), f) if line.strip()]

        if not lines:
            raise ValueError("File vuoto o non leggibile.")

        sample = "\n".join(lines)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=delimiters)
            delim = dialect.delimiter
        except Exception:
            # Fallback: scegli il separatore che compare più spesso nel sample
            delim = max(delimiters, key=sample.count)

        def text_ratio(row: str) -> float:
            """Percentuale di celle con almeno una lettera."""
            cells = row.split(delim)
            if not cells:
                return 0.0
            return sum(any(char.isalpha() for char in cell) for cell in cells) / len(cells)

        # Euristica header: confronta prima e seconda riga
        if len(lines) > 1:
            first_ratio = text_ratio(lines[0])
            second_ratio = text_ratio(lines[1])
            if first_ratio == 0 and second_ratio == 0:
                header = None  # nessuna intestazione
            else:
                # Se la seconda ha più testo della prima, header alla riga 1; altrimenti riga 0
                header = 1 if second_ratio > first_ratio else 0
        else:
            # Solo una riga: se contiene testo è header, altrimenti no
            header = 0 if text_ratio(lines[0]) > 0 else None

        return {"encoding": encoding, "delimiter": delim, "header": header}

    # ---------- Estrazione nomi colonne ----------
    @staticmethod
    def _clean_name(name: str) -> str:
        return str(name).replace("\ufeff", "").strip()

    def extract_columns(self, encoding: str, delimiter: str, header: Optional[int]) -> List[str]:
        """Legge la riga dell'header (se presente) e restituisce i nomi colonne."""
        try:
            with open(
                self.path,
                "r",
                encoding=encoding,
                errors="ignore",
                newline="",
            ) as f:
                reader_kwargs = {
                    "delimiter": delimiter if delimiter else ",",
                    "quotechar": '"',
                    "skipinitialspace": True,
                }

                if header is None:
                    # Genera nomi colonna se il file non dichiara un header
                    first_line = next(f)
                    parsed = next(csv.reader([first_line], **reader_kwargs), [])
                    columns = [f"Column_{i}" for i in range(len(parsed))]
                    log.info("No header detected -> Generated %d column names.", len(columns))
                    return columns

                for index, line in enumerate(f):
                    if index == header:
                        parsed = next(csv.reader([line], **reader_kwargs), [])
                        columns = [self._clean_name(value) for value in parsed]
                        log.info("Extracted %d columns from header row %d.", len(columns), header)
                        return columns
        except Exception as exc:
            log.error("Error extracting columns: %s", exc)

        return []

    # ---------- API ----------
    def analyze(self) -> Dict:
        """Esegue analisi completa e restituisce info + nomi colonne."""
        try:
            encoding = self.detect_bom()
            info = self.detect_delimiter_and_header(encoding)
            log.info(
                "Detected encoding=%s, delimiter='%s', header=%s",
                info["encoding"],
                info["delimiter"],
                info["header"],
            )
            info["columns"] = self.extract_columns(
                info["encoding"],
                info["delimiter"],
                info["header"],
            )
            return info
        except Exception as exc:
            log.error("Error analyzing file: %s", exc, exc_info=True)
            raise


# --------- Funzione compatibile con il resto dell'app ---------
def analyze_csv(path_str: str) -> Dict:
    """
    Restituisce un dict coerente con la UI/loader:
    {
      'encoding': str,
      'delimiter': str|None,
      'header': 0|1|None,
      'columns': [str, ...]
    }
    """
    analyzer = CsvAnalyzer(path_str)
    result = analyzer.analyze()

    # Normalizza output (coerente con attese del loader)
    encoding: str = result.get("encoding", "utf-8")
    delimiter: Optional[str] = result.get("delimiter")
    header = result.get("header", 0)  # può essere 0, 1, o None
    columns: List[str] = [CsvAnalyzer._clean_name(value) for value in result.get("columns", [])]

    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "header": header,
        "columns": columns,
    }
