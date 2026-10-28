"""
Corpus di CSV generati per i test di parità del caricamento (Parte 1, F0).

`tests_csv/` è escluso da git, quindi in CI non esiste: questo modulo genera
in una cartella temporanea un insieme ampio di formati reali (encoding,
separatori, locale numerici, token speciali, quoting, righe malformate...).
Ogni voce produce bytes deterministici, così vecchio e nuovo loader leggono
esattamente lo stesso file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

CorpusBuilder = Callable[[], bytes]


def _rows(header: List[str], rows: List[List[str]], sep: str, eol: str = "\n") -> str:
    lines = [sep.join(header)] + [sep.join(r) for r in rows]
    return eol.join(lines) + eol


def _signal(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 100.0
    return np.sin(2 * np.pi * 5 * t) * 100 + rng.normal(0, 3, n)


def _us_basic() -> bytes:
    y = _signal(200, 1)
    rows = [[str(i), f"{v:.4f}", f"{v * 2:.3f}"] for i, v in enumerate(y)]
    return _rows(["time", "ch1", "ch2"], rows, ",").encode("utf-8")


def _eu_semicolon_decimal_comma() -> bytes:
    y = _signal(200, 2)
    rows = [[str(i), f"{v:.4f}".replace(".", ","), f"{v / 3:.2f}".replace(".", ",")] for i, v in enumerate(y)]
    return _rows(["tempo", "canale A", "canale B"], rows, ";").encode("utf-8")


def _eu_thousands_dot() -> bytes:
    y = _signal(150, 3) * 1000
    rows = []
    for i, v in enumerate(y):
        s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        rows.append([str(i), s])
    return _rows(["n", "valore"], rows, ";").encode("utf-8")


def _us_thousands_comma_quoted() -> bytes:
    y = _signal(150, 4) * 1000
    rows = [[str(i), f'"{v:,.2f}"'] for i, v in enumerate(y)]
    return _rows(["n", "amount"], rows, ",").encode("utf-8")


def _tab_separated() -> bytes:
    y = _signal(120, 5)
    rows = [[str(i), f"{v:.5f}", f"{-v:.5f}"] for i, v in enumerate(y)]
    return _rows(["t", "a", "b"], rows, "\t").encode("utf-8")


def _pipe_separated() -> bytes:
    y = _signal(120, 6)
    rows = [[str(i), f"{v:.3f}"] for i, v in enumerate(y)]
    return _rows(["t", "a"], rows, "|").encode("utf-8")


def _utf8_bom_crlf() -> bytes:
    y = _signal(100, 7)
    rows = [[str(i), f"{v:.3f}"] for i, v in enumerate(y)]
    return _rows(["time", "Temperatura °C"], rows, ",", eol="\r\n").encode("utf-8-sig")


def _utf16le_excel_like() -> bytes:
    """Come il file reale: UTF-16 LE BOM, `sep=;`, nomi duplicati, celle vuote, token MSVC."""
    y = _signal(300, 8)
    header = ["SS Counter", "Undeclared []", "Undeclared []", "ACC11 [m/s2]", "LC11 [N]", "Offset [V]"]
    msvc = ["1.#IO", "-1.#IO", "1.#QO", "-1.#R", "1.#J"]
    rows = []
    for i, v in enumerate(y):
        meta = [str(1000 + i), "100", "101"] if i % 50 == 0 else ["", "", ""]
        rows.append(meta + [f"{v:.3f}", f"{v * 10:.3f}", msvc[i % len(msvc)]])
    text = "sep=;\n" + _rows(header, rows, ";")
    return b"\xff\xfe" + text.encode("utf-16-le")


def _utf16be() -> bytes:
    y = _signal(80, 9)
    rows = [[str(i), f"{v:.3f}"] for i, v in enumerate(y)]
    return b"\xfe\xff" + _rows(["t", "v"], rows, ";").encode("utf-16-be")


def _cp1252_accents() -> bytes:
    y = _signal(60, 10)
    rows = [[str(i), f"{v:.2f}".replace(".", ",")] for i, v in enumerate(y)]
    return _rows(["misura", "pressione àèé"], rows, ";").encode("cp1252")


def _datetime_x() -> bytes:
    y = _signal(100, 11)
    rows = [[f"2025-01-01 00:{i // 60:02d}:{i % 60:02d}", f"{v:.3f}"] for i, v in enumerate(y)]
    return _rows(["timestamp", "value"], rows, ",").encode("utf-8")


def _mixed_tokens() -> bytes:
    tokens = ["12.5", '"€ 1.234,50"', "45%", "<5", "n/a", "NaN", "inf", "-inf", "", "  7.25  ", "1e-5", "-3.2E+3"]
    rows = [[str(i), tokens[i % len(tokens)], f"{i * 0.5:.1f}"] for i in range(96)]
    return _rows(["id", "misto", "pulito"], rows, ",").encode("utf-8")


def _no_header() -> bytes:
    y = _signal(100, 12)
    rows = [[f"{i * 0.01:.2f}", f"{v:.4f}", f"{v * 0.5:.4f}"] for i, v in enumerate(y)]
    return "\n".join(",".join(r) for r in rows).encode("utf-8") + b"\n"


def _quoted_with_delimiter() -> bytes:
    rows = [[str(i), f'"nota, con virgola {i}"', f"{i * 1.5:.2f}"] for i in range(60)]
    return _rows(["id", "nota", "valore"], rows, ",").encode("utf-8")


def _trailing_delimiter() -> bytes:
    y = _signal(80, 13)
    lines = ["t;v;"] + [f"{i};{v:.3f};" for i, v in enumerate(y)]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _bad_lines() -> bytes:
    y = _signal(80, 14)
    lines = ["t,a,b"]
    for i, v in enumerate(y):
        lines.append(f"{i},{v:.3f},{v:.1f},EXTRA" if i % 17 == 5 else f"{i},{v:.3f},{v:.1f}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _empty_and_sparse_columns() -> bytes:
    y = _signal(80, 15)
    rows = [[str(i), "", f"{v:.3f}" if i % 3 == 0 else ""] for i, v in enumerate(y)]
    return _rows(["t", "vuota", "sparsa"], rows, ",").encode("utf-8")


def _integers_and_large() -> bytes:
    rows = [[str(i), str(i * 1_000_003), str(2**40 + i)] for i in range(80)]
    return _rows(["idx", "medio", "grande"], rows, ",").encode("utf-8")


def _apostrophe_and_space_thousands() -> bytes:
    rows = []
    for i in range(80):
        v = 1234.5 + i * 1000
        rows.append([str(i), f"{v:,.1f}".replace(",", "'"), f"{v:,.1f}".replace(",", " ")])
    return _rows(["n", "ch_apos", "ch_spazio"], rows, ";").encode("utf-8")


def _negative_spacing_signs() -> bytes:
    tokens = ["- 3.5", "+4.25", "(12.0)", "−7.5", "  -0.001", "+0"]
    rows = [[str(i), tokens[i % len(tokens)]] for i in range(60)]
    return _rows(["n", "segni"], rows, ",").encode("utf-8")


def _single_column() -> bytes:
    y = _signal(100, 16)
    return ("valore\n" + "\n".join(f"{v:.4f}" for v in y) + "\n").encode("utf-8")


def _wide() -> bytes:
    rng = np.random.default_rng(17)
    data = rng.normal(0, 1, (200, 60))
    header = [f"S{j} [V]" for j in range(60)]
    rows = [[f"{x:.4f}" for x in r] for r in data]
    return _rows(header, rows, ";").encode("utf-8")


def _many_rows() -> bytes:
    y = _signal(20_000, 18)
    rows = [[str(i), f"{v:.4f}".replace(".", ","), f"{v * 0.1:.4f}".replace(".", ",")] for i, v in enumerate(y)]
    return _rows(["n", "a", "b"], rows, ";").encode("utf-8")


CORPUS: Dict[str, CorpusBuilder] = {
    "us_basic": _us_basic,
    "eu_semicolon_decimal_comma": _eu_semicolon_decimal_comma,
    "eu_thousands_dot": _eu_thousands_dot,
    "us_thousands_comma_quoted": _us_thousands_comma_quoted,
    "tab_separated": _tab_separated,
    "pipe_separated": _pipe_separated,
    "utf8_bom_crlf": _utf8_bom_crlf,
    "utf16le_excel_like": _utf16le_excel_like,
    "utf16be": _utf16be,
    "cp1252_accents": _cp1252_accents,
    "datetime_x": _datetime_x,
    "mixed_tokens": _mixed_tokens,
    "no_header": _no_header,
    "quoted_with_delimiter": _quoted_with_delimiter,
    "trailing_delimiter": _trailing_delimiter,
    "bad_lines": _bad_lines,
    "empty_and_sparse_columns": _empty_and_sparse_columns,
    "integers_and_large": _integers_and_large,
    "apostrophe_and_space_thousands": _apostrophe_and_space_thousands,
    "negative_spacing_signs": _negative_spacing_signs,
    "single_column": _single_column,
    "wide": _wide,
    "many_rows": _many_rows,
}


def write_corpus(directory: Path) -> Dict[str, Path]:
    """Scrive tutti i CSV del corpus in `directory` e restituisce nome -> percorso."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name, build in CORPUS.items():
        path = directory / f"{name}.csv"
        path.write_bytes(build())
        paths[name] = path
    return paths
