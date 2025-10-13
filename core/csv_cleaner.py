from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import re

import pandas as pd


DECIMAL_CANDIDATES: Tuple[str, ...] = (",", ".")
THOUSANDS_CANDIDATES: Tuple[Optional[str], ...] = (
    None,
    " ",
    "\u00A0",
    "'",
    ",",
    ".",
)
TRAILING_SYMBOLS: Tuple[str, ...] = ("%", "\u2030", "\u00B0")
PREFIX_SYMBOLS_RE = re.compile(r"^[<>~=]+")
NUMERIC_TOKEN_RE = re.compile(r"[0-9]")
VALIDATED_NUMBER_RE = re.compile(
    r"^[+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?$"
)

MIN_CANDIDATE_VALUES = 3
MIN_CONVERSION_RATE = 0.66
MAX_SAMPLE_VALUES = 640
MAX_NON_NUMERIC_EXAMPLES = 5


@dataclass
class FormatSuggestion:
    decimal: Optional[str]
    thousands: Optional[str]
    confidence: float
    sample_size: int

    def to_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)


@dataclass
class ColumnReport:
    total: int
    candidate_numeric: int
    converted: int
    non_numeric: int
    conversion_rate: float
    dtype: str
    applied: bool
    examples_non_numeric: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "total": self.total,
            "candidate_numeric": self.candidate_numeric,
            "converted": self.converted,
            "non_numeric": self.non_numeric,
            "conversion_rate": self.conversion_rate,
            "dtype": self.dtype,
            "applied": self.applied,
            "examples_non_numeric": self.examples_non_numeric,
        }


@dataclass
class CleaningReport:
    suggestion: FormatSuggestion
    applied_cleaning: bool
    columns: Dict[str, ColumnReport]
    numeric_columns: List[str]
    rows_all_nan_after_clean: List[int]
    warnings: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "suggestion": self.suggestion.to_dict(),
            "applied_cleaning": self.applied_cleaning,
            "columns": {name: report.to_dict() for name, report in self.columns.items()},
            "numeric_columns": self.numeric_columns,
            "rows_all_nan_after_clean": self.rows_all_nan_after_clean,
            "warnings": self.warnings,
        }


@dataclass
class SanitizedResult:
    df: pd.DataFrame
    report: CleaningReport


def clean_dataframe(
    df: pd.DataFrame,
    apply: bool = True,
    decimal: Optional[str] = None,
    thousands: Optional[str] = None,
) -> SanitizedResult:
    """
    Pulisce un DataFrame di stringhe rimuovendo separatori migliaia/decimal,
    coercendo a numerico le colonne idonee e producendo un report.
    """
    df_input = df.copy()
    suggestion = suggest_number_format(df_input, decimal=decimal, thousands=thousands)

    numeric_columns: List[str] = []
    columns_report: Dict[str, ColumnReport] = {}
    warnings: List[str] = []

    if suggestion.sample_size == 0:
        warnings.append("Impossibile stimare il formato: nessun valore numerico trovato nel campione.")

    if suggestion.confidence < 0.25:
        warnings.append(
            f"Bassa confidenza nel formato numerico stimato ({suggestion.confidence:.0%})."
        )

    df_out = df_input.copy()

    for column in df_input.columns:
        series = df_input[column]
        series_str = series.astype("string")
        total = len(series_str)

        candidate_mask = series_str.fillna("").str.contains(NUMERIC_TOKEN_RE, regex=True)
        candidate_count = int(candidate_mask.sum())
        if candidate_count == 0:
            columns_report[column] = ColumnReport(
                total=total,
                candidate_numeric=0,
                converted=0,
                non_numeric=0,
                conversion_rate=0.0,
                dtype=str(series.dtype),
                applied=False,
                examples_non_numeric=[],
            )
            continue

        numeric_series = _convert_series(
            series_str, suggestion.decimal, suggestion.thousands
        )
        converted_mask = candidate_mask & numeric_series.notna()
        converted_count = int(converted_mask.sum())
        non_numeric_count = candidate_count - converted_count
        conversion_rate = converted_count / candidate_count if candidate_count else 0.0

        examples = _collect_examples(
            series_str, candidate_mask, converted_mask, limit=MAX_NON_NUMERIC_EXAMPLES
        )

        should_apply = (
            apply
            and candidate_count >= MIN_CANDIDATE_VALUES
            and conversion_rate >= MIN_CONVERSION_RATE
        )

        if should_apply:
            numeric_columns.append(column)
            df_out[column] = numeric_series
            dtype_repr = str(df_out[column].dtype)
        else:
            dtype_repr = str(series.dtype)

        columns_report[column] = ColumnReport(
            total=total,
            candidate_numeric=candidate_count,
            converted=converted_count,
            non_numeric=non_numeric_count,
            conversion_rate=conversion_rate,
            dtype=dtype_repr,
            applied=should_apply,
            examples_non_numeric=examples,
        )

    rows_all_nan = _detect_all_nan_rows(df_input, df_out, numeric_columns)

    report = CleaningReport(
        suggestion=suggestion,
        applied_cleaning=apply,
        columns=columns_report,
        numeric_columns=numeric_columns,
        rows_all_nan_after_clean=rows_all_nan,
        warnings=warnings,
    )

    return SanitizedResult(df=df_out, report=report)


def suggest_number_format(
    df: pd.DataFrame,
    decimal: Optional[str] = None,
    thousands: Optional[str] = None,
) -> FormatSuggestion:
    """
    Stima decimal separator e separatore migliaia analizzando un campione di valori.
    Se decimal/thousands vengono forniti, vengono rispettati.
    """
    if decimal is not None or thousands is not None:
        forced_decimal = decimal if decimal is not None else "."
        forced_thousands = thousands if thousands != forced_decimal else None
        return FormatSuggestion(
            decimal=forced_decimal,
            thousands=forced_thousands,
            confidence=1.0,
            sample_size=0,
        )

    samples = _collect_samples(df, max_samples=MAX_SAMPLE_VALUES)
    if not samples:
        return FormatSuggestion(decimal=".", thousands=None, confidence=0.0, sample_size=0)

    best_decimal: Optional[str] = None
    best_thousands: Optional[str] = None
    best_score = -1

    for dec in DECIMAL_CANDIDATES:
        for thou in THOUSANDS_CANDIDATES:
            if thou is not None and thou == dec:
                continue
            score = _score_combination(samples, dec, thou)
            if score > best_score:
                best_score = score
                best_decimal = dec
                best_thousands = thou

    confidence = best_score / len(samples) if samples else 0.0
    return FormatSuggestion(
        decimal=best_decimal or ".",
        thousands=best_thousands,
        confidence=confidence,
        sample_size=len(samples),
    )


def _collect_samples(df: pd.DataFrame, max_samples: int) -> List[str]:
    samples: List[str] = []
    for column in df.columns:
        series = df[column].astype("string")
        for value in series.head(max_samples * 4):
            if value is None or pd.isna(value):
                continue
            text = str(value).strip()
            if not text:
                continue
            if NUMERIC_TOKEN_RE.search(text):
                samples.append(text)
            if len(samples) >= max_samples:
                return samples
    return samples


def _score_combination(samples: Sequence[str], decimal: str, thousands: Optional[str]) -> int:
    successes = 0
    for value in samples:
        normalized = _normalize_value(value, decimal, thousands)
        if normalized is None:
            continue
        if VALIDATED_NUMBER_RE.match(normalized):
            successes += 1
    return successes


def _convert_series(
    series: pd.Series, decimal: Optional[str], thousands: Optional[str]
) -> pd.Series:
    normalized_values = [
        _normalize_value(value, decimal, thousands) for value in series.to_numpy()
    ]
    numeric_series = pd.to_numeric(normalized_values, errors="coerce")
    return pd.Series(numeric_series, index=series.index, dtype="float64")


def _normalize_value(
    value: Optional[str], decimal: Optional[str], thousands: Optional[str]
) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None

    text = text.replace("\u202F", " ").replace("\u2009", " ")
    text = PREFIX_SYMBOLS_RE.sub("", text)

    for symbol in TRAILING_SYMBOLS:
        if text.endswith(symbol):
            text = text[: -len(symbol)].strip()

    if thousands:
        if thousands.strip() == "":
            text = re.sub(r"\s+", "", text)
        else:
            text = text.replace(thousands, "")

    text = text.replace(" ", "")

    if decimal and decimal != ".":
        text = text.replace(decimal, ".")

    cleaned = re.sub(r"[^0-9+\-\.eE]", "", text)

    if cleaned.count(".") > 1 and "e" not in cleaned.lower():
        return None

    if cleaned in {"", "+", "-", "+.", "-.", "."}:
        return None

    return cleaned


def _collect_examples(
    series: pd.Series,
    candidate_mask: pd.Series,
    converted_mask: pd.Series,
    limit: int,
) -> List[str]:
    failed_mask = candidate_mask & ~converted_mask
    failed_values = series[failed_mask].head(limit)
    return [str(v) for v in failed_values.tolist()]


def _detect_all_nan_rows(
    df_raw: pd.DataFrame, df_clean: pd.DataFrame, numeric_columns: Sequence[str]
) -> List[int]:
    if not numeric_columns:
        return []

    raw_numeric_subset = df_raw[numeric_columns].astype("string")
    has_digit = raw_numeric_subset.fillna("").applymap(
        lambda v: bool(NUMERIC_TOKEN_RE.search(str(v))) if v is not pd.NA else False
    )

    clean_numeric = df_clean[numeric_columns]
    all_nan_after = clean_numeric.isna().all(axis=1)
    had_digit_before = has_digit.any(axis=1)

    problematic_rows = clean_numeric.index[all_nan_after & had_digit_before]
    return [int(i) + 1 for i in problematic_rows[:50]]
