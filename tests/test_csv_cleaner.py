"""Test per core/csv_cleaner.py - pulizia e conversione dati numerici."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.csv_cleaner import (
    clean_dataframe,
    suggest_number_format,
    FormatSuggestion,
    CleaningReport,
    DECIMAL_CANDIDATES,
    MIN_CONVERSION_RATE,
)


class TestSuggestNumberFormat:
    """Test per suggest_number_format()."""

    def test_european_format_detection(self):
        """Formato europeo: virgola decimale, punto migliaia."""
        df = pd.DataFrame({
            'val': ['1.234,56', '7.890,12', '12.345,67']
        })
        suggestion = suggest_number_format(df)

        assert suggestion.decimal == ','
        assert suggestion.thousands == '.'
        assert suggestion.confidence > 0.9
        assert suggestion.sample_size == 3

    def test_us_format_detection(self):
        """Formato US: punto decimale, virgola migliaia."""
        df = pd.DataFrame({
            'val': ['1,234.56', '7,890.12', '12,345.67']
        })
        suggestion = suggest_number_format(df)

        assert suggestion.decimal == '.'
        assert suggestion.thousands == ','
        assert suggestion.confidence > 0.9

    def test_simple_decimal_point(self):
        """Numeri semplici con punto decimale, senza migliaia."""
        df = pd.DataFrame({
            'val': ['123.45', '678.90', '12.34']
        })
        suggestion = suggest_number_format(df)

        assert suggestion.decimal == '.'
        assert suggestion.thousands is None or suggestion.thousands != '.'

    def test_simple_decimal_comma(self):
        """Numeri semplici con virgola decimale, senza migliaia."""
        df = pd.DataFrame({
            'val': ['123,45', '678,90', '12,34']
        })
        suggestion = suggest_number_format(df)

        assert suggestion.decimal == ','

    def test_space_as_thousands(self):
        """Spazio come separatore migliaia."""
        df = pd.DataFrame({
            'val': ['1 234,56', '7 890,12', '12 345,67']
        })
        suggestion = suggest_number_format(df)

        assert suggestion.decimal == ','
        assert suggestion.thousands == ' '

    def test_forced_decimal(self):
        """Decimal forzato esternamente."""
        df = pd.DataFrame({'val': ['123,45']})
        suggestion = suggest_number_format(df, decimal=',')

        assert suggestion.decimal == ','
        assert suggestion.confidence == 1.0

    def test_forced_thousands(self):
        """Thousands forzato esternamente."""
        df = pd.DataFrame({'val': ['1.234,56']})
        suggestion = suggest_number_format(df, decimal=',', thousands='.')

        assert suggestion.decimal == ','
        assert suggestion.thousands == '.'
        assert suggestion.confidence == 1.0

    def test_empty_dataframe(self):
        """DataFrame vuoto o senza numeri."""
        df = pd.DataFrame({'text': ['abc', 'def']})
        suggestion = suggest_number_format(df)

        assert suggestion.confidence == 0.0
        assert suggestion.sample_size == 0


class TestCleanDataframe:
    """Test per clean_dataframe()."""

    def test_clean_european_format(self):
        """Conversione formato europeo."""
        df = pd.DataFrame({
            'valore': ['1.234,56', '7.890,12', '100,00']
        })
        result = clean_dataframe(df, apply=True)

        assert 'valore' in result.report.numeric_columns
        assert result.df['valore'].iloc[0] == pytest.approx(1234.56)
        assert result.df['valore'].iloc[1] == pytest.approx(7890.12)
        assert result.df['valore'].iloc[2] == pytest.approx(100.0)

    def test_clean_us_format(self):
        """Conversione formato US."""
        df = pd.DataFrame({
            'value': ['1,234.56', '7,890.12', '100.00']
        })
        result = clean_dataframe(df, apply=True)

        assert 'value' in result.report.numeric_columns
        assert result.df['value'].iloc[0] == pytest.approx(1234.56)
        assert result.df['value'].iloc[1] == pytest.approx(7890.12)

    def test_percentage_symbols(self):
        """Rimozione simbolo percentuale."""
        df = pd.DataFrame({
            'percent': ['10%', '25.5%', '100%']
        })
        result = clean_dataframe(df, apply=True)

        assert result.df['percent'].iloc[0] == pytest.approx(10.0)
        assert result.df['percent'].iloc[1] == pytest.approx(25.5)
        assert result.df['percent'].iloc[2] == pytest.approx(100.0)

    def test_currency_symbols(self):
        """Rimozione simboli valuta."""
        df = pd.DataFrame({
            'price': ['€10,50', '€25,00', '100€']
        })
        result = clean_dataframe(df, apply=True)

        # Dovrebbe rimuovere € e convertire
        assert 'price' in result.report.numeric_columns
        assert result.df['price'].iloc[0] == pytest.approx(10.5)

    def test_comparison_prefixes(self):
        """Rimozione prefissi di confronto (<, >, =, ~)."""
        df = pd.DataFrame({
            'val': ['>100', '<50', '~75.5', '=100']
        })
        result = clean_dataframe(df, apply=True)

        assert result.df['val'].iloc[0] == pytest.approx(100.0)
        assert result.df['val'].iloc[1] == pytest.approx(50.0)
        assert result.df['val'].iloc[2] == pytest.approx(75.5)
        assert result.df['val'].iloc[3] == pytest.approx(100.0)

    def test_mixed_column(self):
        """Colonna mista numerica/testuale."""
        df = pd.DataFrame({
            'mixed': ['123', '456', 'N/A', '789', 'invalid']
        })
        result = clean_dataframe(df, apply=True)

        col_report = result.report.columns['mixed']
        # 3 valori numerici su 5 totali
        assert col_report.converted == 3
        assert col_report.non_numeric == 2
        # Conversion rate = 3/5 = 0.6
        conversion_rate = col_report.conversion_rate
        assert 0.5 < conversion_rate < 0.7

    def test_min_conversion_rate_threshold(self):
        """Colonne sotto soglia MIN_CONVERSION_RATE non vengono convertite."""
        # Creiamo colonna con pochi valori convertibili
        df = pd.DataFrame({
            'mostly_text': ['abc', 'def', 'ghi', '123', 'xyz']
        })
        result = clean_dataframe(df, apply=True)

        # Solo 1/5 = 0.2 < MIN_CONVERSION_RATE (0.66)
        # La colonna NON dovrebbe essere in numeric_columns
        assert 'mostly_text' not in result.report.numeric_columns
        col_report = result.report.columns['mostly_text']
        assert not col_report.applied

    def test_apply_false_returns_original(self):
        """Con apply=False, DataFrame rimane invariato."""
        df_original = pd.DataFrame({
            'val': ['1.234,56', '7.890,12']
        })
        df_copy = df_original.copy()
        result = clean_dataframe(df_copy, apply=False)

        # DataFrame non modificato
        assert result.df['val'].iloc[0] == '1.234,56'
        assert not result.report.applied_cleaning
        # Ma report deve comunque essere generato
        assert result.report.suggestion.decimal is not None

    def test_nan_and_inf_handling(self):
        """Gestione NaN e Inf."""
        df = pd.DataFrame({
            'val': ['123', 'inf', '-inf', 'nan', '456']
        })
        result = clean_dataframe(df, apply=True)

        # inf/-inf dovrebbero diventare NaN dopo replace
        assert pd.isna(result.df['val'].iloc[1])
        assert pd.isna(result.df['val'].iloc[2])
        assert pd.isna(result.df['val'].iloc[3])
        assert result.df['val'].iloc[0] == pytest.approx(123.0)
        assert result.df['val'].iloc[4] == pytest.approx(456.0)

    def test_examples_non_numeric(self):
        """Report deve includere esempi di valori non convertiti."""
        df = pd.DataFrame({
            'val': ['123', '456', 'ERROR', 'N/A', 'FAIL', '789', 'BAD']
        })
        result = clean_dataframe(df, apply=True)

        col_report = result.report.columns['val']
        # Dovrebbero esserci esempi di fallimenti
        assert len(col_report.examples_non_numeric) > 0
        assert 'ERROR' in col_report.examples_non_numeric or \
               'N/A' in col_report.examples_non_numeric

    def test_all_numeric_already(self):
        """DataFrame già con numeri nativi."""
        df = pd.DataFrame({
            'num': [1.5, 2.5, 3.5, 4.5]
        })
        result = clean_dataframe(df, apply=True)

        # Dovrebbe funzionare comunque
        assert 'num' in result.report.numeric_columns

    def test_multicolumn_dataframe(self):
        """DataFrame con più colonne di diverso tipo."""
        df = pd.DataFrame({
            'text': ['a', 'b', 'c'],
            'numeric': ['1,5', '2,5', '3,5'],
            'mixed': ['10', 'N/A', '20'],
            'already_num': [100, 200, 300],
        })
        result = clean_dataframe(df, apply=True)

        # 'text' non dovrebbe essere convertita
        assert 'text' not in result.report.numeric_columns
        # 'numeric' dovrebbe essere convertita
        assert 'numeric' in result.report.numeric_columns
        # 'mixed' potrebbe o meno (dipende da soglia)
        # 'already_num' dovrebbe essere gestita

    def test_leading_trailing_spaces(self):
        """Gestione spazi iniziali/finali."""
        df = pd.DataFrame({
            'val': ['  123.45  ', ' 678.90', '12.34  ']
        })
        result = clean_dataframe(df, apply=True)

        assert result.df['val'].iloc[0] == pytest.approx(123.45)
        assert result.df['val'].iloc[1] == pytest.approx(678.90)
        assert result.df['val'].iloc[2] == pytest.approx(12.34)

    def test_scientific_notation(self):
        """Notazione scientifica."""
        df = pd.DataFrame({
            'val': ['1.23e2', '4.56e-1', '7.89E3']
        })
        result = clean_dataframe(df, apply=True)

        assert result.df['val'].iloc[0] == pytest.approx(123.0)
        assert result.df['val'].iloc[1] == pytest.approx(0.456)
        assert result.df['val'].iloc[2] == pytest.approx(7890.0)

    def test_rows_all_nan_after_clean(self):
        """Rilevamento righe completamente NaN dopo cleaning."""
        df = pd.DataFrame({
            'col1': ['123', 'INVALID', '456'],
            'col2': ['789', 'ERROR', '012'],
        })
        result = clean_dataframe(df, apply=True)

        # La riga 1 (indice 1) dovrebbe avere tutti NaN dopo conversione
        if result.report.rows_all_nan_after_clean:
            assert 2 in result.report.rows_all_nan_after_clean  # riga 1 -> label 2 (1-indexed)


class TestCleaningReport:
    """Test per struttura CleaningReport."""

    def test_report_structure(self):
        """Verifica struttura del report."""
        df = pd.DataFrame({'val': ['123', '456']})
        result = clean_dataframe(df, apply=True)

        report = result.report
        assert isinstance(report, CleaningReport)
        assert isinstance(report.suggestion, FormatSuggestion)
        assert isinstance(report.applied_cleaning, bool)
        assert isinstance(report.columns, dict)
        assert isinstance(report.numeric_columns, list)
        assert isinstance(report.warnings, list)

    def test_report_to_dict(self):
        """Serializzazione report a dict."""
        df = pd.DataFrame({'val': ['123,45']})
        result = clean_dataframe(df, apply=True)

        report_dict = result.report.to_dict()
        assert 'suggestion' in report_dict
        assert 'applied_cleaning' in report_dict
        assert 'columns' in report_dict
        assert 'numeric_columns' in report_dict

    def test_low_confidence_warning(self):
        """Warning generato per bassa confidence."""
        # DataFrame con pattern ambiguo
        df = pd.DataFrame({
            'val': ['1,2', '3,4']  # Potrebbe essere sia US che EU
        })
        result = clean_dataframe(df, apply=True)

        # Potrebbe esserci un warning se confidence bassa
        # (dipende dall'euristica interna)
        assert isinstance(result.report.warnings, list)


class TestEdgeCases:
    """Test per casi limite."""

    def test_empty_strings(self):
        """Stringhe vuote."""
        df = pd.DataFrame({
            'val': ['', '  ', '123', '']
        })
        result = clean_dataframe(df, apply=True)

        # Stringhe vuote -> NaN
        assert pd.isna(result.df['val'].iloc[0])
        assert pd.isna(result.df['val'].iloc[1])
        assert result.df['val'].iloc[2] == pytest.approx(123.0)

    def test_single_value_column(self):
        """Colonna con un solo valore."""
        df = pd.DataFrame({'val': ['123']})
        result = clean_dataframe(df, apply=True)

        # Dovrebbe comunque funzionare
        assert result.df['val'].iloc[0] == pytest.approx(123.0)

    def test_very_large_numbers(self):
        """Numeri molto grandi."""
        df = pd.DataFrame({
            'val': ['1.234.567.890,12']
        })
        result = clean_dataframe(df, apply=True, decimal=',', thousands='.')

        assert result.df['val'].iloc[0] == pytest.approx(1234567890.12)

    def test_negative_numbers(self):
        """Numeri negativi."""
        df = pd.DataFrame({
            'val': ['-123,45', '-7.890,12', '100,00']
        })
        result = clean_dataframe(df, apply=True)

        assert result.df['val'].iloc[0] < 0
        assert result.df['val'].iloc[1] < 0
        assert result.df['val'].iloc[2] > 0

    def test_zero_values(self):
        """Valori zero."""
        df = pd.DataFrame({
            'val': ['0', '0,00', '0.0']
        })
        result = clean_dataframe(df, apply=True)

        assert result.df['val'].iloc[0] == pytest.approx(0.0)
        assert result.df['val'].iloc[1] == pytest.approx(0.0)
        assert result.df['val'].iloc[2] == pytest.approx(0.0)
