"""
Script per profilare in dettaglio le funzioni interne della pipeline.
"""
import time
from pathlib import Path
import pandas as pd

# Monkey-patch per instrumentare le funzioni
original_functions = {}


def instrument_function(module, func_name):
    """Wrappa una funzione per misurarne il tempo di esecuzione."""
    func = getattr(module, func_name)
    original_functions[f"{module.__name__}.{func_name}"] = func

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000
        print(f"  [{func_name:30s}] {elapsed:>8.2f} ms")
        return result

    setattr(module, func_name, wrapper)


# Instrumenta le funzioni chiave
from core import analyzer, csv_cleaner

print("Instrumenting functions...")
instrument_function(analyzer.CsvAnalyzer, 'detect_bom')
instrument_function(analyzer.CsvAnalyzer, 'detect_delimiter_and_header')
instrument_function(analyzer.CsvAnalyzer, 'extract_columns')
instrument_function(csv_cleaner, 'suggest_number_format')
instrument_function(csv_cleaner, '_collect_samples')
instrument_function(csv_cleaner, '_evaluate_combination')
instrument_function(csv_cleaner, '_convert_series')
instrument_function(csv_cleaner, '_detect_all_nan_rows')

# Ora importiamo loader che userà le versioni instrumentate
from core.loader import load_csv
from core.analyzer import analyze_csv

print("\n" + "="*80)
print("PROFILING DETTAGLIATO - File 08_big_signal.csv")
print("="*80 + "\n")

test_file = "tests_csv/08_big_signal.csv"

print("\n--- Fase 1: ANALYZE ---")
t0 = time.perf_counter()
metadata = analyze_csv(test_file)
t1 = time.perf_counter()
print(f"\nTotale Analyze: {(t1-t0)*1000:.2f} ms\n")

print("\n--- Fase 2: LOAD + CLEAN ---")
t0 = time.perf_counter()
df, report = load_csv(
    test_file,
    encoding=metadata["encoding"],
    delimiter=metadata["delimiter"],
    header=metadata["header"],
    apply_cleaning=True,
    return_details=True,
)
t1 = time.perf_counter()
print(f"\nTotale Load+Clean: {(t1-t0)*1000:.2f} ms")
print(f"Righe caricate: {len(df):,}")
print(f"Colonne numeriche: {len(report.numeric_columns)}")
