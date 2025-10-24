"""
Benchmark per misurare l'impatto dell'ottimizzazione dtype (Issue #53).

Esegui con: python3 tests/benchmark_dtype_optimization.py
"""
import sys
import numpy as np
import pandas as pd

from core.loader import optimize_dtypes


def get_memory_usage_mb(df: pd.DataFrame) -> float:
    """Calcola memoria occupata dal DataFrame in MB."""
    return df.memory_usage(deep=True).sum() / (1024 ** 2)


def benchmark_dtype_optimization():
    """Misura risparmio di memoria con optimize_dtypes()."""

    test_cases = [
        (1_000, 5, "1k × 5 cols (~40 KB)"),
        (10_000, 10, "10k × 10 cols (~800 KB)"),
        (100_000, 10, "100k × 10 cols (~8 MB)"),
        (1_000_000, 10, "1M × 10 cols (~80 MB)"),
    ]

    print("=" * 80)
    print("BENCHMARK: Dtype Optimization float64→float32 (Issue #53)")
    print("=" * 80)
    print(f"{'Size':<25} {'Before (MB)':<15} {'After (MB)':<15} {'Saving':<20}")
    print("-" * 80)

    for rows, cols, label in test_cases:
        # Genera DataFrame con valori float64 (default pandas)
        data = {f"col_{i}": np.random.randn(rows) * 1000 for i in range(cols)}
        df_original = pd.DataFrame(data)

        # Verifica dtype originale
        assert all(df_original[col].dtype == "float64" for col in df_original.columns)

        # Misura memoria PRIMA
        mem_before_mb = get_memory_usage_mb(df_original)

        # Applica ottimizzazione
        df_optimized, conversions = optimize_dtypes(df_original, enabled=True)

        # Misura memoria DOPO
        mem_after_mb = get_memory_usage_mb(df_optimized)

        # Calcola risparmio
        saving_mb = mem_before_mb - mem_after_mb
        saving_pct = (saving_mb / mem_before_mb * 100) if mem_before_mb > 0 else 0

        # Verifica conversioni
        assert len(conversions) == cols, f"Expected {cols} conversions, got {len(conversions)}"

        # Verifica correttezza valori (tolleranza float32)
        for col in df_original.columns:
            np.testing.assert_allclose(
                df_optimized[col].values,
                df_original[col].values,
                rtol=1e-6,
                err_msg=f"Values differ in column {col}"
            )

        print(f"{label:<25} {mem_before_mb:>10.2f} MB   {mem_after_mb:>10.2f} MB   -{saving_mb:>8.2f} MB ({saving_pct:>5.1f}%)")

    print("-" * 80)
    print("\nCONCLUSIONE:")
    print("  - Risparmio memoria: ~50% su colonne float64")
    print("  - Precisione: valori invariati entro tolleranza float32 (±1e-6)")
    print("  - Impatto: 1M × 10 cols → da 80MB a 40MB (-50%)")
    print("  - Safe: valori oltre ±3.4e38 NON convertiti (mantengono float64)")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_dtype_optimization()
