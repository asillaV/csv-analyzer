"""
Benchmark per il caricamento CSV - misura performance attuali
"""
import time
import tracemalloc
from pathlib import Path
import pandas as pd
import numpy as np

from core.loader import load_csv
from core.analyzer import analyze_csv


def create_test_csv(rows: int, cols: int, filename: str) -> Path:
    """Crea un CSV di test con valori numerici europei"""
    print(f"Creando CSV di test: {rows:,} righe × {cols} colonne...")

    # Genera dati casuali
    data = {}
    for i in range(cols):
        # Mix di formati europei per testare il cleaning
        if i % 3 == 0:
            # Numeri con virgola decimale
            data[f"Col_{i}"] = [f"{np.random.uniform(0, 1000):.2f}".replace(".", ",") for _ in range(rows)]
        elif i % 3 == 1:
            # Numeri con separatore migliaia
            data[f"Col_{i}"] = [f"{int(np.random.uniform(1000, 999999)):,}".replace(",", ".") for _ in range(rows)]
        else:
            # Numeri standard
            data[f"Col_{i}"] = np.random.uniform(-100, 100, rows)

    df = pd.DataFrame(data)
    output_path = Path("tests_csv") / filename
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[OK] CSV creato: {output_path} ({size_mb:.2f} MB)\n")
    return output_path


def benchmark_loading(csv_path: Path, label: str):
    """Benchmark completo: analisi + caricamento + pulizia"""
    print(f"{'='*60}")
    print(f"BENCHMARK: {label}")
    print(f"File: {csv_path.name}")
    print(f"{'='*60}\n")

    # Fase 1: Analisi metadati
    print("1. Analisi metadati...")
    tracemalloc.start()
    t0 = time.perf_counter()

    metadata = analyze_csv(str(csv_path))

    t1 = time.perf_counter()
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"   Tempo: {t1-t0:.3f}s")
    print(f"   Memoria: {mem_peak / 1024 / 1024:.2f} MB")
    print(f"   Encoding: {metadata['encoding']}, Delimiter: '{metadata['delimiter']}'")
    print(f"   Colonne: {len(metadata['columns'])}\n")

    # Fase 2: Caricamento + pulizia
    print("2. Caricamento + pulizia numerica...")
    tracemalloc.start()
    t0 = time.perf_counter()

    df, report = load_csv(
        str(csv_path),
        encoding=metadata['encoding'],
        delimiter=metadata['delimiter'],
        header=metadata['header'],
        apply_cleaning=True,
        return_details=True
    )

    t1 = time.perf_counter()
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"   Tempo: {t1-t0:.3f}s")
    print(f"   Memoria picco: {mem_peak / 1024 / 1024:.2f} MB")
    print(f"   Righe caricate: {len(df):,}")
    print(f"   Colonne numeriche: {len(report.numeric_columns)}/{len(df.columns)}")

    # Memoria del DataFrame
    df_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"   Memoria DataFrame: {df_mem:.2f} MB\n")

    return {
        "rows": len(df),
        "cols": len(df.columns),
        "time_s": t1 - t0,
        "memory_mb": mem_peak / 1024 / 1024,
        "df_memory_mb": df_mem
    }


def main():
    print("BENCHMARK CARICAMENTO CSV - Stato Attuale")
    print("="*60)
    print()

    # Test 1: File piccolo (baseline)
    small_csv = create_test_csv(1000, 10, "bench_small_1k.csv")
    result_small = benchmark_loading(small_csv, "File Piccolo (1k righe)")

    # Test 2: File medio
    medium_csv = create_test_csv(50_000, 20, "bench_medium_50k.csv")
    result_medium = benchmark_loading(medium_csv, "File Medio (50k righe)")

    # Test 3: File grande
    large_csv = create_test_csv(500_000, 30, "bench_large_500k.csv")
    result_large = benchmark_loading(large_csv, "File Grande (500k righe)")

    # Riepilogo
    print("\n" + "="*60)
    print("RIEPILOGO BENCHMARK")
    print("="*60)
    print(f"{'Test':<25} {'Righe':>12} {'Tempo':>10} {'RAM':>12}")
    print("-"*60)
    print(f"{'Piccolo (1k)':<25} {result_small['rows']:>12,} {result_small['time_s']:>9.3f}s {result_small['memory_mb']:>10.1f}MB")
    print(f"{'Medio (50k)':<25} {result_medium['rows']:>12,} {result_medium['time_s']:>9.3f}s {result_medium['memory_mb']:>10.1f}MB")
    print(f"{'Grande (500k)':<25} {result_large['rows']:>12,} {result_large['time_s']:>9.3f}s {result_large['memory_mb']:>10.1f}MB")

    # Calcola scaling
    scaling_50k = result_medium['time_s'] / result_small['time_s']
    scaling_500k = result_large['time_s'] / result_medium['time_s']

    print(f"\nScaling factor tempo:")
    print(f"  1k → 50k (50×): {scaling_50k:.1f}× più lento")
    print(f"  50k → 500k (10×): {scaling_500k:.1f}× più lento")

    print("\n[OK] Benchmark completato. Questi dati verranno usati per confronto post-ottimizzazione.")


if __name__ == "__main__":
    main()
