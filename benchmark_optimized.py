"""
Benchmark del loader ottimizzato vs loader standard
"""
import time
import tracemalloc
from pathlib import Path

# Importa entrambi i loader
from core.loader import load_csv as load_csv_legacy
from core.loader_optimized import load_csv as load_csv_optimized
from core.analyzer import analyze_csv


def benchmark_loader(csv_path: Path, use_optimized: bool, label: str):
    """
    Benchmark di un singolo loader.
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {label}")
    print(f"File: {csv_path.name} ({csv_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"{'='*60}")

    # Analisi metadati
    metadata = analyze_csv(str(csv_path))

    # Caricamento + pulizia
    tracemalloc.start()
    t0 = time.perf_counter()

    if use_optimized:
        df, report = load_csv_optimized(
            str(csv_path),
            encoding=metadata['encoding'],
            delimiter=metadata['delimiter'],
            header=metadata['header'],
            apply_cleaning=True,
            return_details=True,
            use_optimization=True
        )
    else:
        df, report = load_csv_legacy(
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

    # Risultati
    df_mem = df.memory_usage(deep=True).sum() / 1024 / 1024

    print(f"Tempo:           {t1-t0:>8.3f}s")
    print(f"RAM picco:       {mem_peak / 1024 / 1024:>8.1f} MB")
    print(f"RAM DataFrame:   {df_mem:>8.1f} MB")
    print(f"Righe:           {len(df):>8,}")
    print(f"Colonne:         {len(df.columns):>8}")
    print(f"Colonne numeric: {len(report.numeric_columns):>8}")

    return {
        "time_s": t1 - t0,
        "memory_peak_mb": mem_peak / 1024 / 1024,
        "df_memory_mb": df_mem,
        "rows": len(df),
        "cols": len(df.columns)
    }


def main():
    print("="*60)
    print("CONFRONTO LOADER: STANDARD vs OTTIMIZZATO")
    print("="*60)

    # Usa i CSV già creati dal benchmark precedente
    test_files = [
        ("tests_csv/bench_small_1k.csv", "Piccolo (1k righe)"),
        ("tests_csv/bench_medium_50k.csv", "Medio (50k righe)"),
        ("tests_csv/bench_large_500k.csv", "Grande (500k righe)"),
    ]

    results = []

    for csv_file, description in test_files:
        csv_path = Path(csv_file)

        if not csv_path.exists():
            print(f"\n[SKIP] {csv_file} non trovato. Esegui prima benchmark_loading.py")
            continue

        # Test STANDARD
        result_std = benchmark_loader(csv_path, False, f"{description} - STANDARD")

        # Test OTTIMIZZATO
        result_opt = benchmark_loader(csv_path, True, f"{description} - OTTIMIZZATO")

        # Calcola miglioramenti
        time_speedup = result_std["time_s"] / result_opt["time_s"]
        mem_reduction = result_opt["memory_peak_mb"] / result_std["memory_peak_mb"]

        results.append({
            "description": description,
            "std": result_std,
            "opt": result_opt,
            "time_speedup": time_speedup,
            "mem_reduction": mem_reduction
        })

        print(f"\n  -> Speedup tempo:  {time_speedup:>5.2f}x")
        print(f"  -> Riduzione RAM:  {mem_reduction:>5.1f}x ({result_opt['memory_peak_mb']:.1f} MB vs {result_std['memory_peak_mb']:.1f} MB)")

    # Riepilogo finale
    print("\n" + "="*70)
    print("RIEPILOGO CONFRONTO")
    print("="*70)
    print(f"{'Test':<20} {'Tempo STD':>12} {'Tempo OPT':>12} {'Speedup':>10} {'RAM Reduction':>15}")
    print("-"*70)

    for r in results:
        print(f"{r['description']:<20} "
              f"{r['std']['time_s']:>11.2f}s "
              f"{r['opt']['time_s']:>11.2f}s "
              f"{r['time_speedup']:>9.2f}x "
              f"{r['mem_reduction']:>14.2f}x")

    print("\n[OK] Benchmark completato!")


if __name__ == "__main__":
    main()
