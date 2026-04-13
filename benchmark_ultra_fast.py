"""
Benchmark loader ultra-fast (v2) vs loader ottimizzato vs loader standard
"""
import time
import tracemalloc
from pathlib import Path

# Importa tutti i loader
from core.loader import load_csv as load_csv_legacy
from core.loader_optimized import load_csv as load_csv_optimized
from core.loader_v2 import load_csv_ultra_fast
from core.analyzer import analyze_csv


def benchmark_loader(csv_path: Path, loader_func, label: str):
    """Benchmark di un singolo loader"""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {label}")
    print(f"File: {csv_path.name} ({csv_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"{'='*70}")

    # Analisi metadati
    metadata = analyze_csv(str(csv_path))

    # Caricamento + pulizia
    tracemalloc.start()
    t0 = time.perf_counter()

    df, report = loader_func(
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
    print("="*70)
    print("CONFRONTO LOADER: STANDARD vs OTTIMIZZATO vs ULTRA-FAST")
    print("="*70)

    # File di test
    test_file = "tests_csv/bench_large_500k.csv"
    csv_path = Path(test_file)

    if not csv_path.exists():
        print(f"\n[ERROR] {test_file} non trovato.")
        print("Esegui prima: python benchmark_loading.py")
        return

    # Test STANDARD
    print("\n" + "=" * 70)
    result_std = benchmark_loader(csv_path, load_csv_legacy, "STANDARD (legacy)")

    # Test OTTIMIZZATO
    print("\n" + "=" * 70)
    result_opt = benchmark_loader(
        csv_path,
        lambda *args, **kwargs: load_csv_optimized(*args, **kwargs, use_optimization=True),
        "OTTIMIZZATO (chunked)"
    )

    # Test ULTRA-FAST
    print("\n" + "=" * 70)
    result_ultra = benchmark_loader(csv_path, load_csv_ultra_fast, "ULTRA-FAST (v2)")

    # Calcola miglioramenti
    speedup_opt = result_std["time_s"] / result_opt["time_s"]
    speedup_ultra = result_std["time_s"] / result_ultra["time_s"]
    speedup_ultra_vs_opt = result_opt["time_s"] / result_ultra["time_s"]

    mem_reduction_opt = result_opt["memory_peak_mb"] / result_std["memory_peak_mb"]
    mem_reduction_ultra = result_ultra["memory_peak_mb"] / result_std["memory_peak_mb"]

    # Riepilogo finale
    print("\n" + "="*70)
    print("RIEPILOGO CONFRONTO")
    print("="*70)
    print(f"{'Loader':<20} {'Tempo':>12} {'RAM Picco':>12} {'Speedup':>10} {'RAM Red.':>10}")
    print("-"*70)
    print(f"{'Standard':<20} {result_std['time_s']:>11.2f}s "
          f"{result_std['memory_peak_mb']:>10.1f}MB "
          f"{'1.00×':>10} {'1.00×':>10}")
    print(f"{'Ottimizzato':<20} {result_opt['time_s']:>11.2f}s "
          f"{result_opt['memory_peak_mb']:>10.1f}MB "
          f"{speedup_opt:>9.2f}× {mem_reduction_opt:>9.2f}×")
    print(f"{'Ultra-Fast':<20} {result_ultra['time_s']:>11.2f}s "
          f"{result_ultra['memory_peak_mb']:>10.1f}MB "
          f"{speedup_ultra:>9.2f}× {mem_reduction_ultra:>9.2f}×")

    print("\n" + "="*70)
    print("ANALISI MIGLIORAMENTI")
    print("="*70)
    print(f"Standard → Ottimizzato:")
    print(f"  Tempo:  {speedup_opt:>5.2f}× {'più veloce' if speedup_opt > 1 else 'più lento'}")
    print(f"  RAM:    {mem_reduction_opt:>5.2f}× {'meno RAM' if mem_reduction_opt < 1 else 'più RAM'}")
    print(f"\nStandard → Ultra-Fast:")
    print(f"  Tempo:  {speedup_ultra:>5.2f}× più veloce")
    print(f"  RAM:    {mem_reduction_ultra:>5.2f}× {'meno RAM' if mem_reduction_ultra < 1 else 'più RAM'}")
    print(f"\nOttimizzato → Ultra-Fast:")
    print(f"  Tempo:  {speedup_ultra_vs_opt:>5.2f}× più veloce")

    print("\n[OK] Benchmark completato!")


if __name__ == "__main__":
    main()
