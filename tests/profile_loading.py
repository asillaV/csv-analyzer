"""
Script per profilare le prestazioni del caricamento CSV.
Misura i tempi di esecuzione delle varie fasi della pipeline.
"""
import time
from pathlib import Path
from typing import Dict, List

from core.analyzer import analyze_csv
from core.loader import load_csv


def profile_csv_loading(csv_path: str, apply_cleaning: bool = True) -> Dict[str, float]:
    """Profila tutte le fasi del caricamento CSV."""
    timings = {}

    # 1. Analyze phase
    t0 = time.perf_counter()
    metadata = analyze_csv(csv_path)
    t1 = time.perf_counter()
    timings["analyze"] = t1 - t0

    # 2. Load phase (include cleaning)
    t0 = time.perf_counter()
    df, report = load_csv(
        csv_path,
        encoding=metadata["encoding"],
        delimiter=metadata["delimiter"],
        header=metadata["header"],
        apply_cleaning=apply_cleaning,
        return_details=True,
    )
    t1 = time.perf_counter()
    timings["load_total"] = t1 - t0

    # Calculate total
    timings["total"] = timings["analyze"] + timings["load_total"]

    return timings, df, report


def profile_all_test_files():
    """Profila tutti i file CSV di test."""
    test_dir = Path("tests_csv")
    if not test_dir.exists():
        print(f"Directory {test_dir} non trovata!")
        return

    csv_files = sorted(test_dir.glob("*.csv"))
    if not csv_files:
        print(f"Nessun file CSV trovato in {test_dir}")
        return

    print(f"\n{'='*80}")
    print(f"PROFILING CARICAMENTO CSV - {len(csv_files)} files")
    print(f"{'='*80}\n")

    results: List[Dict] = []

    for csv_file in csv_files:
        print(f"\n{'-'*80}")
        print(f"File: {csv_file.name}")
        print(f"{'-'*80}")

        try:
            # Profile with cleaning
            timings_clean, df, report = profile_csv_loading(str(csv_file), apply_cleaning=True)

            # File info
            file_size_kb = csv_file.stat().st_size / 1024
            rows, cols = df.shape
            numeric_cols = len(report.numeric_columns)

            print(f"Dimensioni: {file_size_kb:.1f} KB | Righe: {rows:,} | Colonne: {cols} | Numeriche: {numeric_cols}")
            print(f"\nTempi (con cleaning):")
            print(f"  • Analyze:    {timings_clean['analyze']*1000:>8.2f} ms")
            print(f"  • Load+Clean: {timings_clean['load_total']*1000:>8.2f} ms")
            print(f"  • TOTALE:     {timings_clean['total']*1000:>8.2f} ms")

            # Profile without cleaning
            timings_no_clean, _, _ = profile_csv_loading(str(csv_file), apply_cleaning=False)
            print(f"\nTempi (senza cleaning):")
            print(f"  • Load only:  {timings_no_clean['load_total']*1000:>8.2f} ms")

            # Calculate cleaning overhead
            cleaning_overhead = timings_clean['load_total'] - timings_no_clean['load_total']
            print(f"  • Overhead cleaning: {cleaning_overhead*1000:>8.2f} ms")

            # Calculate ratios
            analyze_pct = (timings_clean['analyze'] / timings_clean['total']) * 100
            load_pct = (timings_clean['load_total'] / timings_clean['total']) * 100

            print(f"\nDistribuzione tempo:")
            print(f"  • Analyze:    {analyze_pct:>5.1f}%")
            print(f"  • Load+Clean: {load_pct:>5.1f}%")

            results.append({
                "file": csv_file.name,
                "size_kb": file_size_kb,
                "rows": rows,
                "cols": cols,
                "numeric_cols": numeric_cols,
                "analyze_ms": timings_clean['analyze'] * 1000,
                "load_ms": timings_clean['load_total'] * 1000,
                "total_ms": timings_clean['total'] * 1000,
                "cleaning_overhead_ms": cleaning_overhead * 1000,
            })

        except Exception as e:
            print(f"ERRORE: {e}")
            continue

    # Summary
    if results:
        print(f"\n\n{'='*80}")
        print(f"RIEPILOGO")
        print(f"{'='*80}\n")

        total_analyze = sum(r['analyze_ms'] for r in results)
        total_load = sum(r['load_ms'] for r in results)
        total_time = sum(r['total_ms'] for r in results)
        total_cleaning = sum(r['cleaning_overhead_ms'] for r in results)

        print(f"Tempo totale per {len(results)} files:")
        print(f"  • Analyze:    {total_analyze:>10.2f} ms ({total_analyze/total_time*100:.1f}%)")
        print(f"  • Load+Clean: {total_load:>10.2f} ms ({total_load/total_time*100:.1f}%)")
        print(f"  • TOTALE:     {total_time:>10.2f} ms")
        print(f"  • Overhead cleaning: {total_cleaning:>10.2f} ms ({total_cleaning/total_time*100:.1f}%)")

        # Find slowest operations
        print(f"\nFile più lenti:")
        sorted_results = sorted(results, key=lambda x: x['total_ms'], reverse=True)[:3]
        for i, r in enumerate(sorted_results, 1):
            print(f"  {i}. {r['file']:30s} - {r['total_ms']:>8.2f} ms ({r['rows']:,} righe)")


if __name__ == "__main__":
    profile_all_test_files()
