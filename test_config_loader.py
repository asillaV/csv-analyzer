"""
Script per testare che la configurazione del loader ottimizzato venga letta correttamente
"""
from pathlib import Path
from core.loader_optimized import (
    SIZE_THRESHOLD_MB,
    ROWS_THRESHOLD,
    CHUNK_SIZE,
    SAMPLE_SIZE,
    should_use_sampling
)

print("="*60)
print("CONFIGURAZIONE LOADER OTTIMIZZATO")
print("="*60)
print(f"SIZE_THRESHOLD_MB:    {SIZE_THRESHOLD_MB:>10} MB")
print(f"ROWS_THRESHOLD:       {ROWS_THRESHOLD:>10,} righe")
print(f"CHUNK_SIZE:           {CHUNK_SIZE:>10,} righe")
print(f"SAMPLE_SIZE:          {SAMPLE_SIZE:>10,} righe")
print()

# Testa con file di esempio
test_files = [
    "tests_csv/bench_small_1k.csv",
    "tests_csv/bench_medium_50k.csv",
    "tests_csv/bench_large_500k.csv",
]

print("="*60)
print("TEST FILE DECISION")
print("="*60)

for file_path in test_files:
    path = Path(file_path)
    if not path.exists():
        print(f"\n[SKIP] {file_path} - file non trovato")
        continue

    size_mb = path.stat().st_size / (1024 * 1024)
    use_sampling, reason = should_use_sampling(path)

    print(f"\nFile: {path.name}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Use chunked: {use_sampling}")
    print(f"  Reason: {reason}")

print("\n" + "="*60)
print("TEST COMPLETATO")
print("="*60)
print()
print("Come modificare la configurazione:")
print("  1. Modifica config.json")
print("  2. Riavvia Streamlit (Ctrl+C e poi 'streamlit run web_app.py')")
print("  3. Controlla log: 'Loader optimized config: SIZE_THRESHOLD=...'")
