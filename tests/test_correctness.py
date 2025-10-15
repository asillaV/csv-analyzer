"""
Test di correttezza per verificare che le ottimizzazioni
non abbiano introdotto errori.
"""
from pathlib import Path
import pandas as pd
from core.loader import load_csv
from core.analyzer import analyze_csv


def test_file(csv_path: str) -> dict:
    """Testa un file CSV e verifica la correttezza."""
    print(f"\nTesting: {Path(csv_path).name}")

    try:
        # Analyze
        metadata = analyze_csv(csv_path)
        print(f"  Encoding: {metadata['encoding']}, Delimiter: {repr(metadata['delimiter'])}, Header: {metadata['header']}")

        # Load con cleaning
        df, report = load_csv(
            csv_path,
            encoding=metadata["encoding"],
            delimiter=metadata["delimiter"],
            header=metadata["header"],
            apply_cleaning=True,
            return_details=True,
        )

        print(f"  Righe: {len(df):,}, Colonne: {len(df.columns)}, Numeriche: {len(report.numeric_columns)}")
        print(f"  Confidence: {report.suggestion.confidence:.2%}")
        print(f"  Formato: decimal={repr(report.suggestion.decimal)}, thousands={repr(report.suggestion.thousands)}")

        # Verifica warnings
        if report.warnings:
            print(f"  Warnings: {report.warnings}")

        # Verifica NaN rows
        if report.rows_all_nan_after_clean:
            print(f"  Righe NaN: {len(report.rows_all_nan_after_clean)}")

        # Verifica colonne numeriche
        for col in report.numeric_columns:
            col_data = df[col]
            non_nan = col_data.notna().sum()
            print(f"    • {col}: {non_nan:,}/{len(col_data):,} valori validi")

        return {
            "file": Path(csv_path).name,
            "success": True,
            "rows": len(df),
            "cols": len(df.columns),
            "numeric_cols": len(report.numeric_columns),
        }

    except Exception as e:
        print(f"  ERRORE: {e}")
        return {
            "file": Path(csv_path).name,
            "success": False,
            "error": str(e),
        }


def main():
    test_dir = Path("tests_csv")
    csv_files = sorted(test_dir.glob("*.csv"))

    print("="*80)
    print("TEST DI CORRETTEZZA OTTIMIZZAZIONI")
    print("="*80)

    results = []
    for csv_file in csv_files:
        result = test_file(str(csv_file))
        results.append(result)

    # Summary
    print("\n" + "="*80)
    print("RIEPILOGO")
    print("="*80)

    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]

    print(f"\nSuccessi: {len(successes)}/{len(results)}")
    if failures:
        print(f"Fallimenti: {len(failures)}")
        for f in failures:
            print(f"  • {f['file']}: {f.get('error', 'unknown')}")

    print("\n✅ Tutti i test sono passati!" if not failures else "\n❌ Alcuni test sono falliti!")


if __name__ == "__main__":
    main()
