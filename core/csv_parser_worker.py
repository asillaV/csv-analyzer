"""
Worker function per il parsing CSV in subprocess separato.

Deve stare in un modulo core (non in web_app.py) perché Streamlit carica
web_app.py come __main__: le funzioni definite lì vengono pickled come
__main__.<nome> e il processo figlio non le trova in run_app.py.
Stando in core.csv_parser_worker vengono pickled correttamente e il
processo figlio le importa senza problemi anche dentro il bundle PyInstaller.
"""
from __future__ import annotations

import multiprocessing as mp

from core.analyzer import analyze_csv


def _parsing_worker(result_queue: "mp.Queue", file_path: str, apply_cleaning: bool) -> None:
    """Worker che analizza e carica il CSV, inviando il risultato tramite coda."""
    try:
        import json

        use_optimized = True
        try:
            with open("config.json") as f:
                cfg = json.load(f)
                use_optimized = cfg.get("performance", {}).get("use_optimized_loader", True)
        except Exception:
            pass

        if use_optimized:
            from core.loader_optimized import load_csv
        else:
            from core.loader import load_csv

        meta = analyze_csv(file_path)
        df, cleaning_report = load_csv(
            file_path,
            encoding=meta.get("encoding"),
            delimiter=meta.get("delimiter"),
            header=meta.get("header"),
            apply_cleaning=apply_cleaning,
            return_details=True,
        )
        result_queue.put(("ok", meta, df, cleaning_report))
    except Exception as exc:
        result_queue.put(("error", exc))
