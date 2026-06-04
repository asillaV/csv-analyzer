"""
Worker per il parsing CSV in un subprocess separato (slow path per file grandi).

Perche' questo modulo e' separato da web_app.py
-----------------------------------------------
Streamlit esegue ``web_app.py`` come modulo ``__main__``. Una funzione definita
li' viene serializzata (pickled) col qualified name ``__main__._parsing_worker``.
Quando il subprocess - avviato con start method ``spawn``, obbligatorio su
Windows - prova a ricostruire il target, cerca ``_parsing_worker`` dentro il
proprio ``__main__``. Nel bundle PyInstaller il vero ``__main__`` e'
``run_app.py``, dove la funzione non esiste, quindi:

    AttributeError: Can't get attribute '_parsing_worker' on <module '__main__'>

Mettendo la funzione in un modulo importabile (``core.csv_parser_worker``) il
pickle usa il path ``core.csv_parser_worker._parsing_worker`` e il processo
figlio la ritrova semplicemente importando il modulo, sia da sorgente sia
dentro l'eseguibile PyInstaller.

Import a livello di modulo: NESSUNO oltre a ``__future__`` e al blocco
``TYPE_CHECKING`` (che non viene eseguito a runtime). Tutti gli import pesanti
(pandas via loader, analyzer, settings) stanno dentro il corpo della funzione,
cosi' l'import del modulo nel processo figlio resta leggero e privo di side
effect che possano interferire con PyInstaller o con il re-spawn di Streamlit.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # solo per i type checker; non importato a runtime
    import multiprocessing as mp


def _parsing_worker(result_queue: "mp.Queue", file_path: str, apply_cleaning: bool) -> None:
    """Analizza e carica il CSV nel processo figlio, restituendo l'esito via coda.

    Tutti gli import sono locali (vedi docstring del modulo): in questo modo
    l'import del modulo nel subprocess non trascina dipendenze pesanti finche'
    la funzione non viene effettivamente eseguita.
    """
    try:
        from core.analyzer import analyze_csv
        from core import settings as app_settings

        # Selezione del loader coerente con web_app.py, leggendo la config
        # effettiva (config.json del pacchetto + override utente) tramite
        # core.settings, che usa resource_path e non dipende dalla cwd.
        use_optimized = True
        try:
            use_optimized = bool(
                app_settings.effective_config()
                .get("performance", {})
                .get("use_optimized_loader", True)
            )
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
    except Exception as exc:  # pragma: no cover - l'esito viene gestito dal caller
        result_queue.put(("error", exc))
