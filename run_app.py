"""
Launcher dell'Analizzatore CSV (interfaccia web Streamlit).

Serve sia da sorgente sia dentro l'eseguibile PyInstaller: dentro un bundle
"congelato" il comando esterno `streamlit run` non esiste, quindi avviamo
Streamlit in-process tramite la sua CLI Python.

Uso:
    python run_app.py              # da sorgente
    AnalizzatoreCSV.exe            # da bundle PyInstaller
"""

from __future__ import annotations

import multiprocessing
import os
import sys

# Indirizzo/porta del server locale. Ci leghiamo SOLO a localhost (127.0.0.1):
# - niente esposizione sulla rete locale -> niente richiesta firewall Windows,
# - l'URL aperto nel browser è http://localhost:<porta> (chiaro per l'utente),
#   invece dell'inutilizzabile 0.0.0.0.
SERVER_ADDRESS = "localhost"
SERVER_PORT = "8501"


def _prepare_user_dirs() -> None:
    """Crea subito le cartelle utente e i preset di default.

    Così l'utente le trova già presenti al primo avvio, senza dover prima
    aprire l'app nel browser (Streamlit esegue lo script solo a sessione
    avviata, quindi senza questo le cartelle comparirebbero in ritardo).
    """
    try:
        from core.paths import outputs_dir, logs_dir, presets_dir, user_data_dir

        user_data_dir()
        outputs_dir()
        logs_dir()
        presets_dir()

        from core.preset_manager import create_default_presets

        create_default_presets()
    except Exception:
        # Non bloccare l'avvio se qualcosa va storto nella preparazione.
        pass


def main() -> None:
    # Streamlit/loader usano processi figli: indispensabile sotto PyInstaller
    # per evitare il re-spawn ricorsivo dell'eseguibile.
    multiprocessing.freeze_support()

    from core.paths import resource_path

    # Apri automaticamente il browser, niente telemetria.
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "false")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    # Forza bind su localhost (anche se l'utente ha variabili d'ambiente residue).
    os.environ["STREAMLIT_SERVER_ADDRESS"] = SERVER_ADDRESS

    _prepare_user_dirs()

    # Il limite di upload di Streamlit (server.maxUploadSize, default 200 MB) è
    # indipendente dal nostro max_file_mb: lo allineiamo alle impostazioni utente
    # così la "dimensione massima file" scelta nel pannello ha davvero effetto.
    max_upload_mb = 2500
    try:
        from core.settings import effective_config

        max_upload_mb = int(effective_config().get("limits", {}).get("max_file_mb", 2500))
    except Exception:
        pass

    target = str(resource_path("web_app.py"))

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        target,
        "--global.developmentMode=false",
        "--server.headless=false",
        f"--server.address={SERVER_ADDRESS}",
        f"--server.port={SERVER_PORT}",
        "--browser.serverAddress=localhost",
        f"--server.maxUploadSize={max_upload_mb}",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
