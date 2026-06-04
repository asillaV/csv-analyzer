"""
Risoluzione centralizzata dei percorsi (single source of truth).

Gestisce due scenari:
  - Esecuzione da sorgente (sviluppo/test): i percorsi sono relativi alla root
    del progetto, come nel comportamento storico.
  - Esecuzione "congelata" (PyInstaller, sys.frozen): gli asset di sola lettura
    vivono nel bundle (sys._MEIPASS), mentre i file scrivibili vengono
    reindirizzati in una cartella utente (%APPDATA%/AnalizzatoreCSV) perché la
    cartella del bundle può essere di sola lettura.

Convenzione:
  - resource_path(): asset di SOLA LETTURA inclusi nel pacchetto
    (config.json, assets/, web_app.py, ...).
  - outputs_dir()/logs_dir()/presets_dir(): cartelle SCRIVIBILI a runtime.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "AnalizzatoreCSV"

__all__ = [
    "is_frozen",
    "resource_path",
    "user_data_dir",
    "outputs_dir",
    "logs_dir",
    "presets_dir",
]


def is_frozen() -> bool:
    """True quando l'app gira come eseguibile PyInstaller."""
    return bool(getattr(sys, "frozen", False))


def _bundle_root() -> Path:
    """Root da cui leggere gli asset inclusi nel pacchetto."""
    if is_frozen():
        # In --onedir PyInstaller espone gli asset in sys._MEIPASS.
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    # Da sorgente: .../core/paths.py -> root del progetto = parent di 'core'.
    return Path(__file__).resolve().parents[1]


def resource_path(rel: str) -> Path:
    """Percorso assoluto di un asset di sola lettura incluso nel pacchetto."""
    return _bundle_root() / rel


def user_data_dir() -> Path:
    """
    Cartella base per i file scrivibili a runtime.

    - Frozen: %APPDATA%/AnalizzatoreCSV (sempre scrivibile, indipendente dalla
      posizione del bundle).
    - Sorgente: root del progetto (comportamento storico, non rompe i test).
    """
    if is_frozen():
        base_env = os.environ.get("APPDATA") or os.environ.get("XDG_DATA_HOME")
        base = (Path(base_env) if base_env else Path.home()) / APP_NAME
    else:
        base = Path(__file__).resolve().parents[1]
    base.mkdir(parents=True, exist_ok=True)
    return base


def outputs_dir() -> Path:
    """Cartella per plot/report generati."""
    d = user_data_dir() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    """Cartella per i file di log."""
    d = user_data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def presets_dir() -> Path:
    """Cartella per i preset di analisi salvati."""
    d = user_data_dir() / "presets"
    d.mkdir(parents=True, exist_ok=True)
    return d
