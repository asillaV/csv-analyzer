"""
Apertura di file locali nella versione desktop (Parte 1, F3).

Nella versione desktop il server Streamlit gira sullo stesso PC dell'utente:
invece di caricare il CSV dal browser (copia intera in RAM, hash, file
temporaneo) l'app lo legge direttamente dal disco.

- `is_desktop_mode()`: vero nell'eseguibile PyInstaller o con
  ANALIZZATORE_DESKTOP=1 (impostato da run_app.py).
- `pick_csv_file()`: finestra nativa "Apri file" di Windows (API Win32 via
  ctypes, senza tkinter, che è escluso dal bundle).
- `recent_files()` / `add_recent_file()`: ultimi file aperti, salvati in
  user_data_dir()/recent_files.json.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional

from .paths import is_frozen, user_data_dir

DESKTOP_ENV = "ANALIZZATORE_DESKTOP"
RECENT_FILE = "recent_files.json"
MAX_RECENT = 10


def is_desktop_mode() -> bool:
    return is_frozen() or os.environ.get(DESKTOP_ENV) == "1"


# ---------------------------------------------------------------------------
# File recenti
# ---------------------------------------------------------------------------
def _recent_path(directory: Optional[Path] = None) -> Path:
    return (Path(directory) if directory is not None else user_data_dir()) / RECENT_FILE


def recent_files(directory: Optional[Path] = None) -> List[str]:
    """Ultimi file aperti ancora esistenti, dal più recente."""
    try:
        items = json.loads(_recent_path(directory).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [p for p in items if isinstance(p, str) and Path(p).is_file()][:MAX_RECENT]


def add_recent_file(path: str | Path, directory: Optional[Path] = None) -> None:
    resolved = str(Path(path).resolve())
    items = [p for p in recent_files(directory) if os.path.normcase(p) != os.path.normcase(resolved)]
    items.insert(0, resolved)
    try:
        target = _recent_path(directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(items[:MAX_RECENT], ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


def normalize_user_path(text: str) -> Optional[Path]:
    """Percorso scritto o incollato dall'utente (anche tra virgolette); None se non è un file."""
    cleaned = (text or "").strip().strip('"').strip("'").strip()
    if not cleaned:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(cleaned)))
    return path if path.is_file() else None


# ---------------------------------------------------------------------------
# Finestra "Apri file" nativa
# ---------------------------------------------------------------------------
def pick_csv_file(initial_dir: Optional[str] = None, title: str = "Apri file CSV") -> Optional[str]:
    """Mostra la finestra nativa di selezione file; None se annullata o non disponibile."""
    if sys.platform != "win32":
        return None
    result: List[Optional[str]] = [None]

    def run() -> None:
        result[0] = _win32_open_dialog(initial_dir, title)

    # Thread dedicato con COM inizializzato: il thread di Streamlit non lo è.
    worker = threading.Thread(target=run, name="csv-file-dialog", daemon=True)
    worker.start()
    worker.join()
    return result[0]


def _win32_open_dialog(initial_dir: Optional[str], title: str) -> Optional[str]:  # pragma: no cover - interattivo
    import ctypes
    from ctypes import wintypes

    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [
            ("lStructSize", wintypes.DWORD),
            ("hwndOwner", wintypes.HWND),
            ("hInstance", wintypes.HINSTANCE),
            ("lpstrFilter", wintypes.LPCWSTR),
            ("lpstrCustomFilter", wintypes.LPWSTR),
            ("nMaxCustFilter", wintypes.DWORD),
            ("nFilterIndex", wintypes.DWORD),
            ("lpstrFile", wintypes.LPWSTR),
            ("nMaxFile", wintypes.DWORD),
            ("lpstrFileTitle", wintypes.LPWSTR),
            ("nMaxFileTitle", wintypes.DWORD),
            ("lpstrInitialDir", wintypes.LPCWSTR),
            ("lpstrTitle", wintypes.LPCWSTR),
            ("Flags", wintypes.DWORD),
            ("nFileOffset", wintypes.WORD),
            ("nFileExtension", wintypes.WORD),
            ("lpstrDefExt", wintypes.LPCWSTR),
            ("lCustData", wintypes.LPARAM),
            ("lpfnHook", ctypes.c_void_p),
            ("lpTemplateName", wintypes.LPCWSTR),
            ("pvReserved", ctypes.c_void_p),
            ("dwReserved", wintypes.DWORD),
            ("FlagsEx", wintypes.DWORD),
        ]

    OFN_NOCHANGEDIR = 0x00000008
    OFN_PATHMUSTEXIST = 0x00000800
    OFN_FILEMUSTEXIST = 0x00001000
    OFN_EXPLORER = 0x00080000
    COINIT_APARTMENTTHREADED = 0x2

    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        ofn = OPENFILENAMEW()
        ofn.lStructSize = ctypes.sizeof(OPENFILENAMEW)
        # In primo piano sopra il browser.
        ofn.hwndOwner = ctypes.windll.user32.GetForegroundWindow()
        ofn.lpstrFilter = "File CSV (*.csv;*.txt)\0*.csv;*.txt\0Tutti i file (*.*)\0*.*\0\0"
        ofn.nFilterIndex = 1
        ofn.lpstrFile = ctypes.cast(buffer, wintypes.LPWSTR)
        ofn.nMaxFile = len(buffer)
        ofn.lpstrInitialDir = initial_dir
        ofn.lpstrTitle = title
        ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR
        if ctypes.windll.comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
            return buffer.value or None
        return None
    finally:
        ole32.CoUninitialize()
