"""
Gestione centralizzata della configurazione effettiva dell'applicazione.

Architettura a due livelli:
  1. config.json incluso nel pacchetto (SOLA LETTURA) -> valori di default.
  2. %APPDATA%/AnalizzatoreCSV/settings.json (SCRIVIBILE) -> override utente.

`effective_config()` fonde i due livelli (l'override utente vince) ed è la
fonte unica da cui tutti i moduli leggono i parametri di performance/limiti.

In più espone:
  - rilevamento hardware del PC (RAM, core),
  - profili di prestazioni predefiniti (Leggero/Bilanciato/Qualità massima),
  - raccomandazione automatica del profilo in base all'hardware.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Optional

from core.paths import resource_path, user_data_dir

__all__ = [
    "DEFAULTS",
    "PROFILE_LABELS",
    "effective_config",
    "load_user_settings",
    "save_user_settings",
    "reset_user_settings",
    "user_settings_path",
    "detect_hardware",
    "recommend_profile",
    "profile_settings",
]

# --- Default "di fabbrica" se config.json manca o è incompleto ---
DEFAULTS: Dict[str, Any] = {
    "quality": {
        "gap_factor_k": 5.0,
        "spike_z": 4.0,
        "min_points": 20,
        "max_examples": 5,
    },
    "performance": {
        "optimize_dtypes": True,
        "aggressive_dtype_optimization": False,
        "use_optimized_loader": True,
        "chunked_loading_threshold_mb": 300,
        "rows_threshold": 100000,
        "chunk_size": 600000,
        "sample_size": 75000,
        "advanced": {
            "use_pyarrow": True,
            "parallel_cleaning": True,
            "early_stop_format_detection": True,
            "skip_nonnumeric_cleaning": True,
            "max_workers": None,
        },
    },
    "limits": {
        "max_file_mb": 2500,
        "max_rows": 20000000,
        "max_cols": 5000,
        "parse_timeout_s": 1200,
    },
}

SETTINGS_FILENAME = "settings.json"

PROFILE_LABELS = {
    "leggero": "Leggero",
    "bilanciato": "Bilanciato",
    "qualita": "Qualità massima",
}


# ----------------------------------------------------------------------
# Merge config bundle + override utente
# ----------------------------------------------------------------------
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Fonde ricorsivamente `override` dentro `base` (l'override vince)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_bundled_config() -> Dict[str, Any]:
    """Legge il config.json incluso nel pacchetto; fallback ai DEFAULTS."""
    cfg = copy.deepcopy(DEFAULTS)
    try:
        path = resource_path("config.json")
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = _deep_merge(cfg, data)
    except Exception:
        # Config corrotto: usa i default senza crashare.
        pass
    return cfg


def user_settings_path():
    """Percorso del file di override utente (scrivibile)."""
    return user_data_dir() / SETTINGS_FILENAME


def load_user_settings() -> Dict[str, Any]:
    """Legge gli override utente; {} se non esistono o sono illeggibili."""
    try:
        path = user_settings_path()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_user_settings(settings: Dict[str, Any]) -> None:
    """Salva gli override utente (sovrascrive il file)."""
    path = user_settings_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


def reset_user_settings() -> None:
    """Rimuove gli override utente (ritorno ai default del pacchetto)."""
    try:
        user_settings_path().unlink(missing_ok=True)
    except Exception:
        pass


def effective_config() -> Dict[str, Any]:
    """
    Config effettiva = config.json (default) + settings.json (override utente).

    Letta fresca ad ogni chiamata (i file sono piccoli) per evitare valori
    obsoleti dopo che l'utente salva nuove impostazioni.
    """
    return _deep_merge(_load_bundled_config(), load_user_settings())


# ----------------------------------------------------------------------
# Hardware e profili
# ----------------------------------------------------------------------
def detect_hardware() -> Dict[str, Any]:
    """
    Rileva RAM totale (GB) e numero di core logici.

    Usa psutil se disponibile; altrimenti fallback con la stdlib (RAM ignota).
    """
    cores = os.cpu_count() or 4
    ram_gb: Optional[float] = None
    try:
        import psutil  # type: ignore

        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        ram_gb = None
    return {"ram_gb": ram_gb, "cores": cores}


def recommend_profile(ram_gb: Optional[float], cores: int) -> str:
    """
    Sceglie un profilo in base all'hardware.

    - RAM ignota -> 'bilanciato' (scelta prudente).
    - RAM < 8 GB o pochi core -> 'leggero'.
    - RAM >= 32 GB e >= 8 core -> 'qualita'.
    - altrimenti -> 'bilanciato'.
    """
    if ram_gb is None:
        return "bilanciato"
    if ram_gb < 8 or cores <= 2:
        return "leggero"
    if ram_gb >= 32 and cores >= 8:
        return "qualita"
    return "bilanciato"


def profile_settings(profile: str, cores: Optional[int] = None) -> Dict[str, Any]:
    """
    Restituisce i valori (performance + limits) per un profilo.

    `cores` permette di tarare il numero di worker sull'hardware reale.
    """
    n_cores = cores or os.cpu_count() or 4

    if profile == "leggero":
        return {
            "performance": {
                "optimize_dtypes": True,
                "aggressive_dtype_optimization": False,
                "use_optimized_loader": True,
                "chunked_loading_threshold_mb": 100,
                "rows_threshold": 50000,
                "chunk_size": 200000,
                "sample_size": 50000,
                "advanced": {
                    "use_pyarrow": True,
                    "parallel_cleaning": False,
                    "max_workers": max(1, min(2, n_cores)),
                },
            },
            "limits": {
                "max_file_mb": 500,
                "max_rows": 2000000,
                "max_cols": 2000,
                "parse_timeout_s": 600,
            },
        }

    if profile == "qualita":
        return {
            "performance": {
                "optimize_dtypes": True,
                "aggressive_dtype_optimization": False,
                "use_optimized_loader": True,
                "chunked_loading_threshold_mb": 500,
                "rows_threshold": 200000,
                "chunk_size": 1000000,
                "sample_size": 100000,
                "advanced": {
                    "use_pyarrow": True,
                    "parallel_cleaning": True,
                    "max_workers": n_cores,
                },
            },
            "limits": {
                "max_file_mb": 5000,
                "max_rows": 50000000,
                "max_cols": 10000,
                "parse_timeout_s": 1800,
            },
        }

    # bilanciato (default)
    return {
        "performance": {
            "optimize_dtypes": True,
            "aggressive_dtype_optimization": False,
            "use_optimized_loader": True,
            "chunked_loading_threshold_mb": 300,
            "rows_threshold": 100000,
            "chunk_size": 600000,
            "sample_size": 75000,
            "advanced": {
                "use_pyarrow": True,
                "parallel_cleaning": True,
                "max_workers": max(1, min(4, n_cores)),
            },
        },
        "limits": {
            "max_file_mb": 2500,
            "max_rows": 20000000,
            "max_cols": 5000,
            "parse_timeout_s": 1200,
        },
    }
