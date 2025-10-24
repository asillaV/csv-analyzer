"""
Preset Manager - Sistema di salvataggio/caricamento configurazioni filtri e FFT.

Issue #36: Permette agli utenti di salvare e riutilizzare configurazioni di analisi.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict
from datetime import datetime

from core.signal_tools import FilterSpec, FFTSpec


# Schema version per compatibilità futura
PRESET_VERSION = "1.0"

# Directory per i preset
PRESETS_DIR = Path(__file__).parent.parent / "presets"


class PresetError(Exception):
    """Eccezione per errori nella gestione dei preset."""
    pass


def _ensure_presets_dir() -> None:
    """Crea la directory presets se non esiste."""
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_name(name: str) -> str:
    """
    Sanitizza il nome del preset per uso come filename.
    Rimuove caratteri non validi e limita lunghezza.
    """
    # Rimuovi caratteri speciali pericolosi
    invalid_chars = '<>:"/\\|?*'
    sanitized = "".join(c if c not in invalid_chars else "_" for c in name)

    # Rimuovi spazi multipli e trim
    sanitized = " ".join(sanitized.split())

    # Limita lunghezza (max 50 caratteri per il nome)
    if len(sanitized) > 50:
        sanitized = sanitized[:50].strip()

    # Fallback se vuoto
    if not sanitized or sanitized.isspace():
        sanitized = f"preset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return sanitized


def _preset_path(name: str) -> Path:
    """Ritorna il path completo del file preset."""
    safe_name = _sanitize_name(name)
    return PRESETS_DIR / f"{safe_name}.json"


def save_preset(
    name: str,
    description: str,
    fspec: FilterSpec,
    fftspec: FFTSpec,
    manual_fs: Optional[float]
) -> None:
    """
    Salva un preset con le configurazioni fornite.

    Args:
        name: Nome del preset (verrà sanitizzato)
        description: Descrizione testuale del preset
        fspec: Configurazione filtro
        fftspec: Configurazione FFT
        manual_fs: Frequenza di campionamento manuale (None = auto)

    Raises:
        PresetError: Se il salvataggio fallisce
    """
    _ensure_presets_dir()

    try:
        # Converti dataclass a dict, gestendo tuple → list per JSON
        filter_dict = asdict(fspec)
        if isinstance(filter_dict.get('cutoff'), tuple):
            filter_dict['cutoff'] = list(filter_dict['cutoff'])

        fft_dict = asdict(fftspec)

        # Crea struttura preset
        preset_data = {
            "version": PRESET_VERSION,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "manual_fs": manual_fs,
            "filter": filter_dict,
            "fft": fft_dict,
        }

        # Salva su file
        path = _preset_path(name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(preset_data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        raise PresetError(f"Impossibile salvare preset '{name}': {e}") from e


def load_preset(name: str) -> Dict[str, any]:
    """
    Carica un preset dal file.

    Args:
        name: Nome del preset da caricare

    Returns:
        Dict con chiavi:
            - manual_fs: Optional[float]
            - filter_spec: FilterSpec
            - fft_spec: FFTSpec

    Raises:
        PresetError: Se il file non esiste o il formato non è valido
    """
    path = _preset_path(name)

    if not path.exists():
        raise PresetError(f"Preset '{name}' non trovato")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Valida versione
        version = data.get("version", "unknown")
        if version != PRESET_VERSION:
            # Per ora accettiamo solo versione 1.0
            # In futuro qui ci sarà migrazione automatica
            pass  # Tollerante per ora

        # Ricostruisci FilterSpec
        filter_dict = data.get("filter", {})
        # Converti tuple da lista (JSON non supporta tuple)
        cutoff = filter_dict.get("cutoff")
        if isinstance(cutoff, list):
            if len(cutoff) == 2:
                cutoff = tuple(cutoff)
            elif len(cutoff) == 1:
                # Singolo valore per LP/HP
                cutoff = (cutoff[0], None)
            else:
                cutoff = None
        filter_dict["cutoff"] = cutoff

        filter_spec = FilterSpec(**filter_dict)

        # Ricostruisci FFTSpec
        fft_dict = data.get("fft", {})
        fft_spec = FFTSpec(**fft_dict)

        # Estrai manual_fs
        manual_fs = data.get("manual_fs")

        return {
            'manual_fs': manual_fs,
            'filter_spec': filter_spec,
            'fft_spec': fft_spec
        }

    except json.JSONDecodeError as e:
        raise PresetError(f"Preset '{name}' corrotto (JSON invalido): {e}") from e
    except (KeyError, TypeError) as e:
        raise PresetError(f"Preset '{name}' ha formato invalido: {e}") from e
    except Exception as e:
        raise PresetError(f"Impossibile leggere preset '{name}': {e}") from e


def list_presets() -> List[Dict[str, str]]:
    """
    Ritorna lista di tutti i preset disponibili con metadata.

    Returns:
        Lista di dict con keys: name, description, created_at, path
        Ordinata per nome (alfabetico)
    """
    _ensure_presets_dir()

    presets = []
    for path in PRESETS_DIR.glob("*.json"):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            presets.append({
                "name": data.get("name", path.stem),
                "description": data.get("description", ""),
                "created_at": data.get("created_at", ""),
                "path": str(path),
            })
        except Exception:
            # Salta file corrotti
            continue

    # Ordina per nome
    presets.sort(key=lambda x: x["name"].lower())

    return presets


def delete_preset(name: str) -> None:
    """
    Elimina un preset.

    Args:
        name: Nome del preset da eliminare

    Raises:
        PresetError: Se il preset non esiste o l'eliminazione fallisce
    """
    path = _preset_path(name)

    if not path.exists():
        raise PresetError(f"Preset '{name}' non trovato.")

    try:
        path.unlink()
    except Exception as e:
        raise PresetError(f"Impossibile eliminare preset '{name}': {e}") from e


def preset_exists(name: str) -> bool:
    """Verifica se un preset esiste."""
    return _preset_path(name).exists()


def get_preset_info(name: str) -> Dict[str, str]:
    """
    Ritorna solo i metadata di un preset senza caricare filter/fft.
    Utile per preview veloce.

    Args:
        name: Nome del preset

    Returns:
        Dict con chiavi: name, description, created_at

    Raises:
        PresetError: Se il preset non esiste o è corrotto
    """
    path = _preset_path(name)

    if not path.exists():
        raise PresetError(f"Preset '{name}' non trovato.")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return {
            "name": data.get("name", name),
            "description": data.get("description", ""),
            "created_at": data.get("created_at", "Unknown"),
        }
    except json.JSONDecodeError as e:
        raise PresetError(f"Preset '{name}' corrotto (JSON invalido): {e}") from e
    except Exception as e:
        raise PresetError(f"Impossibile leggere preset '{name}': {e}") from e


# ==================== Preset di Default ====================

def create_default_presets() -> None:
    """
    Crea preset di default se non esistono.
    Chiamare all'avvio dell'app per garantire preset base.
    """
    defaults = [
        {
            "name": "Media Mobile 5",
            "description": "Smoothing leggero con media mobile a 5 campioni",
            "fspec": FilterSpec(
                kind="ma",
                enabled=True,
                order=4,
                cutoff=None,
                ma_window=5,
            ),
            "fftspec": FFTSpec(enabled=False, detrend=False, window="hann"),
            "manual_fs": None,
        },
        {
            "name": "Media Mobile 20",
            "description": "Smoothing intenso con media mobile a 20 campioni",
            "fspec": FilterSpec(
                kind="ma",
                enabled=True,
                order=4,
                cutoff=None,
                ma_window=20,
            ),
            "fftspec": FFTSpec(enabled=False, detrend=False, window="hann"),
            "manual_fs": None,
        },
        {
            "name": "Butterworth LP 50Hz",
            "description": "Filtro passa-basso Butterworth a 50Hz, ordine 4",
            "fspec": FilterSpec(
                kind="butter_lp",
                enabled=True,
                order=4,
                cutoff=(50.0, None),
                ma_window=5,
            ),
            "fftspec": FFTSpec(enabled=False, detrend=False, window="hann"),
            "manual_fs": None,
        },
        {
            "name": "Analisi Vibrazione Completa",
            "description": "Butterworth BP 10-100Hz + FFT con detrend e finestra Hann",
            "fspec": FilterSpec(
                kind="butter_bp",
                enabled=True,
                order=4,
                cutoff=(10.0, 100.0),
                ma_window=5,
            ),
            "fftspec": FFTSpec(enabled=True, detrend=True, window="hann"),
            "manual_fs": None,
        },
        {
            "name": "Solo FFT",
            "description": "Analisi FFT con detrend e finestra Hann, nessun filtro",
            "fspec": FilterSpec(
                kind="ma",
                enabled=False,
                order=4,
                cutoff=None,
                ma_window=5,
            ),
            "fftspec": FFTSpec(enabled=True, detrend=True, window="hann"),
            "manual_fs": None,
        },
    ]

    for preset in defaults:
        name = preset["name"]
        if not preset_exists(name):
            try:
                save_preset(
                    name=name,
                    description=preset["description"],
                    fspec=preset["fspec"],
                    fftspec=preset["fftspec"],
                    manual_fs=preset["manual_fs"],
                )
            except PresetError:
                # Salta se creazione fallisce
                pass
