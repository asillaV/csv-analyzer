"""
Test per preset_manager module (Issue #36).
"""

import pytest
from pathlib import Path
import shutil

from core.preset_manager import (
    save_preset,
    load_preset,
    list_presets,
    delete_preset,
    preset_exists,
    get_preset_info,
    create_default_presets,
    PresetError,
    PRESETS_DIR,
    _sanitize_name,
)
from core.signal_tools import FilterSpec, FFTSpec


@pytest.fixture
def clean_presets_dir():
    """Fixture per pulire la directory presets prima e dopo ogni test."""
    # Setup: pulisci prima del test
    if PRESETS_DIR.exists():
        shutil.rmtree(PRESETS_DIR)

    yield

    # Teardown: pulisci dopo il test
    if PRESETS_DIR.exists():
        shutil.rmtree(PRESETS_DIR)


def test_sanitize_name():
    """Test sanitizzazione nomi preset."""
    assert _sanitize_name("Test Preset") == "Test Preset"
    assert _sanitize_name("Test/Preset") == "Test_Preset"
    assert _sanitize_name("Test<>:Preset") == "Test___Preset"
    assert _sanitize_name("   Spazi   Multipli   ") == "Spazi Multipli"

    # Nome troppo lungo (>50 caratteri)
    long_name = "A" * 60
    assert len(_sanitize_name(long_name)) == 50

    # Nome vuoto -> fallback timestamp
    sanitized_empty = _sanitize_name("")
    assert sanitized_empty.startswith("preset_")


def test_save_and_load_preset(clean_presets_dir):
    """Test salvataggio e caricamento preset base."""
    fspec = FilterSpec(
        kind="butter_lp",
        enabled=True,
        order=4,
        cutoff=(50.0, None),
        ma_window=5,
    )
    fftspec = FFTSpec(enabled=True, detrend=True, window="hann")
    manual_fs = 1000.0

    # Salva preset
    save_preset(
        name="Test Preset",
        description="Preset di test",
        fspec=fspec,
        fftspec=fftspec,
        manual_fs=manual_fs,
    )

    assert preset_exists("Test Preset")

    # Carica preset
    loaded = load_preset("Test Preset")

    # Verifica FilterSpec
    assert loaded["filter_spec"].kind == "butter_lp"
    assert loaded["filter_spec"].enabled is True
    assert loaded["filter_spec"].order == 4
    assert loaded["filter_spec"].cutoff == (50.0, None)
    assert loaded["filter_spec"].ma_window == 5

    # Verifica FFTSpec
    assert loaded["fft_spec"].enabled is True
    assert loaded["fft_spec"].detrend is True
    assert loaded["fft_spec"].window == "hann"

    # Verifica manual_fs
    assert loaded["manual_fs"] == 1000.0


def test_save_preset_with_bandpass_filter(clean_presets_dir):
    """Test preset con filtro Butterworth BP (cutoff con due valori)."""
    fspec = FilterSpec(
        kind="butter_bp",
        enabled=True,
        order=4,
        cutoff=(10.0, 100.0),
        ma_window=5,
    )
    fftspec = FFTSpec(enabled=False, detrend=False, window="hann")

    save_preset("BP Test", "Bandpass test", fspec, fftspec, None)
    loaded = load_preset("BP Test")

    assert loaded["filter_spec"].cutoff == (10.0, 100.0)


def test_save_preset_no_manual_fs(clean_presets_dir):
    """Test preset senza fs manuale (None)."""
    fspec = FilterSpec(kind="ma", enabled=True, ma_window=10, order=4, cutoff=None)
    fftspec = FFTSpec(enabled=False, detrend=True, window="hann")

    save_preset("No FS", "No manual FS", fspec, fftspec, manual_fs=None)
    loaded = load_preset("No FS")

    assert loaded["manual_fs"] is None


def test_list_presets(clean_presets_dir):
    """Test listing di preset."""
    # Lista vuota all'inizio
    assert list_presets() == []

    # Crea 3 preset
    fspec = FilterSpec(kind="ma", enabled=False, ma_window=5, order=4, cutoff=None)
    fftspec = FFTSpec(enabled=False, detrend=True, window="hann")

    save_preset("Preset A", "Primo", fspec, fftspec, None)
    save_preset("Preset C", "Terzo", fspec, fftspec, None)
    save_preset("Preset B", "Secondo", fspec, fftspec, None)

    # Lista dovrebbe essere ordinata alfabeticamente
    presets = list_presets()
    assert len(presets) == 3
    assert presets[0]["name"] == "Preset A"
    assert presets[1]["name"] == "Preset B"
    assert presets[2]["name"] == "Preset C"
    assert presets[0]["description"] == "Primo"


def test_delete_preset(clean_presets_dir):
    """Test eliminazione preset."""
    fspec = FilterSpec(kind="ma", enabled=False, ma_window=5, order=4, cutoff=None)
    fftspec = FFTSpec(enabled=False, detrend=True, window="hann")

    save_preset("To Delete", "Test deletion", fspec, fftspec, None)
    assert preset_exists("To Delete")

    # Elimina
    delete_preset("To Delete")
    assert not preset_exists("To Delete")

    # Elimina preset inesistente - dovrebbe sollevare eccezione
    with pytest.raises(PresetError, match="non trovato"):
        delete_preset("To Delete")


def test_preset_exists(clean_presets_dir):
    """Test verifica esistenza preset."""
    fspec = FilterSpec(kind="ma", enabled=False, ma_window=5, order=4, cutoff=None)
    fftspec = FFTSpec(enabled=False, detrend=True, window="hann")

    assert not preset_exists("Nonexistent")

    save_preset("Exists", "Exists test", fspec, fftspec, None)
    assert preset_exists("Exists")


def test_get_preset_info(clean_presets_dir):
    """Test recupero metadata senza caricare preset completo."""
    fspec = FilterSpec(kind="ma", enabled=False, ma_window=5, order=4, cutoff=None)
    fftspec = FFTSpec(enabled=False, detrend=True, window="hann")

    save_preset("Info Test", "Test metadata", fspec, fftspec, None)

    info = get_preset_info("Info Test")
    assert info is not None
    assert info["name"] == "Info Test"
    assert info["description"] == "Test metadata"
    assert "created_at" in info

    # Preset inesistente - dovrebbe sollevare eccezione
    with pytest.raises(PresetError, match="non trovato"):
        get_preset_info("Nonexistent")


def test_load_nonexistent_preset(clean_presets_dir):
    """Test caricamento preset inesistente."""
    with pytest.raises(PresetError, match="non trovato"):
        load_preset("Nonexistent")


def test_load_corrupted_preset(clean_presets_dir, tmp_path):
    """Test caricamento file JSON corrotto."""
    # Crea file corrotto manualmente
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    corrupted_file = PRESETS_DIR / "Corrupted.json"
    corrupted_file.write_text("{ invalid json", encoding='utf-8')

    with pytest.raises(PresetError, match="corrotto"):
        load_preset("Corrupted")


def test_create_default_presets(clean_presets_dir):
    """Test creazione preset di default."""
    create_default_presets()

    presets = list_presets()
    assert len(presets) == 5

    # Verifica che esistano preset specifici
    names = [p["name"] for p in presets]
    assert "Media Mobile 5" in names
    assert "Media Mobile 20" in names
    assert "Butterworth LP 50Hz" in names
    assert "Analisi Vibrazione Completa" in names
    assert "Solo FFT" in names

    # Verifica che un preset possa essere caricato
    loaded = load_preset("Analisi Vibrazione Completa")
    assert loaded["filter_spec"].kind == "butter_bp"
    assert loaded["filter_spec"].cutoff == (10.0, 100.0)
    assert loaded["fft_spec"].enabled is True


def test_create_default_presets_idempotent(clean_presets_dir):
    """Test che create_default_presets() non sovrascriva preset esistenti."""
    # Prima creazione
    create_default_presets()
    presets_before = list_presets()

    # Seconda creazione (dovrebbe skippare preset esistenti)
    create_default_presets()
    presets_after = list_presets()

    assert len(presets_before) == len(presets_after)


def test_save_preset_with_special_characters(clean_presets_dir):
    """Test salvataggio preset con caratteri speciali nel nome."""
    fspec = FilterSpec(kind="ma", enabled=False, ma_window=5, order=4, cutoff=None)
    fftspec = FFTSpec(enabled=False, detrend=True, window="hann")

    # Nome con caratteri speciali
    save_preset("Test<>:/Preset", "Special chars", fspec, fftspec, None)

    # Dovrebbe essere salvato con nome sanitizzato
    assert preset_exists("Test<>:/Preset")  # Internamente usa _sanitize_name

    # Caricamento dovrebbe funzionare
    loaded = load_preset("Test<>:/Preset")
    assert loaded["filter_spec"].kind == "ma"


def test_save_and_load_disabled_specs(clean_presets_dir):
    """Test preset con filtri e FFT disabilitati."""
    fspec = FilterSpec(
        kind="butter_lp",
        enabled=False,  # Disabilitato
        order=2,
        cutoff=(30.0, None),
        ma_window=3,
    )
    fftspec = FFTSpec(
        enabled=False,  # Disabilitato
        detrend=False,
        window="hamming",
    )

    save_preset("Disabled", "Disabled specs", fspec, fftspec, None)
    loaded = load_preset("Disabled")

    assert loaded["filter_spec"].enabled is False
    assert loaded["fft_spec"].enabled is False
    assert loaded["fft_spec"].window == "hamming"
