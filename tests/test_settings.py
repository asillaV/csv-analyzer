"""
Test per core.settings: merge config + override utente, persistenza,
rilevamento hardware e profili di prestazioni.
"""

import json

import pytest

import core.settings as settings


@pytest.fixture
def tmp_user_dir(tmp_path, monkeypatch):
    """Reindirizza i file scrivibili delle impostazioni in una cartella temporanea."""
    monkeypatch.setattr(settings, "user_data_dir", lambda: tmp_path)
    yield tmp_path


def test_defaults_structure():
    cfg = settings.effective_config()
    assert "limits" in cfg and "performance" in cfg and "quality" in cfg
    assert "advanced" in cfg["performance"]


def test_save_load_reset_roundtrip(tmp_user_dir):
    assert settings.load_user_settings() == {}

    settings.save_user_settings({"limits": {"max_file_mb": 999}})
    assert settings.user_settings_path().exists()
    assert settings.load_user_settings()["limits"]["max_file_mb"] == 999

    # L'override deve vincere nella config effettiva (deep merge, non sostituzione)
    cfg = settings.effective_config()
    assert cfg["limits"]["max_file_mb"] == 999
    assert "max_rows" in cfg["limits"]  # altri default preservati

    settings.reset_user_settings()
    assert settings.load_user_settings() == {}


def test_deep_merge_preserves_siblings(tmp_user_dir):
    settings.save_user_settings({"performance": {"chunk_size": 12345}})
    perf = settings.effective_config()["performance"]
    assert perf["chunk_size"] == 12345
    # advanced non toccato deve restare presente
    assert "use_pyarrow" in perf["advanced"]


def test_detect_hardware():
    hw = settings.detect_hardware()
    assert isinstance(hw["cores"], int) and hw["cores"] >= 1
    assert hw["ram_gb"] is None or hw["ram_gb"] > 0


@pytest.mark.parametrize(
    "ram,cores,expected",
    [
        (None, 8, "bilanciato"),
        (4.0, 8, "leggero"),
        (16.0, 2, "leggero"),
        (16.0, 4, "bilanciato"),
        (64.0, 16, "qualita"),
    ],
)
def test_recommend_profile(ram, cores, expected):
    assert settings.recommend_profile(ram, cores) == expected


def test_profile_settings_shape_and_workers():
    for prof in ("leggero", "bilanciato", "qualita"):
        vals = settings.profile_settings(prof, cores=8)
        assert "limits" in vals and "performance" in vals
        assert vals["performance"]["advanced"]["max_workers"] >= 1

    # 'qualita' usa tutti i core
    assert settings.profile_settings("qualita", cores=12)["performance"]["advanced"]["max_workers"] == 12
    # 'leggero' ne usa pochi
    assert settings.profile_settings("leggero", cores=12)["performance"]["advanced"]["max_workers"] <= 2


def test_corrupted_user_settings_is_ignored(tmp_user_dir):
    settings.user_settings_path().write_text("{ not valid json", encoding="utf-8")
    # Non deve sollevare: torna ai default
    cfg = settings.effective_config()
    assert cfg["limits"]["max_file_mb"] == settings.DEFAULTS["limits"]["max_file_mb"] or \
        cfg["limits"]["max_file_mb"] > 0
