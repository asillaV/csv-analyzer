"""
Test per core.paths: risoluzione percorsi in modalità sorgente e "frozen".

Questi test proteggono la logica critica del packaging portabile:
- da sorgente i percorsi puntano alla root del progetto (comportamento storico),
- in modalità frozen gli asset vengono letti da sys._MEIPASS e i file scrivibili
  finiscono in %APPDATA%/AnalizzatoreCSV.
"""

import importlib
from pathlib import Path

import pytest

import core.paths as paths


@pytest.fixture
def reload_paths():
    """Ricarica il modulo dopo ogni test per ripulire eventuali monkeypatch."""
    yield
    importlib.reload(paths)


def test_is_frozen_false_from_source():
    assert paths.is_frozen() is False


def test_source_mode_paths_under_project_root():
    project_root = Path(paths.__file__).resolve().parents[1]
    assert paths.resource_path("config.json") == project_root / "config.json"
    assert paths.outputs_dir() == project_root / "outputs"
    assert paths.logs_dir() == project_root / "logs"
    assert paths.presets_dir() == project_root / "presets"


def test_writable_dirs_exist():
    for d in (paths.outputs_dir(), paths.logs_dir(), paths.presets_dir()):
        assert d.exists() and d.is_dir()


def test_frozen_resource_path_uses_meipass(monkeypatch, tmp_path, reload_paths):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setattr(paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(paths.sys, "_MEIPASS", str(bundle), raising=False)

    assert paths.is_frozen() is True
    assert paths.resource_path("config.json") == bundle / "config.json"


def test_frozen_writable_dirs_use_appdata(monkeypatch, tmp_path, reload_paths):
    appdata = tmp_path / "AppData"
    appdata.mkdir()
    monkeypatch.setattr(paths.sys, "frozen", True, raising=False)
    monkeypatch.setenv("APPDATA", str(appdata))

    base = paths.user_data_dir()
    assert base == appdata / paths.APP_NAME
    assert paths.outputs_dir() == base / "outputs"
    assert paths.logs_dir() == base / "logs"
    assert paths.presets_dir() == base / "presets"
    # Devono essere create
    assert (base / "outputs").is_dir()
    assert (base / "presets").is_dir()
