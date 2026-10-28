"""Test di core.local_files (apertura file nella versione desktop, Parte 1 F3)."""
from __future__ import annotations

import json
import threading

import pytest

from core import local_files

pytestmark = pytest.mark.unit


def test_desktop_mode_from_env(monkeypatch):
    monkeypatch.delenv(local_files.DESKTOP_ENV, raising=False)
    monkeypatch.setattr(local_files, "is_frozen", lambda: False)
    assert not local_files.is_desktop_mode()
    monkeypatch.setenv(local_files.DESKTOP_ENV, "1")
    assert local_files.is_desktop_mode()
    monkeypatch.delenv(local_files.DESKTOP_ENV)
    monkeypatch.setattr(local_files, "is_frozen", lambda: True)
    assert local_files.is_desktop_mode()


def test_recent_files_order_dedupe_and_limit(tmp_path):
    store = tmp_path / "store"
    files = []
    for i in range(local_files.MAX_RECENT + 3):
        f = tmp_path / f"f{i}.csv"
        f.write_text("a\n1\n", encoding="utf-8")
        files.append(f)
        local_files.add_recent_file(f, directory=store)
    local_files.add_recent_file(files[5], directory=store)
    recent = local_files.recent_files(directory=store)
    assert len(recent) == local_files.MAX_RECENT
    assert recent[0] == str(files[5].resolve())
    assert recent.count(str(files[5].resolve())) == 1


def test_recent_files_skip_missing_and_bad_json(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    keep = tmp_path / "keep.csv"
    keep.write_text("a\n", encoding="utf-8")
    payload = [str(tmp_path / "gone.csv"), str(keep), 3]
    (store / local_files.RECENT_FILE).write_text(json.dumps(payload), encoding="utf-8")
    assert local_files.recent_files(directory=store) == [str(keep)]
    (store / local_files.RECENT_FILE).write_text("{rotto", encoding="utf-8")
    assert local_files.recent_files(directory=store) == []
    assert local_files.recent_files(directory=tmp_path / "vuota") == []


def test_normalize_user_path(tmp_path):
    f = tmp_path / "misura con spazi.csv"
    f.write_text("a\n", encoding="utf-8")
    assert local_files.normalize_user_path(f'  "{f}"  ') == f
    assert local_files.normalize_user_path(f"'{f}'") == f
    assert local_files.normalize_user_path("") is None
    assert local_files.normalize_user_path(str(tmp_path)) is None
    assert local_files.normalize_user_path(str(tmp_path / "manca.csv")) is None


def test_pick_csv_file_not_windows(monkeypatch):
    monkeypatch.setattr(local_files.sys, "platform", "linux")
    assert local_files.pick_csv_file() is None


def test_pick_csv_file_runs_dialog_in_thread(monkeypatch):
    monkeypatch.setattr(local_files.sys, "platform", "win32")
    seen = {}

    def fake_dialog(initial_dir, title):
        seen["thread"] = threading.current_thread().name
        seen["initial"] = initial_dir
        return r"C:\dati\x.csv"

    monkeypatch.setattr(local_files, "_win32_open_dialog", fake_dialog)
    assert local_files.pick_csv_file(r"C:\dati") == r"C:\dati\x.csv"
    assert seen == {"thread": "csv-file-dialog", "initial": r"C:\dati"}
