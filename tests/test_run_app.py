"""Test dell'avvio del browser nel launcher dell'exe (run_app.py)."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import run_app

pytestmark = pytest.mark.unit


class _FakeKernel32:
    def __init__(self) -> None:
        self.calls = []

    def SetDllDirectoryW(self, value) -> int:  # noqa: N802 - nome dell'API Win32
        self.calls.append(value)
        return 1


@pytest.mark.skipif(sys.platform != "win32", reason="SetDllDirectoryW esiste solo su Windows")
def test_frozen_browser_launch_resets_and_restores_dll_directory(monkeypatch):
    import ctypes

    kernel32 = _FakeKernel32()
    monkeypatch.setattr(ctypes.windll, "kernel32", kernel32)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", r"C:\app\_internal", raising=False)
    opened = []

    def fake_open(url):
        opened.append((url, list(kernel32.calls)))
        raise RuntimeError("browser non disponibile")

    monkeypatch.setattr(run_app.webbrowser, "open", fake_open)
    with pytest.raises(RuntimeError):
        run_app._open_browser_clean("http://localhost:8501")
    # Al momento dell'apertura la ricerca DLL era già standard, poi viene ripristinata.
    assert opened == [("http://localhost:8501", [None])]
    assert kernel32.calls == [None, r"C:\app\_internal"]


def test_source_launch_does_not_touch_dll_directory(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    opened = []
    monkeypatch.setattr(run_app.webbrowser, "open", lambda url: opened.append(url))
    run_app._open_browser_clean("http://localhost:8501")
    assert opened == ["http://localhost:8501"]


def test_wait_until_ready(monkeypatch):
    answers = iter([OSError("non ancora"), SimpleNamespace(status=503), SimpleNamespace(status=200)])

    class _Response:
        def __init__(self, status):
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def fake_urlopen(url, timeout):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return _Response(answer.status)

    monkeypatch.setattr(run_app.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(run_app.time, "sleep", lambda _s: None)
    assert run_app._wait_until_ready("http://localhost:8501", timeout_s=5)


def test_wait_until_ready_times_out(monkeypatch):
    monkeypatch.setattr(run_app.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(run_app.time, "sleep", lambda _s: None)
    clock = iter([0.0, 0.5, 2.0, 3.0])
    monkeypatch.setattr(run_app.time, "monotonic", lambda: next(clock))
    assert not run_app._wait_until_ready("http://localhost:8501", timeout_s=1.0)


def test_browser_opened_only_when_ready(monkeypatch):
    opened = []
    monkeypatch.setattr(run_app, "_open_browser_clean", lambda url: opened.append(url))
    monkeypatch.setattr(run_app, "_wait_until_ready", lambda url: False)
    run_app._open_browser_when_ready()
    assert opened == []
    monkeypatch.setattr(run_app, "_wait_until_ready", lambda url: True)
    run_app._open_browser_when_ready()
    assert opened == ["http://localhost:8501"]
