# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec per l'Analizzatore CSV (interfaccia web Streamlit).

Build:
    pyinstaller analizzatore_csv.spec --noconfirm

Produce una distribuzione --onedir in dist/AnalizzatoreCSV/.
Entry point: run_app.py (avvia Streamlit in-process e apre il browser).

Streamlit è notoriamente ostico da congelare: serve raccogliere i suoi file
dati e i metadata di distribuzione (controllati a runtime via importlib.metadata),
oltre ai submoduli importati dinamicamente.
"""

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

# --- Asset di sola lettura del progetto (risolti da core.paths.resource_path) ---
datas = [
    ("web_app.py", "."),
    ("config.json", "."),
    ("config_web.json", "."),
    ("assets", "assets"),
    (".streamlit", ".streamlit"),
    # Sorgente del pacchetto 'core' incluso come dato in _internal/core/.
    # Streamlit esegue _internal/web_app.py e inserisce _internal in sys.path,
    # quindi 'import core.*' si risolve dal filesystem anche se per qualche
    # motivo gli hidden import non venissero impacchettati nel PYZ.
    ("core", "core"),
    # NB: la cartella 'presets/' NON va inclusa: in modalità frozen i preset di
    # default vengono rigenerati in %APPDATA% da create_default_presets().
]

hiddenimports = []
binaries = []

# --- Streamlit: dati, binari, submoduli, metadata ---
_st_datas, _st_binaries, _st_hidden = collect_all("streamlit")
datas += _st_datas
binaries += _st_binaries
hiddenimports += _st_hidden

# Moduli del progetto importati (alcuni dinamicamente via config)
hiddenimports += collect_submodules("core")

# NB: scipy NON va raccolto con collect_submodules: tirerebbe dentro centinaia
# di moduli di test (scipy.*.tests.*) inutili, gonfiando il pacchetto. L'hook
# integrato di PyInstaller (hook-scipy.py) gestisce già scipy correttamente.
hiddenimports += [
    "plotly",
    "plotly.graph_objects",
    "plotly.io",
    "pandas",
    "numpy",
    "pyarrow",
    "psutil",  # importato lazy in core.settings.detect_hardware
]

# Metadata richiesti a runtime da Streamlit (e da chi usa importlib.metadata.version)
for _pkg in (
    "streamlit",
    "pandas",
    "numpy",
    "plotly",
    "scipy",
    "pyarrow",
    "altair",
    "packaging",
    "kaleido",
):
    try:
        datas += copy_metadata(_pkg)
    except Exception:
        # Pacchetto opzionale assente: non bloccare la build
        pass

# File dati extra di pacchetti che li caricano da disco
for _pkg in ("plotly", "altair"):
    try:
        datas += collect_data_files(_pkg)
    except Exception:
        pass


a = Analysis(
    ["run_app.py"],
    # Root del progetto nel search path: senza questo gli hidden import 'core.*'
    # raccolti da collect_submodules non vengono trovati e finiscono scartati.
    pathex=[SPECPATH],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Interfacce non-web: non servono al bundle Streamlit
        "textual",
        "rich",
        "tkinter",
        # Pesi di sviluppo/test
        "pytest",
        "IPython",
        "notebook",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AnalizzatoreCSV",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # console visibile: mostra l'URL locale di Streamlit
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/logo.png",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AnalizzatoreCSV",
)
