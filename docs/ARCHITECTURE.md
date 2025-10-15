# Architecture Overview

## High-Level Flow
1. **Input**: l'utente carica un CSV tramite una delle UI (`web_app.py`, `desktop_app_tk.py`, `main.py`).
2. **Loader**: `core/loader.py` sfrutta `CsvAnalyzer` per rilevare encoding/delimitatore, costruisce i `DataFrame` e aggiorna gli stati applicativi.
3. **Processing**: `core/signal_tools.py` applica filtri (MA, Butterworth), calcola FFT, gestisce la risoluzione di `fs` e normalizza i segnali.
4. **Quality & Reporting**: `core/quality.py` esegue controlli di coerenza, `core/report_manager.py` produce statistiche/CSV/Markdown, `core/visual_report_manager.py` genera grafici Plotly + immagini.
5. **Output**: i risultati vengono salvati in `outputs/` e i log centralizzati finiscono in `logs/`.

## Module Map
```text
core/
  analyzer.py            -> euristiche CSV (encoding, delimitatore, header, colonne)
  loader.py              -> orchestrazione caricamento + iniezione nel modello app
  csv_cleaner.py         -> sanificazione input e coercizioni numeriche
  signal_tools.py        -> specifiche filtri/FFT, validazioni e computazioni numpy/scipy
  downsampling.py        -> riduzione campioni per grafici e calcoli ad alta densità
  plot_manager.py        -> creazione figure Plotly + salvataggi HTML/PNG
  quality.py             -> controlli dati (monotonia X, gap, anomalie di sampling)
  report_manager.py      -> statistiche descrittive ed esportazioni (CSV/MD/HTML)
  visual_report_manager.py -> composizione report visivi, export PNG/PDF
  logger.py              -> configurazione logging condivisa
```

## UI Layers
- **Streamlit (`web_app.py`)**: entrypoint web con pannello avanzato per filtri/FFT e generazione report visivi.
- **Tkinter (`desktop_app_tk.py`, `ui/desktop_app.py`)**: UI nativa con gestione slice X e overlay.
- **Textual (`main.py`, `ui/main_app.py`)**: interfaccia TUI per ambienti CLI, preview grafici HTML e controllo stato.

Le UI condividono le stesse funzioni core e mantengono la frequenza di campionamento attraverso `resolve_fs`. Qualsiasi nuova interfaccia dovrebbe dipendere dagli stessi servizi (`core/`) per garantire comportamenti coerenti.

## Assets & Scripts
- `tests_csv/`: dataset di riferimento generati da [`scripts/csv_spawner.py`](../scripts/csv_spawner.py).
- `docs/`: documentazione tecnica, report, manual testing e guide agent.
- `patches/`: fix una tantum o migrazioni documentate per future consultazioni.

## Extensibility Notes
- Aggiungi nuovi filtri o trasformazioni in `core/signal_tools.py` mantenendo funzioni pure con parametri tipizzati.
- Per nuovi controlli qualità estendi `core/quality.py` e integra la segnalazione nelle UI tramite i manager correnti.
- Mantieni i percorsi output configurabili via `config.json` per evitare path hard-coded nelle UI.
