# Parte 1 — Motore core: caricamento veloce, cache persistente, sessione di analisi

Branch: `feature/core-engine-performance` (da `feature/desktop-streamlit`, che include `origin/main`).

## Obiettivo

1. Rendere il caricamento dei CSV reali da "minuti" a "secondi", con RAM sotto controllo.
2. Non rifare mai due volte il parsing dello stesso file (cache persistente su disco).
3. Separare la logica di analisi da Streamlit in un oggetto `AnalysisSession` in `core/`, base per CLI, server MCP e integrazione IA (parti successive).

Fuori scope per questa parte: riscrittura dell'interfaccia, migrazione di framework UI, zoom con ricampionamento dinamico, funzioni IA.

## Misure di partenza (28/10/2026, PC di sviluppo, `.venv` Python 3.12, pandas 2.3.3, pyarrow 21)

| File | Dimensione | Forma | Percorso attuale (`analyze_csv` + `loader_optimized.load_csv`) | Picco RAM |
|---|---|---|---|---|
| `tests_csv/bench_large_500k.csv` (UTF-8, `,`, numeri misti EU/US) | 170 MB | 500.000 × 30 | 23,8 s | 1,5 GB |
| Fetta del file reale `20251126_100925_iptpa.csv` | 300 MB | 357.352 × 132 | 44,7 s | 1,9 GB |
| File reale completo (stima lineare) | 2,9 GB | ~3,5 M × 132 | ~7–8 min | > 10 GB (non misurato) |

Prototipo pyarrow sulla stessa fetta da 300 MB: lettura + pulizia **1,5 s**, scrittura Parquet zstd 0,6 s (**11 MB** su disco), riapertura da Parquet a pandas **0,3 s**.

Caratteristiche del file reale da rispettare:
- UTF-16 LE con BOM; prima riga `sep=;` (hint di Excel), header alla riga 1, separatore `;`, decimale `.`;
- 132 colonne con unità nel nome (`ACC11 [m/s2]`), **nomi duplicati** (`Undeclared []` ×3);
- molte celle vuote (colonne di metadati valorizzate solo ogni N righe);
- token speciali MSVC troncati dalla stampa a 3 decimali: `1.#IO`, `-1.#IO`, `1.#QO`, `-1.#R`, `1.#J`… (in origine `1.#INF`, `1.#QNAN`, `1.#IND`).

## Colli di bottiglia individuati

1. **Parsing tutto-stringa.** `loader_optimized` legge con `dtype=str` e poi converte colonna per colonna in `csv_cleaner`: per 3,5 M × 132 sono ~460 M oggetti stringa Python. È la causa principale di tempo e RAM.
2. **Upload dal browser.** Anche nella versione desktop il file passa da `st.file_uploader`: viene caricato via HTTP, tenuto intero in RAM (`upload.getvalue()`), hashato con SHA-1 per intero, riscritto in un file temporaneo e solo poi letto. Per 3 GB sono almeno tre copie complete.
3. **Subprocess + pickle oltre 200 MB.** Il DataFrame risultante torna al processo principale serializzato su `mp.Queue` (una copia completa in più) e si paga lo `spawn` dell'interprete.
4. **Nessuna cache persistente.** La cache vive in `st.session_state`: si perde al riavvio dell'app o aprendo un'altra scheda.
5. Soglie di `loader_optimized` lette all'import (le impostazioni cambiate valgono solo dopo il riavvio).

## Architettura target

```
core/
  ingest.py          # NUOVO: lettura veloce (pyarrow) + fallback su csv_cleaner per colonna
  dataset_cache.py   # NUOVO: cache Parquet persistente in user_data_dir()/cache
  engine.py          # NUOVO: AnalysisSession (dataset, fs, trasformazioni, filtro, FFT, cache risultati)
  analyzer.py        # invariato: rilevamento encoding/separatore/header
  csv_cleaner.py     # invariato: usato solo per le colonne che il percorso veloce non converte
  loader*.py         # restano per Tk/TUI e come riferimento di parità, poi deprecati
web_app.py           # usa AnalysisSession; nessuna logica di caricamento propria
```

Contratto di uscita del caricamento (invariato rispetto a oggi, per non toccare i consumatori):
- colonne numeriche come `Float64` (nullable pandas); i token non numerici e i token MSVC diventano `NA`;
- colonne non numeriche restano stringhe (nessuna inferenza automatica di date in fase di lettura: oggi le date vengono gestite dopo);
- nomi duplicati rinominati come fa pandas (`nome`, `nome.1`, `nome.2`);
- `CleaningReport` e `meta` con gli stessi campi di oggi.

## Fasi

Ogni fase si chiude con test verdi (`pytest`, soglia 80%), `ruff check core/ tests/` pulito e un commit.

### F0 — Banco di prova (prerequisito)
- `scripts/bench_load.py`: misura tempo per fase, picco RAM e forma del risultato; confronta percorso vecchio e nuovo; output tabellare.
- File di riferimento: `bench_small_1k`, `bench_medium_50k`, `bench_large_500k`, una fetta da 300 MB del file reale e il file reale completo (solo manuale, non in CI).
- **Test di parità** (`tests/test_ingest_parity.py`): per ogni CSV in `tests_csv/` (escluso il file reale) il nuovo percorso deve produrre lo stesso DataFrame del vecchio (stesse colonne, stessi dtype, valori uguali con tolleranza 1e-12, stessi NA). È la rete di sicurezza di tutta la parte 1.

### F1 — Lettura veloce (`core/ingest.py`)
- `pyarrow.csv` con encoding, separatore e header da `analyze_csv`; righe precedenti all'header saltate (gestisce `sep=;`).
- Colonne inferite numeriche da arrow: pronte. Colonne stringa: rimozione dei token MSVC (`^-?1\.#`) e cast a float; se il decimale rilevato è `,`, cast con `decimal_point=","`; se resta qualcosa di non convertibile (migliaia, valute, simboli) si passa **solo quella colonna** a `csv_cleaner`, con il formato rilevato una volta sul campione.
- Inferenza delle date disattivata in lettura (`timestamp_parsers=[]`), per parità.
- Conversione a pandas con `types_mapper` verso `Float64`.
- `progress_callback` mantenuto (fasi: lettura, conversione, cache).
- Integrazione: `loader_optimized.load_csv` delega a `ingest` quando pyarrow è disponibile; altrimenti percorso attuale.

### F2 — Cache persistente (`core/dataset_cache.py`)
- Posizione: `user_data_dir()/cache/` (da sorgente: root progetto, già in `.gitignore` da aggiungere).
- Chiave: percorso assoluto + dimensione + mtime + hash veloce (primi e ultimi 4 MB) + versione del formato di ingest + opzioni di pulizia/override decimale. Per i file arrivati da upload: SHA-1 dei byte, come oggi.
- Contenuto: Parquet zstd del DataFrame + JSON con `meta` e `CleaningReport`.
- Limite di spazio configurabile (`performance.cache_max_gb`, default 5) con eliminazione LRU; pulsante "Svuota cache" nella scheda Impostazioni.
- Invalidazione automatica su cambio di chiave; versione del formato da incrementare a ogni modifica di `ingest`.

### F3 — Apertura file da percorso locale (solo modalità desktop)
- In modalità desktop (bundle PyInstaller o server su `localhost`) aggiungere "Apri file dal PC": il server è locale, quindi legge direttamente dal disco, senza upload, senza copia in RAM e senza file temporaneo.
- Selezione con finestra di dialogo nativa (tkinter lanciato lato server) e lista dei file recenti.
- `st.file_uploader` resta per la modalità web e per i file piccoli.

### F4 — Semplificazione dell'esecuzione
- Con il percorso veloce il parsing gira nel thread corrente: il ramo subprocess + pickle viene rimosso per i percorsi locali.
- `parse_timeout_s` diventa un limite controllato tra le fasi (la lettura arrow non è interrompibile a metà).
- `core/csv_parser_worker.py` resta solo finché esiste il ramo upload > 200 MB; da decidere se eliminarlo del tutto (vedi decisioni aperte).

### F5 — `AnalysisSession` (`core/engine.py`)
- Contiene: dataset (DataFrame, meta, report, firma), colonna X, `resolve_fs` calcolato una volta, pipeline di trasformazioni, `FilterSpec`, `FFTSpec`, controlli qualità.
- Sposta qui le cache di filtro, FFT, trasformazioni e qualità che oggi vivono in `st.session_state` (stesse chiavi, stessi limiti di dimensione).
- API Python pura, senza import di Streamlit: `open(path)`, `set_x(col)`, `set_transforms(...)`, `set_filter(...)`, `set_fft(...)`, `series(col)`, `fft(col)`, `quality()`, `stats()`.
- `to_recipe()` / `from_recipe()` in JSON: estensione dei preset attuali (trasformazioni + filtro + FFT + fs), base per CLI e MCP della parte 2.
- `web_app.py` la adotta per gradi: prima caricamento e cache, poi le funzioni di analisi. Nessuna modifica visibile all'utente.
- Test unitari dedicati, con copertura ≥ 90% sul nuovo modulo.

## Obiettivi misurabili (criteri di accettazione)

| Misura | Oggi | Obiettivo |
|---|---|---|
| Fetta reale 300 MB, prima apertura | 44,7 s | ≤ 3 s |
| File reale 2,9 GB, prima apertura (da percorso locale) | ~7–8 min | ≤ 45 s |
| File reale 2,9 GB, riapertura (cache) | uguale alla prima | ≤ 5 s |
| `bench_large_500k`, prima apertura | 23,8 s | ≤ 4 s |
| Picco RAM sul file reale | > 10 GB stimati | ≤ 2,5 × dimensione del DataFrame finale |
| Test di parità su `tests_csv/` | — | 100% verdi |

## Ordine di lavoro proposto

1. F0 (banco di prova + test di parità) — senza questo non si misura né si garantisce nulla.
2. F1 (lettura veloce) — è il guadagno principale.
3. F2 (cache persistente).
4. F3 + F4 (apertura da percorso, rimozione subprocess) — il guadagno sul file da 3 GB arriva soprattutto qui.
5. F5 (`AnalysisSession`) — può iniziare in parallelo a F2–F4 perché tocca file diversi.

## Rischi

- **Differenze di rilevamento formato**: arrow e `csv_cleaner` possono decidere diversamente su colonne ambigue (es. `1.234`). Mitigazione: formato deciso una sola volta da `suggest_number_format` e imposto ad arrow; test di parità.
- **`Float64` nullable** è più lento di `float64` NumPy nei calcoli: si mantiene per parità in questa parte, la valutazione del passaggio a `float64` è rimandata.
- **UTF-16**: arrow transcodifica in streaming; da verificare il picco RAM sul file completo.
- **Righe malformate**: oggi `on_bad_lines="skip"`; con arrow serve `invalid_row_handler` equivalente.
- **PyInstaller**: i nuovi moduli `core/*` devono comparire nel bundle (`collect_submodules("core")` li copre; verificare con una build).

## Decisioni (confermate il 28/10/2026)

1. Il file `tests_csv/20251126_100925_iptpa.csv` è rappresentativo; in più il loader deve aprire **la grande maggioranza dei formati CSV**, non solo quello. Per questo la parità si verifica su un corpus generato di formati diversi (`tests/fixtures/csv_corpus.py`), che gira anche in CI.
2. I token MSVC restano `NA`, come oggi.
3. Nella versione desktop il ramo subprocess si elimina.
4. Limite predefinito della cache su disco: 5 GB.

## Esito F0 (banco di prova)

- `tests/fixtures/csv_corpus.py`: 23 formati generati (UTF-8/BOM/UTF-16 LE e BE/cp1252, separatori `,` `;` tab `|`, decimale `,` e `.`, migliaia con `.` `,` apostrofo e spazio, `sep=;`, nomi duplicati, token MSVC, valute e percentuali, segni e parentesi, date, senza intestazione, campi quotati, separatore finale, righe malformate, colonne vuote, file larghi e lunghi).
- `tests/parity.py`: oracolo `load_reference()` (percorso di produzione attuale) e confronto `compare_outcomes()` secondo il contratto sopra.
- `tests/test_ingest_parity.py`: parità su tutto il corpus e sui CSV locali di `tests_csv/` (≤ 50 MB, marcati `slow`); saltati finché `core.ingest` non esiste.
- `scripts/bench_load.py`: tempi, picco RAM e memoria del DataFrame per ogni loader in processi separati, `--slice-mb` per le fette dei file grandi, `--compare` per le differenze.

Interfaccia attesa da F1: `core.ingest.load(path, *, apply_cleaning=True, decimal=None, thousands=None, progress_callback=None)` che restituisce un oggetto con `df`, `report` (`CleaningReport`), `meta` (dict come `analyze_csv`) e `timings` (dict fase → secondi).

Difetti del loader attuale trovati dal corpus (il nuovo **non** deve replicarli):

| Caso | Oggi | Atteso dal nuovo |
|---|---|---|
| CSV Windows cp1252 con accenti nell'intestazione | crash `UnicodeDecodeError` | caricato, intestazione corretta |
| CSV senza intestazione | la prima riga di dati diventa intestazione (riga persa) | tutte le righe, nomi `Column_N` come `analyze_csv` |
| Ramo chunked (oltre le soglie) | colonne di metadati del file reale lette come testo; righe malformate gestite diversamente dal ramo standard | stesso comportamento del ramo standard |

Per questo la parità si misura sempre contro il ramo **standard** del loader attuale.

Misure di riferimento con `scripts/bench_load.py` (processo isolato):

| File | Tempo | Picco RAM |
|---|---|---|
| `bench_medium_50k.csv` (11 MB) | 1,9 s | 222 MB |
| `bench_large_500k.csv` (170 MB) | 22,3 s | 1,7 GB |
| Fetta reale 300 MB | 40,4 s | 2,0 GB |
