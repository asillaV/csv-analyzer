# PERF-SEC-002 – Hardening caricamento CSV

## Sintesi fix
- Introdotti limiti configurabili su dimensione file (`max_file_mb`), righe (`max_rows`) e colonne (`max_cols`).
- Parsing spostato in processo dedicato con timeout (`parse_timeout_s`) per prevenire blocchi prolungati.
- File temporanei ora generati con `tempfile` e cancellati al termine dell'analisi.
- Messaggi utente espliciti con blocco (`st.stop()`) quando i limiti vengono superati.

## Dettagli implementativi
- `config.json`: nuova sezione `limits` con i default (100 MB, 1M righe, 500 colonne, timeout 30s).
- `web_app.py`:
  - Loader centralizzato `_load_limits_config()` con `lru_cache`.
  - Validazione pre-parsing `_check_size_limit()` su upload e sample.
  - Worker `_parse_csv_with_timeout()` basato su `multiprocessing` (spawn) e coda per trasferire `df` e metadati.
  - Check post-parsing tramite `_check_dataframe_limits()` con invalidazione cache in caso di overflow.
  - Download del file originale servito da memoria (`file_bytes`), niente più `tmp_upload.csv` persistente.
- `core/loader.py` e `core/analyzer.py` riusati dal worker, nessuna modifica diretta richiesta.

## Test e verifica
- Nuovo test `tests/test_web_app_session.py::test_reject_large_file` per verificare il messaggio di blocco su file oltre soglia.
- Nuovo test `tests/test_web_app_session.py::test_parse_cleanup_on_exception` per assicurare la rimozione del temporaneo anche in caso di errore nel parsing.
- Esecuzioni `pytest --no-cov tests/test_web_app_session.py::test_reject_large_file` e `pytest --no-cov -q tests/test_web_app_session.py::test_parse_cleanup_on_exception` con esito positivo.

## TODO follow-up
- Valutare ulteriori test di integrazione per limiti su righe/colonne.
- Considerare logging dedicato per gli eventi di rifiuto (attualmente passa tramite `st.error`).
- Documentare per il team DevOps eventuali modifiche ai valori in `config.json` per installazioni self-hosted.
