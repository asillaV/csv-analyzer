# Contributing Guide

## Workflow
- **Branching**: parti da `main` o dal topic branch più recente e crea un branch descrittivo (`feature/`, `fix/`, `chore/`). Per riorganizzazioni usa il formato `chore/<descrizione>`.
- **Commits**: messaggi brevi e all'imperativo (`Fix loader fallback`). Aggrega modifiche correlate; evita commit di build/output.
- **Pull Request**: descrivi contesto, soluzione e impatti. Collega issue o ticket, allega screenshot/GIF quando tocchi l'UI e riporta i comandi di verifica eseguiti (`pytest`, `streamlit run` manuale, ecc.).

## Ambiente di sviluppo
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```
- Mantieni `requirements.txt` come singola fonte delle dipendenze.
- Le app si avviano da root: `streamlit run web_app.py`, `python desktop_app_tk.py`, `python main.py`.
- Script di servizio e dataset generator sono in [`scripts/`](../scripts/) (`python scripts/csv_spawner.py` rigenera `tests_csv/`).

## Stile & Qualità
- Codice Python con PEP 8 (indentazione 4 spazi) e type hints. Aggiungi docstring brevi quando il comportamento non è evidente.
- Mantieni la logica di dominio in `core/`; le UI (`ui/`, `web_app.py`, `desktop_app_tk.py`, `main.py`) devono consumare API consolidate dei moduli core.
- Centralizza logging tramite `core/logger.py` e configura i percorsi output con `config.json`.
- Aggiorna/crea test in `tests/` seguendo il naming `test_<modulo>.py`. Riutilizza fixture comuni in `tests/fixtures/`.

## Test & Coverage
```bash
pytest
# o, per iterazioni veloci
pytest -m "not slow" --maxfail=1
```
- Pytest è configurato via [`pytest.ini`](../pytest.ini) con copertura minima 80% su `core/`.
- I report di coverage HTML finiscono in `htmlcov/`; i log runtime in `logs/`; mantieni la repo pulita prima del commit.

## Documentazione
- Aggiorna `README.md` per i cambiamenti visibili e documenti dedicati in `docs/` (architettura, manual tests, reports).
- Guide specifiche per agent sono in [`docs/agents/`](agents/); sincronizzale se cambi convenzioni o comandi ricorrenti.
- Registra modifiche notevoli nel [`docs/CHANGELOG.md`](CHANGELOG.md).
