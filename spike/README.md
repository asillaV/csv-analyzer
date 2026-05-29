# Spike — Reflex vs NiceGUI (passo 1 del piano di migrazione)

Mini-prototipo di **una schermata** (upload CSV → selezione colonna Y → filtro →
FFT, con grafico tempo + spettro) implementata in **entrambi** i framework Python
full-stack candidati, per scegliere con dati reali e non sulla carta.

Obiettivo primario dello spike: **de-rischiare l'assunzione critica** della
valutazione, cioè che il dominio in `core/` giri fuori da Streamlit, in un server
multiutente, senza `webbrowser.open` e senza `st.*`. ✅ Confermato.

## Cosa contiene

```
spike/
├── shared.py            # Driver headless framework-agnostico: importa core/, produce 2 figure Plotly
├── nicegui_app.py       # UI A — NiceGUI (processo singolo, pure-Python)
├── reflex_app/          # UI B — Reflex (State reattivo + frontend Next.js compilato)
│   ├── rxconfig.py
│   └── reflex_app/reflex_app.py
└── README.md            # questo file
```

`shared.py` è il punto chiave: **entrambe** le UI lo importano, quindi la stessa
"ricetta" di orchestrazione del core (load → resolve_fs → apply_filter →
compute_fft) è riusata identica. È il pattern che salveremmo da `web_app.py`.

## Come eseguire

Le dipendenze stanno in un venv isolato con accesso ai site-packages di sistema
(eredita pandas/numpy/scipy/plotly già presenti):

```bash
python -m venv --system-site-packages .venv-spike
.venv-spike/bin/pip install nicegui reflex

# Smoke test headless (nessuna UI): prova che core/ gira da solo
.venv-spike/bin/python spike/shared.py

# UI A — NiceGUI  → http://localhost:8080
.venv-spike/bin/python spike/nicegui_app.py

# UI B — Reflex   → frontend http://localhost:3000  (backend :8000)
cd spike/reflex_app && ../../.venv-spike/bin/reflex run
```

## Esito verificato in questo spike

| Verifica | NiceGUI | Reflex |
|---|---|---|
| Importa e riusa `core/` via `shared.py` | ✅ | ✅ |
| Core gira headless (no `webbrowser.open`, no `st.*`) | ✅ (smoke test: fs=100 Hz da indice, FFT 201 punti) | ✅ (stesso `shared.py`) |
| App risponde HTTP 200 | ✅ `:8080` | ✅ frontend `:3000` + backend `:8000` |
| Plotly nativo (serie + FFT) | ✅ `ui.plotly` | ✅ `rx.plotly` |
| Installazione | `pip install nicegui` | `pip install reflex` |
| Toolchain a runtime | **solo Python**, 1 processo | **compila Next.js**, 2 processi (Python + Node) |
| Artefatti di build generati | **0** | **~453 MB** (`.web/`: Next.js + node_modules) |
| Attriti incontrati | nessuno, ha funzionato al primo colpo | setter `set_<var>` non più auto-generati (cambio API), più warning di deprecazione (Radix, Sitemap) |

## Confronto e verdetto

**Modello di stato / concorrenza** (l'asse "Debole" della valutazione):
- **NiceGUI**: la funzione `@ui.page("/")` viene eseguita per-connessione → le
  variabili locali sono isolate per utente. Aggiornamenti imperativi
  (`plot.update_figure`). Semplice, adatto a una schermata.
- **Reflex**: `State` esplicito e tipizzato con var reattive ed event handler. È
  un modello **più strutturato**, che regge meglio la crescita dello stato — e
  questa app ne ha molto (form complessi, pipeline di trasformazioni, cache).

**Costo operativo / toolchain**:
- **NiceGUI**: un solo linguaggio, un solo processo, nessuno step di build. Deploy
  banale (un container Python).
- **Reflex**: sotto il cofano è **React/Next.js**. Il primo `reflex run` scarica
  un toolchain JS (bun) e compila ~450 MB di frontend; in produzione girano due
  processi. Più potente ma più pesante da gestire, e con API ancora in movimento
  (in questo spike abbiamo dovuto adattare il codice a un cambio di versione).

### Raccomandazione (aggiornata dai dati dello spike)

Il piano iniziale indicava **Reflex** come candidato primario per il modello di
stato. Lo spike **corregge** questa ipotesi alla luce dei tuoi vincoli reali
(**MVP / pochi clienti**, **team solo-Python**, time-to-market):

> **Per l'MVP: NiceGUI.** Massimizza il vantaggio chiave dell'opzione B (un solo
> linguaggio) senza reintrodurre di fatto un frontend React (che è ciò che Reflex
> compila). Deploy e manutenzione minimi, zero toolchain JS, ha funzionato senza
> attriti. È la via più veloce e meno rischiosa al mercato.

> **Reflex resta il candidato per dopo**, *se e quando* la complessità dello stato
> dell'app (pipeline, molti widget interdipendenti) ripaga il modello reattivo
> strutturato e il costo della toolchain Next.js. Dato che `shared.py`/`core/`
> sono framework-agnostici, passare a Reflex più avanti **non comporta riscrivere
> il dominio** — solo lo strato UI.

Nota onesta: l'ironia di Reflex è che "full-stack Python" nasconde un frontend
React. Se un giorno vorrai davvero React (opzione A), Reflex non ti fa risparmiare
quel costo — al massimo lo posticipa. Per un MVP Python-only, NiceGUI è la scelta
coerente con gli obiettivi dichiarati.

## Stato

Spike completo e verificato. Nessuna modifica al codice esistente (`core/`,
`web_app.py`, test intatti). Prossimo passo del piano: estrazione del package
`csv_core` (passo 2) — da avviare su tua conferma.
