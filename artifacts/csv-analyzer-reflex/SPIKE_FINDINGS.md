# Spike Reflex — Risultati & Stima Effort

**Data**: 2026-05-29 | **Reflex**: 0.9.3 | **Python**: 3.11 | **Node.js**: 24.13

---

## 1. Cosa è stato verificato

| Test | Risultato | Note |
|------|-----------|-------|
| Import `core/analyzer.py` in Reflex | ✅ OK | Zero modifiche al modulo |
| Import `core/signal_tools.py` (`resolve_fs`, `apply_filter`) | ✅ OK | Zero modifiche al modulo |
| Sintassi `csv_spike.py` (1 pagina completa) | ✅ OK | `py_compile` + import runtime |
| `CsvState` (sostituisce 208 `st.session_state`) | ✅ OK | Tutti i campi verificati a runtime |
| `rx.upload`, `rx.plotly`, `rx.select`, `rx.checkbox` | ✅ Presenti | Verificati con `dir(rx)` |
| `rx.callout`, `rx.divider`, `rx.badge`, `rx.cond` | ✅ Presenti | Verificati con `dir(rx)` |

---

## 2. Portabilità del codice

### core/ — portabilità 95% (zero riscrittura)

```
core/analyzer.py          → ✅ riutilizzabile as-is
core/signal_tools.py      → ✅ riutilizzabile as-is   (resolve_fs, apply_filter, FilterSpec)
core/csv_cleaner.py       → ✅ riutilizzabile as-is
core/loader*.py           → ✅ riutilizzabile as-is
core/quality.py           → ✅ riutilizzabile as-is
core/signal_transforms.py → ✅ riutilizzabile as-is
core/downsampling.py      → ✅ riutilizzabile as-is

core/preset_manager.py    → ⚠️  path globale hardcoded (presets/ dir)
                             Refactoring: iniettare `presets_dir` per-utente → DB o object storage
core/plot_manager.py      → ⚠️  webbrowser.open() e outputs/ globale
                             Refactoring: rimuovere webbrowser, iniettare output_dir o ritornare bytes
core/report_manager.py    → ⚠️  outputs/ globale, kaleido
                             Refactoring: object storage per-utente
core/logger.py            → ⚠️  log file globale per giorno
                             Refactoring: logging strutturato per-tenant
```

### web_app.py — riscrittura ~85%

La "ricetta" (sequenza di chiamate a `core/`) è salvabile; l'impalcatura Streamlit è da buttare.

| Pattern Streamlit | Equivalente Reflex | Difficoltà |
|---|---|---|
| `st.session_state` (208 ref) | Campi tipizzati in `rx.State` | ⭐ Semplice — più pulito |
| `st.rerun()` (7 call) | Non esiste — non serve | ⭐ Più semplice |
| `st.form(f"controls_{nonce}")` | Event handler diretto, nessun nonce | ⭐ Più semplice |
| `st.file_uploader` | `rx.upload` + `async handle_upload` | ⭐⭐ Media |
| `st.plotly_chart` | `rx.plotly(data=State.figure)` | ⭐ Semplice |
| `st.sidebar` | `rx.box` CSS width fissa in `rx.hstack` | ⭐ Semplice |
| `st.selectbox`, `st.checkbox`, `st.number_input` | `rx.select`, `rx.checkbox`, `rx.input` | ⭐ Semplice |
| `st.expander` | `rx.accordion` o `rx.cond` | ⭐ Semplice |
| `@st.cache_data` / cache manuale FIFO in session_state | `@rx.cached_var` + cache lato server | ⭐⭐ Media |
| Preset load/save (JSON su filesystem globale) | DB per-utente + object storage | ⭐⭐⭐ Alta (richiede infra) |
| Report/export (HTML, immagini su outputs/) | Object storage + signed URL | ⭐⭐⭐ Alta (richiede infra) |
| Auth / multi-tenant | 0 attuale → da costruire ex-novo | ⭐⭐⭐ Alta |

---

## 3. Sfide di deployment su Replit

### Problema principale: architettura a due porte

Reflex lancia **due servizi**:
- **Backend FastAPI** (WebSocket + REST) — porta N
- **Frontend Next.js** — porta N+1

Il proxy di Replit espone **una porta per artifact**. In sviluppo locale questo è un problema.

**Soluzioni**:

| Opzione | Pro | Contro |
|---------|-----|--------|
| `reflex run --env prod` | Un solo processo, porta singola | Richiede build (~2-3 min primo avvio) |
| Nginx interno | Massima flessibilità | Aggiunge complessità all'infra |
| VPS esterno (Hetzner, Fly.io) | Architettura standard, nessun limite Replit | Esce da Replit |

**Raccomandazione**: per il deploy finale usare VPS o PaaS Python-friendly (Railway, Fly.io, Render).
Replit va bene per lo sviluppo, non per la produzione SaaS multiutente.

### Startup time

- Prima installazione: `reflex init` scarica Node deps (~30-60 sec)
- Build frontend: `reflex export` (~60-120 sec con Next.js)
- Riavvio a caldo (dopo la prima build): ~3-5 sec

---

## 4. DataFrame serialization in Reflex State

`rx.State` richiede che tutti i campi siano **JSON-serializzabili** (dict, list, str, int, float, bool).
Un `pd.DataFrame` non lo è — deve essere serializzato/deserializzato ad ogni accesso.

**Strategia adottata nello spike**:
```python
df_json: str = ""   # DataFrame.to_json(orient="split")
# accesso: pd.read_json(io.StringIO(self.df_json), orient="split")
```

**Overhead stimato**: ~1.5-2× rispetto ad accesso diretto (DataFrame ~5 MB → JSON ~12 MB).

**Strategia produzione**: salvare il CSV in object storage (S3/R2) o PostgreSQL binary,
tenerlo fuori dallo State Reflex. Lo State tiene solo metadati (nome file, colonne, num_rows).

---

## 5. Stima effort MVP (team solo-Python, ambizione ~pochi clienti)

### Fase A — Core funzionale (upload → plot → filtro → FFT)

| Attività | Settimane |
|----------|-----------|
| Pacchettizzare `core/` (pyproject.toml, parametrizzare I/O) | 0.5 |
| Porting UI principale in Reflex (upload, colonne, plot, filtro, FFT, trasformazioni) | 2 - 2.5 |
| Preset → DB SQLite/PostgreSQL per-utente | 1 |
| Report/export → object storage (Cloudflare R2 o S3) | 1 |
| **Totale Fase A** | **4.5 - 5 settimane** |

### Fase B — SaaS foundation

| Attività | Settimane |
|----------|-----------|
| Auth (Clerk o Auth0 via Reflex) | 1 |
| Multi-tenancy DB schema (user_id su preset/report) | 0.5 |
| Billing (Stripe) | 1.5 |
| Deploy infra (Railway o Fly.io) | 0.5 |
| **Totale Fase B** | **3.5 settimane** |

### Totale: **8 - 9 settimane** per MVP vendibile

> **Nota**: questa è la stima *minima* con sviluppo continuo e zero context-switch.
> Con tempi part-time o refactoring iterativo: 3-4 mesi reali.

---

## 6. Confronto stato/pattern: 20 righe Reflex vs Streamlit

```python
# ── STREAMLIT (da web_app.py) ────────────────────────────────────────
# 208 accessi sparsi in 2988 righe:
st.session_state["manual_fs"] = preset_manual_fs
st.session_state["_filter_cache"][key] = result
if "df" not in st.session_state:
    st.session_state["df"] = load_csv(...)
st.session_state["_plots_ready"] = True
st.rerun()  # x7 in tutta la codebase

# ── REFLEX (csv_spike.py) ───────────────────────────────────────────
# Tutti i campi dichiarati in un unico posto, tipizzati:
class CsvState(rx.State):
    manual_fs_input: str = "0"
    figure: dict = {}
    has_plot: bool = False
    ...
    def plot(self):           # evento esplicito, nessun rerun
        self.has_plot = True
        self.figure = fig.to_dict()
```

---

## 7. Conclusione

**Vale la pena?** Sì, con le seguenti condizioni:

1. **core/ è il valore** — migra senza toccarla (95% as-is)
2. **Reflex è adatto** al profilo (team Python, MVP, form complessi con stato)
3. **Il bottleneck non è la UI** — è auth + persistenza + infra (come previsto dal documento di valutazione)
4. **Reflex richiede ~9 settimane** per un MVP funzionale e vendibile; Streamlit ne richiederebbe 3-4 ma con tetto di scalabilità basso
5. **Deploy su Replit non è raccomandato** per la produzione SaaS; Railway o Fly.io sono più adatti

**Prossimi passi concreti (se si decide di procedere)**:
1. `pyproject.toml` per `core/` (1 giorno)
2. Spike auth: Reflex + Clerk (2 giorni)
3. Spike DB: Reflex + SQLAlchemy + PostgreSQL (1 giorno)
4. Decisione go/no-go basata sui 3 spike
