# Analizzatore CSV

> Dashboard multipiattaforma per esplorare file CSV, filtrare segnali e calcolare FFT.

![Streamlit](https://img.shields.io/badge/UI-Streamlit-EA4C89?logo=streamlit&logoColor=white)
![Tkinter](https://img.shields.io/badge/UI-Tkinter-1C4D9B)
![Textual](https://img.shields.io/badge/UI-Textual-6C4DF5)

---

## 📌 In breve
- Auto-rilevamento di encoding, delimitatore, header e colonne dei CSV.
- Filtri MA e Butterworth (LP/HP/BP) con convalida Nyquist e gestione degli errori.
- FFT con detrend opzionale e messaggi informativi su serie troppo corte.
- Gestione centralizzata della frequenza di campionamento (fs) condivisa tra filtri e FFT.
- Interfacce Web (Streamlit), Desktop (Tkinter) e TUI (Textual) con output Plotly HTML.

## 🧭 Indice
1. [Interfacce disponibili](#interfacce-disponibili)
2. [Requisiti](#requisiti)
3. [Installazione](#installazione)
4. [Avvio rapido](#avvio-rapido)
5. [Output generati](#output-generati)
6. [Architettura del progetto](#architettura-del-progetto)
7. [Gestione della frequenza di campionamento](#gestione-della-frequenza-di-campionamento)
8. [Troubleshooting & test manuali](#troubleshooting--test-manuali)

---

## Interfacce disponibili

| Interfaccia | Descrizione | Avvio |
|-------------|-------------|-------|
| **Web (Streamlit)** | UI moderna con pannello *Advanced* per filtri, FFT e override di fs. | `streamlit run web_app.py` *(oppure esegui `Start(Web_app).bat` su Windows).* |
| **Desktop (Tkinter)** | Interfaccia classica con slice X, modalità di plot separati/sovrapposti e overlay del segnale originale. | `python desktop_app_tk.py` *(crea il file se non presente usando la versione fornita).* |
| **TUI (Textual)** | Interfaccia a terminale con selezione Y via checkbox e preview dei grafici HTML. | `python main.py` |

---

## Requisiti
- **Python 3.10+**
- Dipendenze Python elencate in [`requirements.txt`](requirements.txt)
- **SciPy** (opzionale) per abilitare il filtro Butterworth: `python -m pip install scipy`

Se SciPy non è installato, il filtro Butterworth viene disabilitato automaticamente; MA e FFT restano operativi.

---

## Installazione
```bash
# 1) (consigliato) Crea un ambiente virtuale
python -m venv .venv

# 2) Attiva l'ambiente
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3) Aggiorna pip e installa le dipendenze
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4) (opzionale) Abilita il filtro Butterworth
python -m pip install scipy
```

---

## Avvio rapido
### Web (Streamlit)
```bash
streamlit run web_app.py
```

### Desktop classica (Tkinter)
```bash
python desktop_app_tk.py
```

### TUI (Textual)
```bash
python main.py
```

Durante l'uso:
- Carica il CSV e seleziona colonne X/Y.
- Imposta la frequenza di campionamento (0 = stima automatica).
- Applica filtri, FFT e genera report/plot.

---

## Output generati
- **Grafici**: HTML Plotly salvati in `outputs/` e aperti automaticamente nel browser.
- **Report**: esportazioni CSV/Markdown/HTML gestite da `ReportManager`.
- **Log**: file `logs/analizzatore_YYYYMMDD.log` con messaggi di validazione, warning ed errori.

---

## Architettura del progetto
```text
core/
  analyzer.py        -> Auto-rileva metadati dei CSV
  loader.py          -> Carica i DataFrame in base ai metadati
  signal_tools.py    -> FilterSpec, FFTSpec, resolve_fs, validate_filter_spec,
                         apply_filter (MA/Butterworth), compute_fft
  report_manager.py  -> Statistiche descrittive + export CSV/MD/HTML
  logger.py          -> Logging centralizzato

ui/
  main_app.py        -> UI Textual (checkbox Y, fs avanzato, plot HTML)

web_app.py           -> UI Streamlit (pannello Advanced)
desktop_app_tk.py    -> UI Tkinter (slice X, overlay, modalità plot)
main.py              -> Entrypoint TUI
requirements.txt
README.md
```

---

## Gestione della frequenza di campionamento
La funzione condivisa `resolve_fs` in `core/signal_tools.py` è l'unica fonte di verità per `fs`:

```python
def resolve_fs(x_values, manual_fs: float | None) -> tuple[float | None, str]:
    """Restituisce (fs, source).
    - manual_fs > 0  -> (manual_fs, "manual")
    - altrimenti stima da x_values -> (fs, "estimated") se > 0
    - se non disponibile -> (None, "none")
    """
```

**Regole principali**
- Un valore manuale > 0 ha sempre priorità.
- Se non specificato, viene stimato dall'intervallo medio di `x_values` (numeric o datetime).
- Se `fs` non è disponibile, Butterworth e FFT vengono ignorati con messaggi chiari; il filtro MA rimane disponibile.
- `fs_override` proveniente dalle UI evita ri-stime non necessarie.

### FFT
- Richiede `fs > 0`.
- Serie con meno di 4 campioni restituiscono array vuoti e un avviso informativo.

---

## Troubleshooting & test manuali

| Scenario | Atteso |
|----------|--------|
| `fs = 0` o vuoto | Stima automatica da `x_values`; Butterworth/FFT restano attivi se la stima è valida. |
| `x_values` irregolare | Fornisci `fs` manuale > 0. |
| Cutoff ≥ Nyquist (`fs/2`) | Viene mostrato un warning e il filtro non viene applicato. |
| SciPy assente | Butterworth disabilitato; MA e FFT funzionano. |
| Serie `N < 4` | FFT non calcolata, viene mostrato un messaggio informativo. |
| Slice X (Tkinter) | Supporta valori numerici/datetime oppure indici posizionali se X non è selezionata. |
| Output HTML | I file in `outputs/` hanno nomi sanificati e si aprono senza errori `as_uri`. |
| Log | Controlla `logs/*.log` per verificare warning e parametri stimati. |

---

Buone analisi! 📊
