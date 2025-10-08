Analizzatore CSV — Plot, Filtri, FFT

Interfacce: Web (Streamlit) · Desktop classica (Tkinter) · TUI (Textual)

Strumento per analizzare file CSV, plottare segnali, applicare filtri MA/Butterworth e calcolare FFT.
La frequenza di campionamento (fs) è unificata e condivisa tra filtraggio e FFT tramite una singola funzione centrale.

Caratteristiche principali

Auto-rilevamento CSV: encoding, delimitatore, header, colonne.

Gestione fs unificata: manuale o stimata in modo coerente per filtro e FFT.

Filtri:

Media mobile (MA) — non richiede fs.

Butterworth (LP/HP/BP) — con convalida Nyquist; richiede SciPy.

FFT: con detrend opzionale.

UI:

Web (Streamlit) con pannello Advanced.

Desktop classica (Tkinter): X-slice (X min/X max), modalità plot separati/sovrapposti, overlay originale.

TUI (Textual): selezione Y con checkbox, plot HTML.

Output: grafici Plotly HTML in outputs/, report statistiche via ReportManager.

Requisiti

Python 3.10+

Dipendenze: pip install -r requirements.txt

SciPy (opzionale) per Butterworth:

python -m pip install scipy


Se SciPy manca: Butterworth viene disabilitato con avvisi; MA e FFT restano utilizzabili.

Installazione
# 1) (consigliato) Ambiente virtuale
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2) Dipendenze
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# (opzionale, per Butterworth)
python -m pip install scipy

Avvio rapido

Web (Streamlit)

streamlit run web_app.py


oppure doppio click su Start(Web_app).bat.

Desktop classica (Tkinter)

python desktop_app_tk.py


(Se non presente, crea desktop_app_tk.py dalla versione fornita e tienilo nella root del progetto.)

TUI (Textual)

python main.py


Log file: logs/analizzatore_YYYYMMDD.log
Grafici: outputs/*.html (aperti automaticamente nel browser)

Struttura del progetto
core/
  analyzer.py        # autodetect CSV: encoding, delimiter, header, colonne
  loader.py          # carica DataFrame in base ai metadati rilevati
  signal_tools.py    # FilterSpec, FFTSpec, resolve_fs, validate_filter_spec,
                     # apply_filter (MA/Butterworth), compute_fft
  report_manager.py  # statistiche descrittive + export CSV/MD/HTML
  logger.py          # logging centralizzato

ui/
  main_app.py        # TUI (Textual): checkbox Y, fs avanzato, plot HTML

web_app.py           # Streamlit UI (Advanced con fs manuale)
desktop_app_tk.py    # Tkinter UI (slice X, sovrapposti/separati, overlay)
main.py              # entrypoint TUI
requirements.txt
README.md

Frequenza di campionamento (fs) — sorgente unica

Funzione centrale in core/signal_tools.py:

def resolve_fs(x_values, manual_fs: float | None) -> tuple[float | None, str]:
    """
    Restituisce (fs, source):
      - se manual_fs > 0  -> (manual_fs, "manual")
      - altrimenti prova a stimare da x_values -> ("estimated") se >0
      - altrimenti -> (None, "none")
    """


Regole:

manual_fs > 0 ha priorità su tutto.

Se non fornita/valida, si stima da x_values (numeric/datetime) usando l’intervallo medio tra campioni (Nyquist).

Se fs non è disponibile, Butterworth e FFT vengono saltati con messaggi chiari; MA rimane disponibile.

UI:

Web: pannello Advanced mostra fs [Hz]: <valore> (manual/estimated).

Tkinter: campo fs [Hz] (0=auto), log con fonte (manuale/stimata).

TUI: notifica fs: <valore> (manuale/stimata).

Convalida filtri (pre-flight)

Funzione in core/signal_tools.py:

def validate_filter_spec(spec: FilterSpec, fs: float | None) -> tuple[bool, str]:
    """
    True/False + messaggio.
    Regole:
      - spec.enabled == False -> OK.
      - kind == 'ma' -> OK (fs non richiesto).
      - Butterworth:
         * fs > 0 obbligatoria.
         * order >= 1.
         * cutoff presenti e > 0.
         * LP/HP: 0 < fc < fs/2.
         * BP: 0 < flo < fhi < fs/2 (fhi > flo).
    """


Se invalidi, i filtri vengono saltati con warning, senza crash.

Streamlit: st.warning(...).
TUI/Tkinter: messaggi nella status bar/log.

Filtri e FFT

apply_filter(y, x_values, spec, fs_override=None)

Se fs_override è passato dall’UI (da resolve_fs), non si ri-stima dentro la funzione.

Compatibile: se fs_override=None, vale il comportamento precedente.

compute_fft(y, fs, detrend)

Richiede fs > 0.

Serie troppo corte (< 4 campioni): ritorna array vuoti; l’UI mostra un messaggio informativo, non crasha.

Slice X, modalità di plot, overlay

Slice X (Tkinter):

X min / X max su X numerica o datetime;

se nessuna X è scelta, lo slice agisce sulle posizioni di indice (interi).

Lo slice si applica a plot, filtro e FFT per coerenza.

Modalità di plot (Tkinter):

separati: un HTML per ciascuna Y.

sovrapposti: tutte le Y in un unico grafico.

Overlay originale: se filtro attivo, aggiunge la curva originale (linea tratteggiata) oltre al filtrato.

Uso delle interfacce
Web (Streamlit) — web_app.py

Carica il CSV.

Seleziona X/Y.

Advanced: imposta fs (0=auto), filtri e FFT.

Click Plot/Report.
Nota: se fs invalido, comparirà un warning e Butterworth/FFT verranno saltati.

Desktop classica (Tkinter) — desktop_app_tk.py

Analizza il CSV → populate X e lista Y.

Seleziona X (opzionale), slice X, fs, Y (multi-selezione), filtri, FFT, modalità plot e overlay.

Plot: salva/apre HTML in outputs/.

Report CSV: statistiche con ReportManager.

TUI (Textual) — ui/main_app.py

Carica CSV, scegli X, spunta Y con checkbox, imposta fs/filtri/FFT, Plot.

I grafici si aprono nel browser come HTML.

Output e log

Grafici: outputs/*.html (nomenclatura sanificata; path assoluti per apertura browser, niente errori as_uri).

Report: generati da ReportManager (CSV/MD/HTML, a seconda della UI).

Log: logs/analizzatore_YYYYMMDD.log (rotazione per data; messaggi di convalida/warning/errore).

Troubleshooting → Frequenza di campionamento

fs=0 o vuoto → auto-stima da X (numeric/datetime).

X irregolare o non interpretabile → fornisci fs manuale (>0).

Cutoff ≥ Nyquist (fs/2) → correggi i valori (riduci cutoff o aumenta fs).

SciPy assente → Butterworth disabilitato (usa MA o installa SciPy).

Serie corta (N<4) → FFT non calcolata (messaggio informativo).

Test veloci (manuali)

fs:

X numerica, fs=0 → in log: fs (stimata); Butterworth/FFT attivi.

fs manuale > 0 → in log: fs (manuale); usato identico da filtro e FFT.

Filtri:

MA con ma_window=1 → identico all’originale.

Butter LP con fc >= fs/2 → warning, filtro non applicato.

FFT:

Serie con pochi campioni → messaggio FFT non calcolabile (no crash).

Slice X (Tkinter):

Datetime: inserisci 2025-01-01 → taglio corretto.

Nessuna X: X min=1000, X max=5000 → usa indici posizionali.

Output:

Controlla outputs/*.html si aprono nel browser.

Verifica log in logs/*.log.

(Se desideri, si possono aggiungere test pytest per resolve_fs, validate_filter_spec, compute_fft, MA.)

Novità principali (unificazione fs)

Nuova resolve_fs(...) → unica fonte di fs per filtro e FFT.

Nuova validate_filter_spec(...) → controlli preventivi (Nyquist, ordini, cutoff).

UI allineate: warning non bloccanti, nessun traceback.

Tkinter: slice X, plot sovrapposti, overlay originale.

Note

Nessuna dipendenza pesante aggiuntiva: Tkinter è incluso in Python (Windows).

Textual usa un suo “CSS”: es. border: solid #444; (niente 1px).

I nomi dei file HTML sono sanificati per evitare problemi con caratteri speciali.