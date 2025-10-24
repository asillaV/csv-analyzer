# Patch: Disabilita "Carica sample" con upload attivo

## Contesto
Durante il debug dell'Issue #46 è emerso che la coesistenza di un CSV già caricato (tramite `st.file_uploader`) e del pulsante **"Carica sample"** continuava a generare comportamenti inattesi. In particolare, l'utente poteva avviare il caricamento del sample mentre nello `st.session_state` erano ancora presenti i riferimenti all'upload precedente, riaprendo il bug originale (grafico/anteprima che restano agganciati al CSV vecchio).

## Problema
- Il bottone "Carica sample" risultava sempre abilitato.
- Il suo tooltip indicava ancora l'helper generico (“Carica un dataset demo…”), anche quando doveva essere bloccato.

In queste condizioni il form permetteva all’utente di richiedere il sample mentre l’upload era ancora valido; la logica downstream cercava di forzare il sample, ma la presenza dei dati dell’upload continuava a innescare race condition e cache hit indesiderate.

## Soluzione
Nel blocco Streamlit che gestisce uploader e pulsante (`web_app.py:721-735`):

1. **Disabilitazione dinamica** – il flag `disabled` ora vale `True` quando esiste un file in `upload` oppure quando il sample non è disponibile (`sample_available == False`).  
   ```python
   sample_disabled = not sample_available or upload is not None
   sample_clicked = st.button(..., disabled=sample_disabled, ...)
   ```

2. **Helper contestuale** – il parametro `help` mostra un messaggio diverso a seconda dello stato:
   - Con upload presente: “Devi eliminare il CSV in memoria prima di caricare il sample.”
   - In caso contrario: helper originale (“Carica un dataset demo multi-canale…”).

Questo espediente (“artefatto”) impedisce all’utente di riattivare il bug e chiarisce immediatamente la ragione del blocco, senza stravolgere la logica di session state già consolidata in `main`.

## Note
- Il blocco di invalidazione cache immediatamente successivo al commento `# FIX #46` non è stato toccato: resta necessario per liberare i dati dell’upload precedente prima del rerun del sample.
- Nessun altro flusso (upload → upload, sample → plot) viene impattato: il bottone torna automaticamente cliccabile appena l’uploader è vuoto.

## Verifica
- Upload CSV → “Carica sample” appare disabilitato + tooltip dedicato ✅
- Cancella l’upload (o ricarica app) → pulsante torna abilitato con helper originale ✅
- Con sample disponibile e nessun upload, il comportamento rimane invariato rispetto a `main` ✅
