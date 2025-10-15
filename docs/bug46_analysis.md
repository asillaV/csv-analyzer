# Bug #46 - Analisi Dettagliata del Problema di Priorità Upload/Sample

## Metadata
- **Issue**: #46
- **Milestone**: v1.0 Pro
- **Data analisi**: 2025-10-15
- **Status**: BLOCCATO - Problema architetturale fondamentale

---

## Descrizione del Problema

L'interazione tra il widget `file_uploader` di Streamlit e la gestione di `sample_bytes` in session_state presenta un **conflitto architetturale irrisolvibile** con l'approccio corrente.

### Comportamento Atteso

1. **Priorità Sample**: Quando l'utente clicca "Carica sample", il sample deve avere priorità assoluta sull'upload widget
2. **Aggiornamento Upload**: Quando l'utente carica un nuovo CSV, il nuovo file deve sostituire il sample e aggiornare i dati visualizzati
3. **Persistenza Corretta**: Non devono esserci ricaricamenti indesiderati quando si clicca "Applica/Plot"

### Comportamento Attuale (Problematico)

Il sistema presenta un **loop infinito di problemi**:

#### Tentativo 1: Sample con Priorità Assoluta
```python
if sample_bytes is not None:
    current_file = SimpleNamespace(name=sample_name, size=len(sample_bytes))
elif upload is not None:
    st.session_state.pop("_sample_bytes", None)
    current_file = upload
```

**Problema**: Quando upload widget contiene un file e sample_bytes esiste, il sistema usa sempre il sample anche se l'utente ha caricato un nuovo CSV.

**Sintomo**: Ricaricando un CSV dopo aver caricato sample, i dataset non si aggiornano.

#### Tentativo 2: Rilevamento Nuovo Upload
```python
if upload is not None and sample_bytes is not None:
    upload_sig = (upload.name, upload.size)
    sample_sig = (sample_name, len(sample_bytes))
    if upload_sig != sample_sig:
        st.session_state.pop("_sample_bytes", None)
        sample_bytes = None
```

**Problema**: Quando si clicca "Applica/Plot", Streamlit **ri-esegue tutto il codice** e il widget upload contiene ancora il file. Il controllo `upload is not None` è sempre True, quindi pulisce sample_bytes ogni volta.

**Sintomo**: Se premi "Carica sample" e poi "Applica/Plot", il sistema ricarica il CSV dell'upload widget invece di usare il sample.

---

## Causa Radice: Architettura di Streamlit

### Il Problema Fondamentale

Streamlit ha un modello di esecuzione **stateless** dove:
1. Ogni interazione (click button, checkbox, etc.) ri-esegue **tutto** lo script dall'inizio
2. Il widget `file_uploader` **mantiene il file** in memoria fino a quando l'utente non lo rimuove manualmente
3. Non c'è modo di distinguere tra:
   - "Upload widget contiene un file perché l'utente lo ha appena caricato"
   - "Upload widget contiene un file da una precedente interazione"

### Scenario Problematico

```
Step 1: User carica file.csv
  → upload widget popolato
  → current_file = upload

Step 2: User clicca "Carica sample"
  → sample_bytes salvato in session_state
  → upload widget ANCORA popolato (Streamlit non lo svuota)
  → Tentativo 1: current_file = sample ✓
  → Tentativo 2: rileva upload != sample, pulisce sample_bytes ✗

Step 3: User clicca "Applica/Plot"
  → Script ri-eseguito
  → upload widget ANCORA popolato
  → sample_bytes presente in session_state
  → Tentativo 1: usa sample, ignora nuovo upload ✗
  → Tentativo 2: rileva upload != sample, pulisce sample_bytes ✗
```

### Il Loop Impossibile

Non possiamo:
- ✗ Dare priorità assoluta a sample → upload nuovo file non funziona
- ✗ Rilevare "nuovo" upload → ogni click pulisce sample anche se non è nuovo
- ✗ Pulire upload widget programmaticamente → non supportato da Streamlit API
- ✗ Usare `on_change` callback → viene chiamato a ogni rerun, non solo al cambio reale

---

## Soluzioni Tentate (Tutte Fallite)

### 1. Flag `using_sample` basato su priorità
**Codice**: `using_sample = sample_bytes is not None`
**Problema**: Upload nuovo file non aggiorna dataset

### 2. Confronto file signature
**Codice**: `if upload_sig != sample_sig: clear sample`
**Problema**: Pulisce sample anche quando non dovrebbe (click Applica/Plot)

### 3. Cache invalidation esplicita
**Codice**: Pulizia cache quando si clicca "Carica sample"
**Problema**: Non risolve il conflitto upload/sample, solo performance

### 4. Flag `_clear_file_uploader`
**Codice**: `st.session_state.pop("file_upload", None)`
**Problema**: Non svuota il widget, solo il session_state key

---

## Soluzioni Possibili (Architetturali)

### Opzione A: Rimuovere Feature "Carica Sample" da Upload Widget
Separare completamente le due funzionalità:
- Upload widget per CSV utente
- Sezione separata con button "Usa dataset demo" che carica sample senza interagire con upload

**Pro**: Elimina il conflitto
**Contro**: UX peggiore, più click necessari

### Opzione B: Forzare Upload Widget a None dopo Sample
Usare un placeholder nel widget upload e settarlo programmaticamente:
```python
if sample_bytes is not None:
    upload = None  # Forza widget a None
```

**Pro**: Logica più chiara
**Contro**: Non testato, potrebbe non funzionare con Streamlit

### Opzione C: Modal Dialog per Conferma Switch
Quando sample_bytes esiste e upload is not None, mostrare dialog:
"Vuoi sostituire il sample con il file uploadato?"

**Pro**: Esplicito, nessuna ambiguità
**Contro**: UX pesante, troppi click

### Opzione D: Sample come File Pre-caricato
Invece di button separato, avere dropdown con:
- "Carica tuo file"
- "Usa dataset demo"

**Pro**: Design più chiaro
**Contro**: Richiede refactoring significativo UI

---

## Raccomandazioni

### Soluzione Raccomandata: **Opzione A (Separazione UI)**

**Implementazione**:
1. Rimuovere button "Carica sample" dalla sezione upload
2. Creare expander separato "Dataset Demo" con:
   - Descrizione del sample disponibile
   - Button "Carica dataset demo"
   - Messaggio "Questo sostituirà qualsiasi file caricato"
3. Quando si clicca "Carica dataset demo":
   - Pulire upload widget key da session_state
   - Settare sample_bytes
   - Mostrare messaggio "Dataset demo caricato. Per usare un tuo file, caricalo sopra."

**Vantaggi**:
- Elimina completamente il conflitto
- UX più chiara (due sezioni separate)
- Codice più semplice
- Nessun comportamento ambiguo

**Codice esempio**:
```python
# Sezione 1: Upload utente
upload = st.file_uploader("Carica il tuo CSV", type=["csv"])

# Sezione 2: Dataset demo (separata)
with st.expander("Oppure usa un dataset demo"):
    st.info("Il dataset demo contiene segnali multi-canale per testare le funzionalità.")
    if st.button("Carica dataset demo"):
        st.session_state["_sample_bytes"] = SAMPLE_CSV_PATH.read_bytes()
        st.session_state.pop("file_upload", None)  # Clear upload
        st.rerun()

# Logica priorità: sample_bytes > upload > none
if st.session_state.get("_sample_bytes"):
    current_file = SimpleNamespace(...)
elif upload:
    current_file = upload
else:
    current_file = None
```

---

## Conclusioni

Il bug #46 **non è risolvibile** con l'architettura corrente del widget upload + button sample nella stessa sezione.

Il problema è fondamentale: **Streamlit non permette di distinguere tra "upload presente da prima" e "upload appena caricato"** durante un rerun.

**Azione necessaria**: Redesign UI con separazione netta tra upload utente e sample demo.

**Priorità**: ALTA - Il bug impatta l'usabilità core dell'applicazione.

---

## File Modificati Durante Analisi

- `web_app.py`: Linee 761-787 (logica priorità upload/sample)
- `tests/manual/issue46_repro.md`: Casi di test
- `tests/manual/issue46_qa.md`: QA checklist
- `tests/manual/issue46_fix_final.md`: Documentazione fix tentati
- `tests/test_web_app_session.py`: Test automatizzati (6/6 passing, ma non coprono il rerun behavior)

---

## Timeline Fix Tentati

1. **Fix v1**: Priorità assoluta sample → Upload nuovo file non aggiorna
2. **Fix v2**: Rilevamento file signature → Click Aplica/Plot pulisce sample
3. **Fix v3** (proposto ma non implementato): Redesign UI separato

**Tempo investito**: ~3 ore
**Risultato**: Problema irrisolvibile con architettura corrente
**Next step**: Proporre redesign UI al team
