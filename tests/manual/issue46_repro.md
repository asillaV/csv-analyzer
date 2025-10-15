# Issue #46 - Riproduzione Bug Session State

## Descrizione Bug
Dopo aver cliccato "Carica sample", il pulsante "Applica/Plot" usa ancora i dati del file precedentemente caricato via upload, e il form si azzera indebitamente.

## Root Cause
**File**: `web_app.py`, linee 751-760 (versione pre-fix)

**Problema**: Race condition nel session state management:
1. Click "Carica sample" → salva `_sample_bytes` in session_state → `st.rerun()`
2. Al rerun successivo:
   - Widget `upload` potrebbe essere ancora `!= None` per 1 ciclo
   - Codice vecchio fa: `if upload is not None: st.session_state.pop("_sample_bytes")`
   - ❌ **Cancella il sample appena caricato!**
3. `current_file` usa dati stale o None → cache non invalidata
4. Form si azzera ma plot mostra dati precedenti

**Fix Implementato** (linee 751-766, versione post-fix):
- **Priorità esplicita**: `sample_bytes` ha sempre priorità su `upload`
- Se `sample_bytes` esiste → usa sample, marca `_clear_file_uploader`
- Se solo `upload` esiste → usa upload, pulisci residui sample
- Eliminata race condition

## Steps di Riproduzione (Pre-Fix)

### Setup
1. Avvia app: `streamlit run web_app.py`
2. Prepara file CSV test: `tests_csv/01_basic.csv` (10 righe)
3. Verifica sample disponibile: `assets/sample_timeseries.csv` (100+ righe)

### Test Case 1: Upload → Sample → Plot (Bug Principale)

**Steps**:
1. **Upload** `01_basic.csv` (10 righe)
2. Seleziona X = "time", Y = ["value"]
3. Click "Applica / Plot" → ✅ Vedi grafico con 10 punti
4. Click **"Carica sample"**
5. ⚠️ **OSSERVA**: Form si svuota (colonne X/Y resettate)
6. Riseleziona X = "time", Y = ["value"] (dal sample, dovrebbero essere diverse colonne)
7. Click "Applica / Plot"

**Risultato Atteso** (dopo fix):
- Plot mostra dati del **sample** (100+ punti)
- Anteprima DataFrame mostra colonne del sample
- Nessun messaggio di errore

**Risultato Buggy** (pre-fix):
- ❌ Plot mostra ancora dati di `01_basic.csv` (10 punti)
- ❌ O errore "colonna non trovata" se nomi diversi
- ❌ Anteprima DataFrame potrebbe essere inconsistente

---

### Test Case 2: Sample → Upload → Sample (Ciclo Completo)

**Steps**:
1. Click "Carica sample" → Plot → ✅ OK
2. **Upload** `02_with_x_numeric.csv` → Seleziona colonne → Plot → ✅ OK
3. Click "Carica sample" di nuovo
4. Seleziona colonne → Plot

**Risultato Atteso**:
- Ogni click "Carica sample" invalida upload e mostra sample
- Ogni upload invalida sample e mostra file caricato
- Nessuna cross-contamination

**Risultato Buggy** (pre-fix):
- ❌ Step 3: sample non carica correttamente, mostra dati di step 2

---

### Test Case 3: Upload File A → Upload File B (No Sample Involved)

**Steps**:
1. Upload `01_basic.csv` → Plot
2. Upload `05_multicolumn.csv` → Plot

**Risultato Atteso & Osservato**:
- ✅ Sempre funzionato correttamente (bug specifico di sample)

---

## Verifica Fix

### Checklist Post-Fix
- [ ] Test Case 1: Upload → Sample → Plot mostra **solo dati sample**
- [ ] Test Case 2: Sample → Upload → Sample cicla correttamente senza residui
- [ ] Form non si azzera inaspettatamente
- [ ] Cache `_cached_df` invalidata correttamente al cambio file
- [ ] `_reset_generated_reports_marker()` chiamata con `current_file` corretto
- [ ] Nessun warning/errore in console Streamlit

### Script di Verifica Rapida (5 min)
```bash
# 1. Avvia app
streamlit run web_app.py

# 2. Ciclo rapido 3x: Upload → Sample → Upload
#    - Ogni passaggio deve mostrare dati corretti
#    - Form deve mantenere selezioni se colonne compatibili

# 3. Controlla logs
tail -f logs/analizzatore_*.log | grep -i "error\|warning\|cache"
```

---

## Analisi Session State (Debug)

### Variabili Chiave da Monitorare
```python
# In web_app.py, aggiungere temporaneamente per debug:
st.sidebar.code(f"""
upload: {upload is not None}
_sample_bytes: {st.session_state.get('_sample_bytes') is not None}
_cached_file_sig: {st.session_state.get('_cached_file_sig')}
_last_uploaded_file_id: {st.session_state.get('_last_uploaded_file_id')}
current_file.name: {current_file.name if current_file else None}
""")
```

### Stato Corretto (Post-Fix)
- **Dopo "Carica sample"**:
  - `_sample_bytes`: NOT None
  - `upload`: None (o viene ignorato)
  - `current_file.name`: `sample_timeseries.csv`
  - `_cached_file_sig`: aggiornato con hash del sample

- **Dopo Upload**:
  - `_sample_bytes`: None
  - `upload`: NOT None
  - `current_file.name`: nome file caricato
  - `_cached_file_sig`: aggiornato con hash dell'upload

---

## Regression Prevention

### Unit Test (TODO - Task successivo)
```python
# tests/test_session_state.py
def test_sample_load_clears_upload():
    """Issue #46: Sample deve avere priorità su upload widget stale."""
    # Simula: upload attivo + sample_bytes salvato
    # Assert: current_file usa sample, non upload

def test_upload_clears_sample():
    """Upload deve invalidare sample precedente."""
    # Simula: sample_bytes attivo + nuovo upload
    # Assert: current_file usa upload, sample_bytes == None
```

---

## Timeline Fix
- **Reported**: Issue #46 aperto (data non specificata)
- **Root Cause Identified**: [DATA FIX]
- **Fix Implementato**: [DATA FIX]
- **Verified**: [DATA QA]
- **Merged**: [DATA MERGE]
