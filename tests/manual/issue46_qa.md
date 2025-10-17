# Issue #46 - QA Manual Checklist

## Pre-requisiti
- [ ] App avviata: `streamlit run web_app.py`
- [ ] File test disponibili:
  - `tests_csv/01_basic.csv` (10 righe, colonne: time, value)
  - `tests_csv/05_multicolumn.csv` (colonne multiple)
  - `assets/sample_timeseries.csv` (100+ righe)

---

## Test Case 1: Upload → Sample → Verify Data Correct

### Steps
1. **Upload** `tests_csv/01_basic.csv`
2. Seleziona X = "time", Y = ["value"]
3. Click "Applica / Plot"
4. ✅ **Verifica**: Grafico mostra ~10 punti (dati di 01_basic.csv)
5. ✅ **Verifica**: Anteprima DataFrame mostra colonne "time", "value"
6. Click **"Carica sample"**
7. ⚠️ **Osserva**: Form si svuota (normale, by design per evitare colonne incompatibili)
8. ✅ **Verifica**: Messaggio "Sample 'sample_timeseries.csv' caricato" appare
9. Seleziona colonne del sample (verificare che siano DIVERSE da 01_basic)
10. Click "Applica / Plot"
11. ✅ **Verifica**: Grafico mostra **molti più punti** (100+ dal sample)
12. ✅ **Verifica**: Anteprima DataFrame mostra colonne del **sample** (non più time/value)
13. ✅ **Verifica**: Nessun errore in console Streamlit

**Criterio Pass**: Plot e DataFrame mostrano **solo dati del sample**, nessuna traccia di 01_basic.csv

---

## Test Case 2: Sample → Upload → Sample (Cycle Test)

### Steps
1. Click "Carica sample"
2. Seleziona colonne → Plot
3. ✅ **Verifica**: Plot mostra dati sample
4. **Upload** `tests_csv/05_multicolumn.csv`
5. Seleziona colonne → Plot
6. ✅ **Verifica**: Plot mostra dati di 05_multicolumn (non più sample)
7. ✅ **Verifica**: Anteprima DataFrame ha colonne di 05_multicolumn
8. Click "Carica sample" di nuovo
9. Seleziona colonne → Plot
10. ✅ **Verifica**: Plot mostra di nuovo dati sample (non più 05_multicolumn)

**Criterio Pass**: Ogni switch file → sample → file cicla correttamente senza residui

---

## Test Case 3: Form State Preservation (Regression Check)

### Steps
1. Upload `01_basic.csv`
2. Seleziona X = "time", Y = ["value"]
3. Advanced panel: abilita filtro MA, window=5
4. Click "Applica / Plot" → ✅ Plot con filtro applicato
5. Click "Carica sample"
6. ⚠️ **Verifica**: Form si resetta (X/Y svuotati)
7. ✅ **Verifica**: Parametri Advanced (filtro MA) **mantengono valori** (window=5 ancora impostato)
8. Seleziona colonne compatibili con sample
9. Click "Applica / Plot"
10. ✅ **Verifica**: Filtro MA applicato anche al sample (se colonne numeriche)

**Criterio Pass**: Solo selezione colonne X/Y si resetta; parametri Advanced preservati

---

## Test Case 4: Upload File A → Upload File B (No Sample)

### Steps
1. Upload `01_basic.csv` → Plot
2. Upload `05_multicolumn.csv` (sostituisce 01_basic) → Plot
3. ✅ **Verifica**: Plot mostra solo dati di 05_multicolumn
4. ✅ **Verifica**: Nessun errore/warning

**Criterio Pass**: Upload-to-upload funziona come sempre (bug non impattava questo scenario)

---

## Test Case 5: Cache Invalidation Stress Test

### Steps
1. Upload `tests_csv/08_big_signal.csv` (file grande, ~100k righe)
2. Seleziona colonne → Plot → **ATTENDERE caricamento cache**
3. ✅ **Verifica**: Messaggio cache "Analisi CSV..." mostrato una sola volta
4. Modifica limiti assi (X min/max) → "Applica / Plot"
5. ✅ **Verifica**: Plot aggiorna SENZA ricaricare CSV (cache hit)
6. Click "Carica sample"
7. ✅ **Verifica**: Spinner "Analisi CSV..." appare di nuovo (cache invalidata)
8. Seleziona colonne → Plot
9. ✅ **Verifica**: Grafico mostra dati sample (non più big_signal)
10. Upload `01_basic.csv` (file piccolo)
11. ✅ **Verifica**: Spinner "Analisi CSV..." appare (cache invalidata di nuovo)

**Criterio Pass**: Cache si invalida SOLO quando file cambia, non su modifiche parametri

---

## Test Case 6: Error Handling (Edge Cases)

### Scenario A: Sample Non Disponibile
1. Rinomina temporaneamente `assets/sample_timeseries.csv` → `sample_OLD.csv`
2. Riavvia app
3. ✅ **Verifica**: Pulsante "Carica sample" **disabilitato**
4. ✅ **Verifica**: Caption "Sample non disponibile" mostrato
5. Upload file normale → Plot
6. ✅ **Verifica**: Funzionalità normale preservata (app non crasha)
7. Ripristina `sample_timeseries.csv`

### Scenario B: File Upload Corrotto
1. Crea file `corrupt.csv` con contenuto non-CSV (es. immagine rinominata)
2. Upload `corrupt.csv`
3. ✅ **Verifica**: Messaggio errore user-friendly (non stack trace)
4. ✅ **Verifica**: App rimane responsive (non freeze)

**Criterio Pass**: Errori gestiti gracefully, nessun crash

---

## Performance Checks

### Metrica 1: Tempo Switch File
- **Upload → Sample**: < 2 secondi (file sample ~100k righe)
- **Sample → Upload**: < 1 secondo (file upload tipicamente più piccoli)

### Metrica 2: Memory Leak Check
1. Ciclo 10x: Upload → Sample → Upload → Sample
2. ✅ **Verifica**: RAM app stabile (nessun aumento lineare)
3. ✅ **Verifica**: Nessun accumulo infinito in `st.session_state` (controllare con debug sidebar)

---

## Debug Sidebar (Opzionale)

Aggiungere temporaneamente in `web_app.py` dopo linea 768:

```python
# DEBUG: Issue #46
with st.sidebar:
    st.markdown("### 🐛 Debug #46")
    st.code(f"""
upload widget: {"presente" if upload is not None else "assente"}
_sample_bytes: {"presente" if st.session_state.get('_sample_bytes') else "assente"}
current_file: {current_file.name if current_file else "None"}
_cached_file_sig: {st.session_state.get('_cached_file_sig', 'None')[:8] if st.session_state.get('_cached_file_sig') else 'None'}
_last_uploaded_file_id: {st.session_state.get('_last_uploaded_file_id')}
    """)
```

**Uso**: Attivare durante QA per verificare stato interno in ogni step

---

## Sign-Off

**Tester**: _______________
**Data**: _______________
**Browser**: _______________
**OS**: _______________

### Result Summary
- [ ] **PASS** - Tutti i test case passati senza regressioni
- [ ] **FAIL** - Issue trovati (dettagliare sotto)
- [ ] **BLOCKED** - Impossibile testare (specificare motivo)

### Issues Found (se FAIL)
_Descrivere issue, steps di riproduzione, screenshot se applicabile_

---

## Cleanup Post-QA
- [ ] Rimuovere debug sidebar se aggiunto
- [ ] Cancellare file `tmp_upload.csv` se rimasto
- [ ] Verificare `outputs/` non pieno di file test
