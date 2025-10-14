# Test Plan: Reset Button Functionality (Issue #44)

## Bug Description
Il tasto "Reset impostazioni" non resettava completamente lo stato dell'applicazione, causando comportamenti inconsistenti tra sessioni.

## Root Cause
La funzione `_reset_all_settings()` non ripuliva alcuni stati critici:
- `_visual_report_prev_selection` - Tracciamento selezione report visivo
- `_visual_report_last_default_x_label` - Label X di default per report
- `_quality_file_sig` - Signature file per modalità qualità

Questi stati "fantasma" causavano:
1. Report visivi con etichette di sessioni precedenti
2. Modalità qualità (Prestazioni/Alta fedeltà) non resettata al default
3. Widget dei report visivi che mantenevano valori obsoleti

## Fix Implemented
Aggiunto reset di 3 stati mancanti in `_reset_all_settings()` (linea 70):

```python
# Reset visual report tracking state
st.session_state.pop("_visual_report_prev_selection", None)
st.session_state.pop("_visual_report_last_default_x_label", None)

# Reset quality mode to default
st.session_state.pop("_quality_file_sig", None)
```

## Test Cases (Manual Testing)

### Test 1: Reset Base Controls
**Steps:**
1. Carica un CSV file
2. Seleziona colonna X = "Time [s]"
3. Seleziona colonne Y = ["Signal [V]", "Noise [V]"]
4. Imposta limiti X: min=0, max=100
5. Imposta limiti Y: min=-5, max=5
6. Clicca "Applica / Plot"
7. Clicca "Reset impostazioni"

**Expected Result:**
- ✅ Colonna X torna a "—"
- ✅ Colonne Y deselezionate
- ✅ Limiti assi puliti (campi vuoti)
- ✅ Grafici nascosti (riappare messaggio "Compila il form...")

---

### Test 2: Reset Advanced Settings
**Steps:**
1. Carica CSV
2. Espandi "Advanced"
3. Imposta fs manuale = 1000 Hz
4. Abilita filtro = ON
5. Tipo filtro = "Butterworth LP"
6. Cutoff low = 50
7. Overlay originale = OFF
8. Abilita FFT = ON
9. Detrend = OFF
10. Clicca "Reset impostazioni"

**Expected Result:**
- ✅ fs manuale torna a 0
- ✅ Filtro disabilitato
- ✅ Tipo filtro torna a "Media mobile (MA)"
- ✅ Tutti i campi numerici ai valori default
- ✅ Overlay originale = ON (default)
- ✅ FFT disabilitata (se dataset >128 righe)

---

### Test 3: Reset Quality Mode
**Steps:**
1. Carica file GRANDE (>100k righe, es. 08_big_signal.csv)
2. Verifica che "Prestazioni" sia selezionato automaticamente
3. Cambia manualmente a "Alta fedeltà"
4. Clicca "Applica / Plot"
5. Clicca "Reset impostazioni"

**Expected Result:**
- ✅ Modalità qualità torna a "Prestazioni" (default per file grandi)

**Test 3b: Small File**
1. Carica file PICCOLO (<100k righe, es. 01_basic.csv)
2. Verifica che "Alta fedeltà" sia selezionato automaticamente
3. Cambia a "Prestazioni"
4. Clicca "Reset impostazioni"

**Expected Result:**
- ✅ Modalità qualità torna a "Alta fedeltà" (default per file piccoli)

---

### Test 4: Reset Report Visivo
**Steps:**
1. Carica CSV multi-colonna
2. Seleziona 3 colonne per report visivo
3. Espandi "Grafico 1" e imposta:
   - Titolo = "Test Signal 1"
   - Asse X = "Tempo [ms]"
   - Asse Y = "Ampiezza [mV]"
4. Espandi "Grafico 2" e imposta valori custom
5. Imposta "Titolo report visivo" = "My Custom Report"
6. Clicca "Reset impostazioni"

**Expected Result:**
- ✅ Titoli grafici tornano ai nomi colonna
- ✅ Asse X torna al default (colonna X o "Index")
- ✅ Asse Y torna al nome colonna
- ✅ Titolo report visivo pulito
- ✅ Nome file opzionale pulito

---

### Test 5: Reset Report Statistico
**Steps:**
1. Carica CSV
2. Imposta formato report = "csv+md+html"
3. Imposta nome base = "test_report_123"
4. Clicca "Genera report"
5. Verifica report generato
6. Clicca "Reset impostazioni"

**Expected Result:**
- ✅ Formato torna a "csv"
- ✅ Nome base pulito
- ✅ Report generato rimosso (pulsanti download spariscono)

---

### Test 6: Persistenza Cache File
**Steps:**
1. Carica CSV grande (es. 08_big_signal.csv)
2. Attendi caricamento completo
3. Imposta parametri vari e clicca "Applica / Plot"
4. Clicca "Reset impostazioni"
5. Riseleziona colonne e clicca "Applica / Plot"

**Expected Result:**
- ✅ File NON ricaricato (cache intatta)
- ✅ Nessun spinner "Analisi CSV..."
- ✅ Grafici si generano immediatamente

**IMPORTANTE:** Il reset NON deve invalidare la cache del DataFrame!

---

### Test 7: Multiple Resets
**Steps:**
1. Carica CSV
2. Imposta parametri complessi
3. Clicca "Reset impostazioni" (1° volta)
4. Imposta parametri diversi
5. Clicca "Reset impostazioni" (2° volta)
6. Imposta parametri ancora diversi
7. Clicca "Reset impostazioni" (3° volta)

**Expected Result:**
- ✅ Ogni reset pulisce completamente lo stato
- ✅ Nessun "accumulo" di stato tra reset
- ✅ Nessun errore console/log

---

## Regression Tests

### Regression 1: File Upload dopo Reset
**Steps:**
1. Carica file A
2. Imposta parametri
3. Clicca "Reset impostazioni"
4. Carica file B (diverso)

**Expected Result:**
- ✅ File B caricato correttamente
- ✅ Metadati di file A completamente rimossi
- ✅ Form inizializza con valori default per file B

---

### Regression 2: Sample Loading dopo Reset
**Steps:**
1. Clicca "Carica sample"
2. Imposta parametri
3. Clicca "Reset impostazioni"
4. Clicca "Carica sample" di nuovo

**Expected Result:**
- ✅ Sample ricaricato con stato pulito
- ✅ Nessun parametro residuo dalla sessione precedente

---

## Automated Test (Future)

```python
# tests/test_reset_button.py
import streamlit as st
from web_app import _reset_all_settings, RESETTABLE_KEYS

def test_reset_clears_all_keys():
    """Verify all resettable keys are cleared."""
    # Setup: populate session state
    for key in RESETTABLE_KEYS:
        st.session_state[key] = "dummy_value"

    st.session_state["_plots_ready"] = True
    st.session_state["_generated_report"] = {"test": "data"}
    st.session_state["_visual_report_prev_selection"] = ["col1"]
    st.session_state["_quality_file_sig"] = "abc123"

    # Action: reset
    _reset_all_settings()

    # Assert: all cleared
    for key in RESETTABLE_KEYS:
        assert key not in st.session_state

    assert "_plots_ready" not in st.session_state
    assert "_generated_report" not in st.session_state
    assert "_visual_report_prev_selection" not in st.session_state
    assert "_quality_file_sig" not in st.session_state
    assert st.session_state.get("_controls_nonce", 0) > 0


def test_reset_preserves_cache():
    """Verify reset does NOT clear file cache."""
    # Setup: simulate cached file
    st.session_state["_cached_df"] = "mock_dataframe"
    st.session_state["_cached_meta"] = {"encoding": "utf-8"}
    st.session_state["_cached_file_sig"] = ("size", "hash")

    # Action: reset
    _reset_all_settings()

    # Assert: cache preserved
    assert "_cached_df" in st.session_state
    assert "_cached_meta" in st.session_state
    assert "_cached_file_sig" in st.session_state
```

---

## Acceptance Criteria

✅ **Test 1-7 passano** senza comportamenti anomali
✅ **Regression 1-2 passano** senza errori
✅ **Cache file preservata** dopo reset
✅ **Nessun warning/errore** in console browser o log Streamlit

## Status
- [x] Fix implementato
- [ ] Test manuali completati
- [ ] Issue #44 chiusa

---

**Note:** Questo test plan va eseguito **manualmente** lanciando `streamlit run web_app.py` e seguendo gli step.
