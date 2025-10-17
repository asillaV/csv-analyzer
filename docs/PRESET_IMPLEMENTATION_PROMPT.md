# Prompt per Implementazione Issue #36 - Sistema Preset

## 🎯 Obiettivo

Implementare un sistema completo di gestione preset per salvare, caricare e riutilizzare configurazioni di filtri e FFT nell'applicazione Analizzatore CSV. Questo migliora significativamente la UX per task di analisi ripetitivi.

---

## 📋 Context & Requirements

### Problema da Risolvere
Attualmente gli utenti devono riconfigurare manualmente i parametri di filtro e FFT ogni volta che analizzano un nuovo file CSV con le stesse impostazioni. Serve un sistema per salvare configurazioni riutilizzabili.

### Scope dell'Implementazione
1. **Core Module**: `core/preset_manager.py` - Gestione CRUD preset
2. **Test Suite**: `tests/test_preset_manager.py` - Unit tests completi
3. **UI Integration**: Modifiche a `web_app.py` - Interfaccia utente in expander

### Tecnologie & Constraints
- **Storage**: File JSON in `presets/` directory
- **Testing**: pytest, target coverage >80%
- **UI**: Streamlit expander (NON sidebar - richiesto dall'utente)
- **Python**: 3.10+, type hints required
- **Compatibilità**: Non rompere test esistenti

---

## 🏗️ Architettura

### 1. Core Module: `core/preset_manager.py`

Crea un modulo Python con le seguenti funzionalità:

#### Struttura Dati Preset
```python
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
from typing import Optional
from core.signal_tools import FilterSpec, FFTSpec

@dataclass
class PresetData:
    """Rappresenta un preset salvato."""
    version: str  # "1.0" per future migrations
    name: str
    description: str
    created_at: str  # ISO format datetime
    manual_fs: Optional[float]
    filter: dict  # FilterSpec serialized
    fft: dict  # FFTSpec serialized
```

#### Funzioni Richieste

**1. `save_preset(name: str, description: str, fspec: FilterSpec, fftspec: FFTSpec, manual_fs: Optional[float]) -> None`**
- Salva preset come JSON in `presets/<sanitized_name>.json`
- Sanitizza nome file: rimuovi `< > : " / \ | ? *`, max 50 chars
- Gestisci tuple → list conversion (cutoff può essere tuple)
- Solleva `PresetError` se salvataggio fallisce

**2. `load_preset(name: str) -> dict`**
- Carica preset da `presets/<sanitized_name>.json`
- Ritorna dict con chiavi: `manual_fs`, `filter_spec`, `fft_spec`
- Converti list → tuple per `cutoff` se necessario
- Solleva `PresetError` se preset non esiste o JSON corrotto

**3. `list_presets() -> list[dict]`**
- Ritorna lista di dict con `name` e `description` di tutti i preset
- Ordinati alfabeticamente per nome
- Ignora file non-JSON o corrotti

**4. `delete_preset(name: str) -> None`**
- Elimina preset file
- Solleva `PresetError` se preset non esiste

**5. `preset_exists(name: str) -> bool`**
- Verifica se preset esiste

**6. `get_preset_info(name: str) -> dict`**
- Ritorna solo metadata (nome, descrizione, created_at) senza caricare filter/fft
- Per preview veloce

**7. `create_default_presets() -> None`**
- Crea 5 preset di default **solo se non esistono già** (idempotent)
- Preset richiesti:
  - **Media Mobile 5**: MA window=5, no FFT
  - **Media Mobile 20**: MA window=20, no FFT
  - **Butterworth LP 50Hz**: Butterworth LP cutoff=50, order=4, no FFT
  - **Analisi Vibrazione Completa**: Butterworth BP 10-100Hz order=4 + FFT detrend=True window=hann
  - **Solo FFT**: No filter, solo FFT detrend=True window=hann

#### Eccezioni Custom
```python
class PresetError(Exception):
    """Raised for preset operations errors."""
    pass
```

#### File Path Management
```python
PRESETS_DIR = Path(__file__).parent.parent / "presets"

def _sanitize_filename(name: str) -> str:
    """Rimuovi caratteri invalidi e limita lunghezza."""
    # Rimuovi < > : " / \ | ? *
    # Max 50 caratteri
    # Se vuoto dopo sanitization, usa timestamp
    pass
```

---

### 2. Test Suite: `tests/test_preset_manager.py`

Crea test suite con **almeno 14 test cases**:

```python
import pytest
from pathlib import Path
from core.preset_manager import (
    save_preset, load_preset, list_presets, delete_preset,
    preset_exists, get_preset_info, create_default_presets,
    PresetError, _sanitize_filename
)
from core.signal_tools import FilterSpec, FFTSpec

@pytest.fixture
def temp_presets_dir(tmp_path, monkeypatch):
    """Usa directory temporanea per i test."""
    monkeypatch.setattr("core.preset_manager.PRESETS_DIR", tmp_path)
    return tmp_path

# Test Cases Richiesti:
def test_sanitize_filename():
    # Test rimozione caratteri invalidi
    assert _sanitize_filename("test<>:file") == "testfile"
    assert _sanitize_filename("a" * 60) == "a" * 50
    assert _sanitize_filename("  ") != ""  # Fallback a timestamp

def test_save_and_load_preset(temp_presets_dir):
    # Test ciclo save → load
    fspec = FilterSpec(kind="ma", enabled=True, cutoff=0.0, order=2, ma_window=5)
    fftspec = FFTSpec(enabled=False, detrend=False, window="hann")
    save_preset("Test", "Description", fspec, fftspec, None)

    loaded = load_preset("Test")
    assert loaded["manual_fs"] is None
    assert loaded["filter_spec"].ma_window == 5

def test_save_preset_with_bandpass_filter(temp_presets_dir):
    # Test tuple cutoff conversion
    fspec = FilterSpec(kind="butter_bp", enabled=True, cutoff=(10.0, 100.0), order=4, ma_window=5)
    # ... salva e verifica che tuple sia gestita correttamente

def test_list_presets(temp_presets_dir):
    # Test listing e ordinamento alfabetico
    save_preset("Zebra", "Last", ...)
    save_preset("Alpha", "First", ...)
    presets = list_presets()
    assert presets[0]["name"] == "Alpha"
    assert presets[1]["name"] == "Zebra"

def test_delete_preset(temp_presets_dir):
    # Test cancellazione
    save_preset("ToDelete", "", ...)
    assert preset_exists("ToDelete")
    delete_preset("ToDelete")
    assert not preset_exists("ToDelete")

def test_load_nonexistent_preset(temp_presets_dir):
    # Test error handling
    with pytest.raises(PresetError):
        load_preset("NotExist")

def test_load_corrupted_preset(temp_presets_dir):
    # Test JSON corrotto
    (temp_presets_dir / "bad.json").write_text("{invalid json")
    with pytest.raises(PresetError):
        load_preset("bad")

def test_create_default_presets(temp_presets_dir):
    # Test creazione default
    create_default_presets()
    assert preset_exists("Media Mobile 5")
    assert preset_exists("Solo FFT")

def test_create_default_presets_idempotent(temp_presets_dir):
    # Test che ri-chiamare non sovrascrive
    create_default_presets()
    first_time = (temp_presets_dir / "Media Mobile 5.json").stat().st_mtime
    create_default_presets()
    second_time = (temp_presets_dir / "Media Mobile 5.json").stat().st_mtime
    assert first_time == second_time  # Non sovrascritto

# Altri test: special characters, disabled specs, get_preset_info, etc.
```

**Target Coverage**: >85%

**Comando Test**: `.venv\Scripts\python.exe -m pytest tests/test_preset_manager.py -v`

---

### 3. UI Integration: `web_app.py`

#### Posizionamento UI: Expander SOTTO Advanced Panel

**IMPORTANTE**: L'utente richiede che il preset UI sia in un **expander**, NON nella sidebar. Posizionalo **subito dopo il pannello Advanced** (dopo l'expander con filtri/FFT) nella sezione principale.

#### Modifiche Richieste

**1. Imports (linee ~30)**
```python
from core.preset_manager import (
    save_preset, load_preset, list_presets, delete_preset,
    preset_exists, create_default_presets, PresetError
)
```

**2. Inizializzazione App (prima del form, linee ~760)**
```python
# Crea preset di default se non esistono
try:
    create_default_presets()
except Exception as e:
    logger.warning(f"Impossibile creare preset default: {e}")
```

**3. Preset UI Expander (DOPO Advanced panel, linee ~1050+)**
```python
with st.expander("🎯 Preset Configurazioni"):
    st.markdown("Salva e riutilizza configurazioni filtri/FFT frequenti.")

    # Lista preset disponibili
    try:
        available_presets = list_presets()
        preset_names = [p["name"] for p in available_presets]
    except Exception as e:
        st.error(f"Errore caricamento preset: {e}")
        preset_names = []

    # Layout: 2 colonne
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_preset = st.selectbox(
            "Preset disponibili",
            options=["---"] + preset_names,
            key="preset_selector"
        )

    with col2:
        load_clicked = st.button("📂 Carica", disabled=selected_preset == "---")

    # Buttons sotto: Save e Delete
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])

    with btn_col1:
        save_clicked = st.button("💾 Salva nuovo")

    with btn_col2:
        delete_clicked = st.button("🗑️ Elimina", disabled=selected_preset == "---")

    # Logica Load
    if load_clicked and selected_preset != "---":
        try:
            preset_data = load_preset(selected_preset)
            # Salva in session_state per popolare form al prossimo rerun
            st.session_state["_loaded_preset"] = preset_data
            st.session_state["_loaded_preset_name"] = selected_preset
            st.success(f"✅ Preset '{selected_preset}' caricato! Controlla i parametri Advanced.")
            st.rerun()
        except PresetError as e:
            st.error(f"Errore caricamento: {e}")

    # Logica Save (dialog)
    if save_clicked:
        st.session_state["_show_save_dialog"] = True

    if st.session_state.get("_show_save_dialog"):
        with st.form("save_preset_form"):
            new_name = st.text_input("Nome preset", placeholder="Es: Vibrazione 50Hz")
            new_desc = st.text_area("Descrizione (opzionale)", placeholder="Butterworth LP + FFT...")

            save_col1, save_col2 = st.columns(2)
            with save_col1:
                save_confirmed = st.form_submit_button("✓ Salva")
            with save_col2:
                cancel = st.form_submit_button("✗ Annulla")

            if save_confirmed and new_name.strip():
                # Flag per salvare DOPO che form Advanced viene submitted
                st.session_state["_pending_preset_save"] = {
                    "name": new_name.strip(),
                    "description": new_desc.strip()
                }
                st.session_state.pop("_show_save_dialog", None)
                st.info("⏳ Configura filtri/FFT e clicca 'Applica/Plot' per completare il salvataggio.")
                st.rerun()

            if cancel:
                st.session_state.pop("_show_save_dialog", None)
                st.rerun()

    # Logica Delete
    if delete_clicked and selected_preset != "---":
        try:
            delete_preset(selected_preset)
            st.success(f"🗑️ Preset '{selected_preset}' eliminato.")
            st.rerun()
        except PresetError as e:
            st.error(f"Errore eliminazione: {e}")
```

**4. Preset Loading Logic (PRIMA del form Advanced, linee ~1100)**
```python
# Popola valori default da preset caricato
preset_fs = None
preset_enable_filter = False
preset_filter_kind_idx = 0
preset_ma_win = 5
preset_filter_order = 2
preset_f_lo = ""
preset_f_hi = ""
preset_enable_fft = False
preset_detrend = False

if "_loaded_preset" in st.session_state:
    preset_data = st.session_state.pop("_loaded_preset")
    preset_name = st.session_state.pop("_loaded_preset_name", "")

    # Mapping da preset a widget values
    preset_fs = preset_data.get("manual_fs")

    fspec = preset_data.get("filter_spec")
    if fspec:
        preset_enable_filter = fspec.enabled
        # Map kind to index (ma=0, butter_lp=1, butter_hp=2, butter_bp=3)
        kind_map = {"ma": 0, "butter_lp": 1, "butter_hp": 2, "butter_bp": 3}
        preset_filter_kind_idx = kind_map.get(fspec.kind, 0)
        preset_ma_win = fspec.ma_window
        preset_filter_order = fspec.order

        # Cutoff handling
        if isinstance(fspec.cutoff, tuple):
            preset_f_lo, preset_f_hi = fspec.cutoff
        else:
            if "lp" in fspec.kind:
                preset_f_hi = fspec.cutoff
            elif "hp" in fspec.kind:
                preset_f_lo = fspec.cutoff

    fftspec = preset_data.get("fft_spec")
    if fftspec:
        preset_enable_fft = fftspec.enabled
        preset_detrend = fftspec.detrend

    st.info(f"📂 Preset '{preset_name}' applicato ai parametri.")
```

**5. Widget Population (nel form Advanced)**

Modifica tutti i widget per usare i valori preset come `value=`:

```python
with st.expander("⚙️ Advanced"):
    manual_fs = st.number_input(
        "Frequenza campionamento (Hz)",
        value=preset_fs if preset_fs else 0.0,  # ← Usa preset
        ...
    )

    enable_filter = st.checkbox(
        "Abilita filtro",
        value=preset_enable_filter,  # ← Usa preset
        ...
    )

    f_kind = st.selectbox(
        "Tipo filtro",
        options=["Media mobile", "Butterworth LP", "Butterworth HP", "Butterworth BP"],
        index=preset_filter_kind_idx,  # ← Usa preset
        ...
    )

    # ... e così via per tutti i widget
```

**6. Preset Save Logic (DOPO form submit)**

```python
if submitted:
    # ... costruisci fspec e fftspec come già fai ...

    # Se c'è un salvataggio pending
    if "_pending_preset_save" in st.session_state:
        save_info = st.session_state.pop("_pending_preset_save")
        try:
            save_preset(
                name=save_info["name"],
                description=save_info["description"],
                fspec=fspec,
                fftspec=fftspec,
                manual_fs=manual_fs if manual_fs > 0 else None
            )
            st.success(f"💾 Preset '{save_info['name']}' salvato con successo!")
        except PresetError as e:
            st.error(f"Errore salvataggio preset: {e}")
```

---

## 📐 Design Guidelines

### UX Principles
1. **Non-invasivo**: Expander chiuso di default, non ingombra UI principale
2. **Feedback chiaro**: Success/error messages per ogni operazione
3. **Flusso naturale**: Load → Modify → Plot OR Configure → Plot → Save
4. **Sicurezza**: Conferma prima di delete (già implementato con disabled state)

### Posizionamento UI

```
[Carica CSV]
[Seleziona X/Y]
[Anteprima DataFrame]

▼ ⚙️ Advanced (expander)
  [Freq. campionamento]
  [Filtri]
  [FFT]

▼ 🎯 Preset Configurazioni (expander) ← NUOVO, SUBITO DOPO Advanced
  [Dropdown preset]
  [Carica] [Salva] [Elimina]

[Applica / Plot] ← Form submit button
```

### Styling Considerations
- **No custom CSS** richiesto (usa Streamlit default)
- **Icone emoji** per chiarezza visiva (📂 💾 🗑️ ✓ ✗)
- **Colonne** per layout pulito (selectbox + button fianco a fianco)

---

## ✅ Acceptance Criteria

### Funzionalità
- [ ] Salvare preset con nome e descrizione custom
- [ ] Caricare preset e popolare automaticamente form
- [ ] Eliminare preset esistenti
- [ ] 5 preset di default creati automaticamente
- [ ] Preset persistono tra restart app

### Qualità Codice
- [ ] Type hints completi
- [ ] Docstrings per tutte le funzioni pubbliche
- [ ] Error handling robusto (PresetError per tutti i casi)
- [ ] Nessun warning linter

### Testing
- [ ] 14+ test cases passanti
- [ ] Coverage >85%
- [ ] Test automatizzati non rompono test esistenti
- [ ] Manual testing checklist completato

### Performance
- [ ] Load preset <5ms
- [ ] Save preset <10ms
- [ ] List presets <20ms (anche con 50+ preset)

---

## 🧪 Testing Workflow

### Automated Tests
```bash
# Run solo i nuovi test
.venv\Scripts\python.exe -m pytest tests/test_preset_manager.py -v

# Run con coverage
.venv\Scripts\python.exe -m pytest tests/test_preset_manager.py --cov=core.preset_manager --cov-report=term

# Run TUTTI i test (verifica non rompere esistenti)
.venv\Scripts\python.exe -m pytest tests/ -v
```

### Manual Testing Checklist

1. **Setup**
   - [ ] Avvia app: `streamlit run web_app.py`
   - [ ] Verifica preset default esistono nell'expander

2. **Load Preset**
   - [ ] Seleziona "Media Mobile 5"
   - [ ] Click "Carica"
   - [ ] Verifica messaggio success
   - [ ] Apri Advanced: MA window deve essere 5
   - [ ] Click "Applica/Plot": filtro MA deve applicarsi

3. **Modify & Save New**
   - [ ] Modifica MA window a 10
   - [ ] Click "Salva nuovo"
   - [ ] Inserisci nome "Media Mobile 10"
   - [ ] Inserisci descrizione "Smoothing leggero"
   - [ ] Click "Applica/Plot"
   - [ ] Verifica messaggio success salvataggio
   - [ ] Riapri expander: "Media Mobile 10" deve apparire nel dropdown

4. **Delete Preset**
   - [ ] Seleziona "Media Mobile 10"
   - [ ] Click "Elimina"
   - [ ] Verifica messaggio success
   - [ ] Preset non deve più apparire nel dropdown

5. **Complex Preset** (Butterworth + FFT)
   - [ ] Seleziona "Analisi Vibrazione Completa"
   - [ ] Click "Carica"
   - [ ] Verifica: Butterworth BP 10-100Hz, order=4, FFT enabled, detrend=True
   - [ ] Click "Applica/Plot": grafico + FFT devono apparire

6. **Persistence**
   - [ ] Ferma app (Ctrl+C)
   - [ ] Riavvia app
   - [ ] Verifica preset salvati persistono

7. **Edge Cases**
   - [ ] Prova nome preset con caratteri speciali: "Test<>:File"
   - [ ] Verifica file salvato senza caratteri invalidi
   - [ ] Prova caricare preset, chiudere expander, riaprire: valori devono persistere

---

## 🚨 Common Pitfalls & Solutions

### Problema: Form non si popola con preset
**Causa**: Streamlit widgets non si aggiornano senza rerun
**Soluzione**: Usa `st.rerun()` dopo load preset + salva preset data in `session_state`

### Problema: Tuple/List conversion errors
**Causa**: JSON non supporta tuple, solo list
**Soluzione**: Converti esplicitamente in `load_preset()`:
```python
if isinstance(cutoff_val, list) and len(cutoff_val) == 2:
    cutoff_val = tuple(cutoff_val)
```

### Problema: Preset salvato durante load invece che dopo Plot
**Causa**: Click "Salva" triggera save immediato
**Soluzione**: Usa flag `_pending_preset_save` e salva solo dopo form submit

### Problema: Default preset sovrascritti ad ogni restart
**Causa**: `create_default_presets()` riscrive file
**Soluzione**: Check `preset_exists()` prima di creare:
```python
if not preset_exists("Media Mobile 5"):
    save_preset("Media Mobile 5", ...)
```

---

## 📝 File Checklist

Al termine implementazione, dovresti avere:

- [ ] `core/preset_manager.py` (~350-400 linee)
- [ ] `tests/test_preset_manager.py` (~250-300 linee, 14+ test)
- [ ] `presets/` directory creata automaticamente
- [ ] `presets/Media Mobile 5.json` (e altri 4 default)
- [ ] `web_app.py` modificato (~100 linee aggiunte/modificate)
- [ ] Tutti i test passanti (vecchi + nuovi)
- [ ] No errori linter/type checker

---

## 🎓 Summary

**Obiettivo**: Sistema preset completo, testato, integrato in UI non-invasiva.

**Approccio**:
1. Core module puro (no Streamlit dependencies)
2. Test-first development (scrivi test prima di UI)
3. UI integration last (expander dopo Advanced panel)

**Deliverable**:
- Funzionalità: Save/Load/Delete preset
- Testing: >85% coverage, 14+ test cases
- UX: Expander non-invasivo, feedback chiaro
- Performance: <10ms per operazione

**Tempo stimato**: 6-8 ore (3h core+tests, 3h UI integration, 2h testing+polish)

---

Buona implementazione! 🚀
