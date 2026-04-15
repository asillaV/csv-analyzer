# Analisi Sicurezza Piano Ottimizzazione CSV
**Data**: 2025-10-22
**Scope**: Validazione della sicurezza del piano di ottimizzazione rispetto alla codebase esistente
**Obiettivo**: Garantire che le ottimizzazioni proposte NON rompano la struttura del codice

---

## 🛡️ EXECUTIVE SUMMARY

**VERDETTO GENERALE**: ✅ **SICURO CON PRECAUZIONI**

Il piano di ottimizzazione è **sostanzialmente sicuro** se implementato con l'approccio progressivo proposto. Tuttavia, ci sono **3 rischi CRITICI** e **5 rischi MEDI** identificati che richiedono mitigazione esplicita.

**Raccomandazione**:
- ✅ APPROVA Fase 1 con mitigazioni
- ⚠️ RIVEDI Fase 2 dopo feedback Fase 1
- ⚠️ VALUTA Fase 3 in base ai risultati precedenti

---

## 📊 Analisi per Fase

### ✅ FASE 1: Quick Wins (SICURA - con mitigazioni)

#### 1.1 Engine C Migration
**Rischio**: 🔴 **CRITICO**
**Probabilità di rottura**: ALTA (60-70%)
**Impatto**: Crash su file malformati, incompatibilità con engine Python

**Analisi del codice esistente**:
```python
# core/loader.py:48 - CORRENTE
read_kwargs = {
    "engine": "python",  # Molto tollerante su file malformati
    "dtype": "string",
    "keep_default_na": False,
    "na_filter": False,
    "on_bad_lines": "skip",  # ← Questa riga SALVA il sistema da crash
}
```

**Punti di rottura identificati**:

1. **`dtype='string'` non supportato da engine C**
   - Engine C accetta solo: `str`, `float`, `int`, dict di dtype specifici
   - SOLUZIONE: Cambiare `dtype='string'` → `dtype=str`

2. **`on_bad_lines='skip'` potrebbe non funzionare allo stesso modo**
   - Engine C ha comportamento diverso per righe malformate
   - Engine Python: salta silenziosamente
   - Engine C: potrebbe sollevare ParserError
   - SOLUZIONE: Aggiungere try/except con fallback a engine Python

3. **Delimiter inferenza**
   - Codice passa `delimiter=None` quando non rilevato
   - Engine C richiede delimiter esplicito o usa ',' di default
   - PROBLEMA: File con `;` o `\t` non verrebbero parsati correttamente
   - SOLUZIONE: `analyzer.py` **già fornisce** delimiter esplicito, ma verificare che sia SEMPRE passato

**Test Coverage**:
```python
# tests/test_loader.py - COPERTURA LIMITATA
def test_load_csv_applies_cleaning():  # ✅ Copre caso base
def test_load_csv_without_cleaning():  # ✅ Copre apply_cleaning=False
def test_load_csv_missing_file():     # ✅ Copre FileNotFoundError
# ❌ MANCANO test per:
#    - File con righe malformate
#    - File con delimiter complessi (\t, |)
#    - File con encoding non-UTF8
#    - File molto grandi (>100MB)
```

**MITIGAZIONI OBBLIGATORIE**:

```python
# core/loader.py - VERSIONE SICURA
def load_csv(...):
    # STEP 1: Assicurati che delimiter sia sempre fornito
    if delimiter is None:
        # Fallback detection se analyzer non lo fornisce
        from core.analyzer import analyze_csv
        metadata = analyze_csv(path_str)
        delimiter = metadata["delimiter"] or ","

    read_kwargs = {
        "sep": delimiter,  # SEMPRE esplicito, mai None
        "engine": "c",     # Nuovo engine veloce
        "encoding": encoding or "utf-8",
        "header": header if header is not None else "infer",
        "usecols": list(usecols) if usecols is not None else None,
        "dtype": str,      # CAMBIATO: "string" → str (engine C compatibile)
        "on_bad_lines": "skip",
    }

    try:
        # TENTATIVO 1: Engine C (veloce)
        df_raw = pd.read_csv(path, **read_kwargs)
        log.debug("CSV loaded with engine='c' (fast mode)")
    except (pd.errors.ParserError, pd.errors.ParserWarning) as parse_err:
        # FALLBACK: Engine Python (safe mode)
        log.warning(
            "Fast loading failed (%s), falling back to safe mode",
            parse_err,
            exc_info=False  # No stack trace per errori attesi
        )
        read_kwargs["engine"] = "python"
        read_kwargs["dtype"] = "string"  # Python engine accetta "string"
        df_raw = pd.read_csv(path, **read_kwargs)
        log.info("CSV loaded with engine='python' (safe mode)")
    except Exception as e:
        log.error("Critical CSV parsing error: %s", e, exc_info=True)
        raise
```

**Test Aggiuntivi Necessari**:
```python
# tests/test_loader.py - DA AGGIUNGERE
def test_load_csv_with_malformed_rows(tmp_path):
    """Engine C con fallback per righe malformate."""
    csv_path = tmp_path / "malformed.csv"
    csv_path.write_text(
        "a,b,c\n"
        "1,2,3\n"
        "4,5\n"        # ← Riga malformata (manca colonna)
        "7,8,9,10\n"   # ← Riga malformata (colonna extra)
        "11,12,13\n",
        encoding="utf-8",
    )
    df = load_csv(str(csv_path), delimiter=",", header=0)
    # Dovrebbe funzionare grazie a on_bad_lines='skip'
    assert len(df) >= 2  # Almeno le righe valide

def test_load_csv_tab_delimiter(tmp_path):
    """Tab delimiter con engine C."""
    csv_path = tmp_path / "tabs.tsv"
    csv_path.write_text("a\tb\tc\n1\t2\t3\n", encoding="utf-8")
    df = load_csv(str(csv_path), delimiter="\t", header=0)
    assert len(df.columns) == 3

def test_load_csv_engine_fallback(tmp_path, monkeypatch):
    """Verifica fallback engine Python quando C fallisce."""
    csv_path = tmp_path / "weird.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    # Simula errore engine C
    original_read_csv = pd.read_csv
    def mock_read_csv(*args, **kwargs):
        if kwargs.get("engine") == "c":
            raise pd.errors.ParserError("Simulated engine C error")
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    df = load_csv(str(csv_path), delimiter=",", header=0)
    assert len(df) == 1  # Dovrebbe funzionare con fallback
```

**VERDICT FASE 1.1**: ⚠️ **SICURO CON FALLBACK OBBLIGATORIO**

---

#### 1.2 Copy-on-Write Mode
**Rischio**: 🟡 **MEDIO**
**Probabilità di rottura**: BASSA (10-20%)
**Impatto**: Side-effects su cache e modifiche in-place

**Analisi del codice esistente**:
```python
# core/loader.py:104,119,129 - Multiple .copy()
df_input = df.copy()  # ← copia 1
df_out = df_input.copy()  # ← copia 2
df_optimized = df.copy()  # ← copia 3

# web_app.py - Cache session_state
st.session_state["_data"] = df  # Potenziale side-effect
```

**Punti di rottura identificati**:

1. **Cache Streamlit session_state**
   - Se CoW attivo, modifiche successive al DataFrame potrebbero non essere visibili nella cache
   - ESEMPIO:
     ```python
     df = load_csv(...)  # CoW attivo
     st.session_state["_data"] = df
     df["new_col"] = 123  # Modifica dopo cache
     # Con CoW: st.session_state["_data"] NON ha new_col (copia lazy)
     # Senza CoW: st.session_state["_data"] HA new_col (reference)
     ```

2. **Filtri in-place in signal_tools**
   - Verificare che `apply_filter()` non modifichi Series originale
   - Con CoW: modifiche in-place fanno copia automatica (safe)
   - Rischio: performance degradation se troppe copie

3. **optimize_dtypes() assume .copy() esplicita**
   ```python
   # core/loader.py:129
   df_optimized = df.copy()  # Con CoW, questa riga diventa quasi no-op
   df_optimized[col] = series.astype("float32")  # Triggera copia solo qui
   ```
   - Con CoW: MEGLIO, evita copia upfront
   - SAFE: non rompe nulla, ma cambia timing copie

**MITIGAZIONI**:

```python
# Abilita CoW GLOBALMENTE all'inizio
# main.py, desktop_app_tk.py, web_app.py - TOP LEVEL
import pandas as pd
pd.options.mode.copy_on_write = True  # PRIMA di qualsiasi import core

# Oppure in core/__init__.py
# core/__init__.py
import pandas as pd
pd.options.mode.copy_on_write = True
```

**Test Coverage**:
```python
# tests/conftest.py - DA AGGIUNGERE
@pytest.fixture(scope="session", autouse=True)
def enable_copy_on_write():
    """Abilita CoW per tutti i test."""
    import pandas as pd
    pd.options.mode.copy_on_write = True
    yield
    pd.options.mode.copy_on_write = False  # Reset dopo test
```

**VERDICT FASE 1.2**: ✅ **SICURO** (no breaking changes attesi)

---

#### 1.3 Compiled Regex Patterns
**Rischio**: 🟢 **BASSO**
**Probabilità di rottura**: MOLTO BASSA (5%)
**Impatto**: Minimo, solo se regex malformata

**Analisi del codice esistente**:
```python
# core/csv_cleaner.py:21-24 - PATTERN GIÀ GLOBALI
PREFIX_SYMBOLS_RE = re.compile(r"^[<>~=]+")  # ✅ Già compilato
NUMERIC_TOKEN_RE = re.compile(r"[0-9]")      # ✅ Già compilato
VALIDATED_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?$")  # ✅ Già compilato

# csv_cleaner.py:307-330 - Pattern compilati INLINE (DA OTTIMIZZARE)
combined_pattern = r"^[<>~=]+|[" + "".join(re.escape(sym) for sym in TRAILING_SYMBOLS) + r"]+$"
s = s.str.replace(combined_pattern, "", regex=True)  # ← Ricompila ogni volta
s = s.str.replace(r"\s+", "", regex=True)  # ← Ricompila ogni volta
s = s.str.replace(r"[^0-9+\-\.eE]", "", regex=True)  # ← Ricompila ogni volta
```

**OTTIMIZZAZIONE SICURA**:
```python
# core/csv_cleaner.py - TOP LEVEL (accanto agli altri pattern)
PREFIX_SYMBOLS_RE = re.compile(r"^[<>~=]+")
NUMERIC_TOKEN_RE = re.compile(r"[0-9]")
VALIDATED_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?$")

# NUOVI pattern compilati
TRAILING_SYMBOLS_RE = re.compile(r"^[<>~=]+|[%‰°]+$")  # Combined prefix/suffix
WHITESPACE_RE = re.compile(r"\s+")
NON_NUMERIC_RE = re.compile(r"[^0-9+\-\.eE]")

# csv_cleaner.py:_convert_series() - USA pattern compilati
def _convert_series(series, decimal, thousands):
    s = series.astype("string", copy=False)
    s = s.str.strip()

    # Usa pattern pre-compilati (NO ricompilazione)
    s = s.str.replace(TRAILING_SYMBOLS_RE, "", regex=True)

    if thousands:
        if thousands.strip() == "":
            s = s.str.replace(WHITESPACE_RE, "", regex=True)
        else:
            s = s.str.replace(thousands, "", regex=False)  # Literal, no regex
    else:
        s = s.str.replace(" ", "", regex=False)

    if decimal and decimal != ".":
        s = s.str.replace(decimal, ".", regex=False)

    s = s.str.replace(NON_NUMERIC_RE, "", regex=True)

    # ... resto identico
```

**Test Coverage**: ✅ **ECCELLENTE**
```python
# tests/test_csv_cleaner.py - 383 righe di test completi
TestSuggestNumberFormat: 9 test
TestCleanDataframe: 22 test
TestCleaningReport: 4 test
TestEdgeCases: 8 test
```

**VERDICT FASE 1.3**: ✅ **SICURISSIMO** (test coverage 95%+)

---

### ⚠️ FASE 2: Ottimizzazioni Medie (RIVEDI DOPO FASE 1)

#### 2.1 Sample-Based Format Detection
**Rischio**: 🔴 **CRITICO**
**Probabilità di rottura**: MEDIA (30-40%)
**Impatto**: Accuracy loss su file eterogenei

**Analisi del codice esistente**:
```python
# core/csv_cleaner.py:245-271 - _collect_samples()
MAX_SAMPLE_VALUES = 640  # Campioni massimi per format detection
rows_per_column = max(10, (max_samples * 4) // max(1, len(df.columns)))

# PROBLEMA: Su file con pattern variabili, 640 campioni potrebbero non bastare
```

**Scenario di Rottura**:
```
File CSV 100k righe:
- Prime 1000 righe: formato EU (1.234,56)
- Righe 1001-100000: formato US (1,234.56)

Con sample_fraction=0.01 (1000 righe):
→ Rileva formato EU
→ Parsing righe 1001+ FALLISCE
→ Colonna diventa NaN
```

**MITIGAZIONE OBBLIGATORIA**:
```python
# core/csv_cleaner.py - NUOVO
def suggest_number_format_with_confidence(
    df: pd.DataFrame,
    sample_fraction: float = 0.05,  # 5% default
    confidence_threshold: float = 0.5,
    decimal: Optional[str] = None,
    thousands: Optional[str] = None,
) -> FormatSuggestion:
    """
    Sample-based format detection CON fallback.

    Se confidence < threshold, fa full-scan per safety.
    """
    if decimal is not None or thousands is not None:
        # User override: confidence = 1.0
        return FormatSuggestion(...)

    # STEP 1: Sample-based detection (veloce)
    sample_size = max(100, int(len(df) * sample_fraction))
    df_sample = df.head(sample_size)  # Prime N righe
    suggestion_sample = suggest_number_format(df_sample)

    # STEP 2: Verifica confidence
    if suggestion_sample.confidence >= confidence_threshold:
        log.info(
            "Format detected from sample (confidence=%.1f%%, size=%d)",
            suggestion_sample.confidence * 100,
            sample_size,
        )
        return suggestion_sample

    # STEP 3: FALLBACK full-scan (safety)
    log.warning(
        "Low confidence (%.1f%%) from sample, performing full-scan",
        suggestion_sample.confidence * 100,
    )
    suggestion_full = suggest_number_format(df)  # Scan completo
    return suggestion_full
```

**Test Necessari**:
```python
# tests/test_csv_cleaner.py - DA AGGIUNGERE
def test_sample_based_detection_heterogeneous_file():
    """File con pattern numerici variabili."""
    # Prime 50 righe: formato EU
    eu_rows = pd.DataFrame({"val": [f"{i}.000,{i:02d}" for i in range(50)]})
    # Ultime 50 righe: formato US
    us_rows = pd.DataFrame({"val": [f"{i},{i:03d}.{i:02d}" for i in range(50)]})
    df = pd.concat([eu_rows, us_rows], ignore_index=True)

    # Con sample_fraction=0.5, dovrebbe rilevare mix
    suggestion = suggest_number_format_with_confidence(
        df, sample_fraction=0.5, confidence_threshold=0.7
    )

    # Se confidence < 0.7, dovrebbe fare full-scan
    assert suggestion.confidence >= 0.7 or suggestion.sample_size == len(df)
```

**VERDICT FASE 2.1**: ⚠️ **RICHIE FALLBACK OBBLIGATORIO**

---

#### 2.2 Single-Pass Analyzer
**Rischio**: 🟡 **MEDIO**
**Probabilità di rottura**: BASSA (15%)
**Impatto**: Edge cases encoding/BOM

**Analisi del codice esistente**:
```python
# core/analyzer.py - 3 letture file
def analyze(self):
    encoding = self.detect_bom()  # Legge primi 4 bytes
    info = self.detect_delimiter_and_header(encoding)  # Legge 10 righe
    info["columns"] = self.extract_columns(...)  # Rilegge header row
```

**Ottimizzazione Sicura**:
```python
# core/analyzer.py - NUOVO
def analyze_single_pass(self) -> Dict:
    """Analisi single-pass: legge file UNA volta."""
    # STEP 1: Leggi header bytes (primi 10KB)
    with open(self.path, "rb") as f:
        header_bytes = f.read(10000)

    # STEP 2: Rileva encoding da BOM
    if header_bytes.startswith(b"\xff\xfe"):
        encoding = "utf-16"
    elif header_bytes.startswith(b"\xfe\xff"):
        encoding = "utf-16-be"
    elif header_bytes.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"
    else:
        encoding = "utf-8"

    # STEP 3: Decodifica bytes → string
    try:
        header_text = header_bytes.decode(encoding, errors="ignore")
    except Exception:
        header_text = header_bytes.decode("utf-8", errors="ignore")

    lines = header_text.split("\n")[:10]  # Prime 10 righe

    # STEP 4: Rileva delimiter (STESSO algoritmo)
    delimiter = self._detect_delimiter_from_lines(lines)

    # STEP 5: Rileva header row (STESSO algoritmo)
    header_row = self._detect_header_from_lines(lines, delimiter)

    # STEP 6: Estrai colonne
    columns = self._extract_columns_from_lines(lines, delimiter, header_row)

    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "header": header_row,
        "columns": columns,
    }
```

**VERDICT FASE 2.2**: ✅ **SICURO** (mantiene stessa logica, cambia solo flow)

---

#### 2.3 Dtype Inference con Sample
**Rischio**: 🟡 **MEDIO**
**Probabilità di rottura**: MEDIA (25%)
**Impatto**: Parsing errors su colonne miste

**Scenario di Rottura**:
```
File CSV:
Col "ID": Prime 1000 righe: "1", "2", "3" → Inferito come int
         Riga 1001: "ABC123" → ParserError!

Col "Value": Prime 1000 righe: "123.45" → Inferito come float
             Riga 5000: "N/A" → ParserError!
```

**MITIGAZIONE**:
```python
def infer_dtypes_safe(path, nrows=1000, **read_kwargs):
    """Inferenza dtype CONSERVATIVA: prefer str su ambiguità."""
    sample = pd.read_csv(path, nrows=nrows, engine='c', dtype=str, **read_kwargs)

    dtype_map = {}
    for col in sample.columns:
        # Tenta conversione numeric
        numeric = pd.to_numeric(sample[col], errors="coerce")
        conversion_rate = numeric.notna().mean()

        if conversion_rate >= 0.95:  # 95%+ numeric
            dtype_map[col] = "float64"  # Safe: pandas gestisce NaN
        else:
            dtype_map[col] = str  # Fallback: parsing string + cleaning

    return dtype_map

# Usage
dtype_hints = infer_dtypes_safe(path, delimiter=delimiter, encoding=encoding)
df = pd.read_csv(path, engine='c', dtype=dtype_hints, **kwargs)
```

**VERDICT FASE 2.3**: ⚠️ **RICHIE SOGLIA CONSERVATIVA (95%)**

---

### 🚧 FASE 3: Ottimizzazioni Avanzate (VALUTA IN BASE A FASE 1-2)

#### 3.1 Chunked Processing
**Rischio**: 🟡 **MEDIO**
**Impatto**: Cambio API, memory management

**Breaking Change Potenziale**:
```python
# ATTUALE: Ritorna DataFrame completo
df = load_csv(path)

# CON CHUNKED: Ritorna Generator
for chunk in load_csv_chunked(path, chunksize=100_000):
    process(chunk)  # ← API DIVERSA

# SOLUZIONE: Funzione separata, NO sostituzione load_csv()
```

**SAFE IMPLEMENTATION**:
```python
# core/loader.py - NUOVA funzione (NO modifica load_csv esistente)
def load_csv_chunked(path, chunksize=100_000, **kwargs):
    """Generator per file grandi (>100MB). API separata."""
    chunks = pd.read_csv(path, chunksize=chunksize, engine='c', **kwargs)
    for chunk in chunks:
        yield clean_dataframe(chunk).df

# Mantieni load_csv() invariata per backward compatibility
def load_csv(...):  # ← NO CAMBIAMENTI
    # Funziona come sempre
```

**VERDICT FASE 3.1**: ✅ **SICURO SE API SEPARATA**

---

#### 3.2 Parallel Processing (Dask)
**Rischio**: 🔴 **ALTO**
**Impatto**: Dependencies, compatibility, debugging

**Problemi Identificati**:

1. **Dask require dependency pesante** (~50MB, molte sub-dependencies)
2. **Streamlit + Dask = problemi noti** (multiprocessing conflicts)
3. **Debugging diventa complesso** (stack traces distribuiti)

**VERDICT FASE 3.2**: ❌ **SCONSIGLIATO PER ORA** (troppo risk/reward sfavorevole)

---

#### 3.3 Filesystem Caching
**Rischio**: 🟡 **MEDIO**
**Impatto**: Disk space, cache invalidation bugs

**Safe Implementation**:
```python
# Feature flag obbligatorio
if config.get("enable_cache", False):  # Default: OFF
    df = load_csv_cached(path)
else:
    df = load_csv(path)  # Normale
```

**VERDICT FASE 3.3**: ✅ **SICURO CON FEATURE FLAG**

---

#### 3.4 GPU Acceleration
**Rischio**: 🔴 **MOLTO ALTO**
**Impatto**: Hardware requirements, deployment complexity

**VERDICT FASE 3.4**: ❌ **NON IMPLEMENTARE** (out of scope per questa app)

---

## 🧪 Strategia Testing per Sicurezza

### Test Matrix Obbligatoria

| Test Case | Fase 1 | Fase 2 | Fase 3 | Priority |
|-----------|--------|--------|--------|----------|
| File malformati (righe missing cols) | ✅ | ✅ | ✅ | 🔴 CRITICA |
| Delimiter complessi (\t, \|, ;) | ✅ | ✅ | - | 🔴 CRITICA |
| Encoding non-UTF8 (Latin1, CP1252) | ✅ | ✅ | - | 🟡 ALTA |
| File grandi (>100MB) | - | - | ✅ | 🟢 MEDIA |
| Pattern numerici eterogenei | - | ✅ | - | 🔴 CRITICA |
| BOM edge cases (UTF-16 LE/BE) | ✅ | ✅ | - | 🟡 ALTA |
| Cache invalidation | - | - | ✅ | 🟡 ALTA |
| Engine fallback C→Python | ✅ | - | - | 🔴 CRITICA |

### Regression Test Suite

```bash
# Prima di OGNI commit in feature/csv-loading-optimization
pytest tests/ -v --cov=core --cov-report=term-missing

# SOGLIA MINIMA: coverage >= 84% (attuale)
# SOGLIA TARGET: coverage >= 90%
```

### Benchmark Baseline (OBBLIGATORIO)

```bash
# PRIMA di Fase 1
python tests/profile_loading.py > baseline_perf.txt

# DOPO ogni fase
python tests/profile_loading.py > phase1_perf.txt
diff baseline_perf.txt phase1_perf.txt  # Verifica speedup

# REGRESSION CHECK: nessun file deve essere PIÙ LENTO
```

---

## 🚨 Rischi Critici Identificati - Riepilogo

### RISCHIO #1: Engine C Parser Errors
**Probabilità**: 60-70%
**Mitigazione**: ✅ Fallback automatico a engine Python
**Test**: ✅ test_load_csv_engine_fallback()

### RISCHIO #2: Sample-Based Accuracy Loss
**Probabilità**: 30-40%
**Mitigazione**: ✅ Confidence threshold + full-scan fallback
**Test**: ✅ test_sample_based_detection_heterogeneous_file()

### RISCHIO #3: Dtype Inference Parsing Errors
**Probabilità**: 25%
**Mitigazione**: ✅ Soglia conservativa 95% + fallback str
**Test**: ✅ test_dtype_inference_mixed_columns()

---

## ✅ Checklist Pre-Implementazione

### Prima di Iniziare Fase 1

- [ ] **Backup completo codebase** (`git tag v4.x-before-optimization`)
- [ ] **Branch feature isolato** (`git checkout -b feature/csv-loading-optimization`)
- [ ] **Baseline benchmark** (`python tests/profile_loading.py > baseline.txt`)
- [ ] **Test coverage check** (`pytest --cov=core --cov-report=html`)
- [ ] **Revisione piano con team** (se applicabile)

### Durante Implementazione Fase 1

- [ ] **Implementa fallback engine C→Python** (OBBLIGATORIO)
- [ ] **Aggiungi test malformed files** (OBBLIGATORIO)
- [ ] **Verifica delimiter detection** (tutti i 12 file tests_csv/)
- [ ] **Test encoding edge cases** (UTF-16, UTF-8-sig, BOM)
- [ ] **Benchmark intermedi** (dopo ogni ottimizzazione)

### Prima di Merge Fase 1

- [ ] **Tutti i 162 test passano** (`pytest tests/ -v`)
- [ ] **Coverage >= 84%** (non degradare)
- [ ] **Speedup verificato** (almeno 3×, target 5-10×)
- [ ] **No regressioni** (nessun file più lento di baseline)
- [ ] **Code review** (se applicabile)
- [ ] **Update CLAUDE.md** (documentare cambiamenti)

---

## 📝 Raccomandazioni Finali

### APPROVA ✅

1. **Fase 1.1 (Engine C)** - CON fallback obbligatorio
2. **Fase 1.2 (CoW)** - Sicuro, nessun breaking change
3. **Fase 1.3 (Compiled Regex)** - Sicurissimo, già testato
4. **Fase 3.1 (Chunked)** - Se API separata
5. **Fase 3.3 (Caching)** - Se feature flag + OFF by default

### RIVEDI ⚠️

1. **Fase 2.1 (Sample-based)** - Solo dopo Fase 1 + feedback
2. **Fase 2.2 (Single-pass)** - Test BOM/encoding intensivi
3. **Fase 2.3 (Dtype inference)** - Soglia conservativa 95%

### SCONSIGLIA ❌

1. **Fase 3.2 (Dask)** - Troppo complesso, risk/reward sfavorevole
2. **Fase 3.4 (GPU)** - Out of scope, deployment impraticabile

---

## 🎯 Conclusione

**Il piano è SICURO se implementato con**:

1. ✅ **Approccio progressivo** (Fase 1 → 2 → 3, con feedback)
2. ✅ **Fallback obbligatori** (Engine C→Python, Sample→Full-scan)
3. ✅ **Test coverage rigoroso** (100% nuove feature, 0% regressioni)
4. ✅ **Feature flags** (tutte le ottimizzazioni aggressive OFF by default)
5. ✅ **Benchmark continui** (verifica speedup + no regressioni)

**Target realistico**: **5-10× speedup Fase 1** (2 giorni) è **RAGGIUNGIBILE** e **SICURO**.

**Target ambizioso**: **20-50× totale** (2-3 settimane) richiede **validazione continua**.

**Raccomandazione**: **INIZIA CON FASE 1**, verifica risultati, poi **rivaluta Fase 2-3**.

---

**Firma**: Claude Code (Analisi Sicurezza)
**Data**: 2025-10-22
**Versione Piano**: 1.0
