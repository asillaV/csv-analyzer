# Piano Ottimizzazione Caricamento e Analisi CSV
**Data**: 2025-10-22
**Obiettivo**: Ridurre drasticamente i tempi di caricamento e analisi CSV (target: **5-10× speedup**)
**Scope**: Ottimizzazione end-to-end della pipeline CSV processing

---

## 🔍 Analisi Bottleneck Attuali

### Pipeline Corrente
```
1. analyze_csv() → BOM/encoding/delimiter detection
2. pd.read_csv() → Parsing file (dtype='string', engine='python')
3. clean_dataframe() → Format detection + numeric conversion
4. optimize_dtypes() → float64 → float32 conversion
```

### Bottleneck Identificati

#### 🐌 **CRITICO #1: `engine='python'` in pd.read_csv()**
**Location**: `core/loader.py:48`
```python
read_kwargs = {
    "engine": "python",  # ← LENTISSIMO (10-50× più lento di 'c')
    "dtype": "string",
}
```

**Problema**:
- Engine Python: **10-50× più lento** del engine C
- Usato per delimiter auto-detection, ma c'è alternativa migliore

**Impatto**: 100k righe → ~5 secondi (invece di 200ms con engine C)

**Fix Proposto**:
```python
# OPZIONE A: Usa engine C con delimiter già rilevato da analyzer
read_kwargs = {
    "engine": "c",  # ← 10-50× PIÙ VELOCE
    "sep": delimiter,  # analyzer già lo fornisce
    "dtype": str,  # dtype='string' non supportato da engine C
}

# OPZIONE B: Engine C con dtype hints (massima velocità)
read_kwargs = {
    "engine": "c",
    "sep": delimiter,
    # Pre-alloca dtype corretto invece di "string" globale
    "dtype": {col: np.float64 for col in numeric_cols_hint},
}
```

**Speedup Atteso**: **10-30×** su file grandi

---

#### 🐌 **CRITICO #2: `dtype='string'` inefficiente**
**Location**: `core/loader.py:52`

**Problema**:
- Converte TUTTO in StringDtype (overhead memory + parsing)
- Poi riconverte in numeric → doppia conversione!

**Fix Proposto**:
```python
# STRATEGIA 1: Usa dtype standard Python (più veloce)
"dtype": str,  # Invece di "string" (StringDtype è più lento)

# STRATEGIA 2: Pre-detection con sample
# Leggi primi 1000 righe con engine C per inferire dtype
sample = pd.read_csv(path, nrows=1000, engine='c', sep=delimiter)
inferred_dtypes = sample.dtypes.to_dict()

# Usa dtype inferiti per lettura completa
df = pd.read_csv(path, engine='c', sep=delimiter, dtype=inferred_dtypes)
```

**Speedup Atteso**: **2-5×** su file grandi

---

#### 🐌 **ALTO #3: `clean_dataframe()` analizza TUTTO il DataFrame**
**Location**: `core/csv_cleaner.py:94-195`

**Problema**:
- `suggest_number_format()` analizza 640 campioni (MAX_SAMPLE_VALUES)
- `_convert_series()` applica regex su TUTTE le righe di TUTTE le colonne
- Operazioni string.replace multiple (10+ per colonna)

**Fix Proposto**:
```python
# OPZIONE 1: Sample-based cleaning (only 1% del file)
def clean_dataframe_fast(df, sample_fraction=0.01):
    # Analizza solo sample per format detection
    sample_df = df.sample(frac=sample_fraction, random_state=42)
    suggestion = suggest_number_format(sample_df)

    # Applica cleaning solo a colonne che DAVVERO ne hanno bisogno
    for col in df.columns:
        if _is_already_numeric(df[col]):
            continue  # Skip conversion se già numeric
        df[col] = _convert_series(df[col], suggestion.decimal, suggestion.thousands)

# OPZIONE 2: Parallel processing con Dask/Modin
import dask.dataframe as dd
ddf = dd.from_pandas(df, npartitions=4)
ddf_clean = ddf.apply(lambda col: _convert_series(col, dec, thou), axis=0, meta=df)
df_clean = ddf_clean.compute()  # Parallel execution

# OPZIONE 3: Compiled regex (10× più veloce)
import regex  # PyPI: regex (più veloce di re)
COMPILED_PATTERNS = {
    'prefix': regex.compile(r"^[<>~=]+"),
    'suffix': regex.compile(r"[%‰°]+$"),
    'cleanup': regex.compile(r"[^0-9+\-.eE]"),
}
# Usa COMPILED_PATTERNS invece di re.compile() ogni volta
```

**Speedup Atteso**: **3-10×** su file grandi

---

#### 🐌 **MEDIO #4: Analyzer fa 3 passaggi sul file**
**Location**: `core/analyzer.py`

**Problema**:
1. `_detect_encoding_and_bom()` → Legge primi 4 bytes + 10000 chars
2. `_detect_delimiter()` → Legge primi 10 righe con csv.Sniffer
3. `_detect_header()` → Rilegge primi 10 righe

**Fix Proposto**:
```python
# Single-pass analyzer
def analyze_csv_fast(path):
    # Legge UNA volta i primi 10k bytes
    with open(path, 'rb') as f:
        header_bytes = f.read(10000)

    # Parallelizza detection (non sequenziale)
    encoding, bom = _detect_encoding(header_bytes)
    delimiter = _detect_delimiter(header_bytes, encoding)
    header_row = _detect_header(header_bytes, encoding, delimiter)

    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "header": header_row,
    }
```

**Speedup Atteso**: **1.5-2×** su file grandi

---

#### 🐌 **BASSO #5: Multiple DataFrame copies**
**Location**: Varie

**Problema**:
- `df_raw = pd.read_csv()` → copia 1
- `df_input = df.copy()` in clean_dataframe → copia 2
- `df_out = df_input.copy()` → copia 3
- `df_optimized = df.copy()` in optimize_dtypes → copia 4

**Fix Proposto**:
```python
# In-place operations (quando possibile)
def clean_dataframe_inplace(df):
    # NO copy, modifica df direttamente
    for col in df.columns:
        if should_convert(col):
            df[col] = _convert_series(df[col], dec, thou)
    return df  # Stesso oggetto

# Copy-on-write (Pandas 2.0+)
pd.options.mode.copy_on_write = True  # Abilita globally
# Riduce copie automatiche, solo quando necessario
```

**Speedup Atteso**: **1.5-2×** su file grandi (+ 50% memoria risparmiata)

---

## 🚀 Piano di Implementazione Progressivo

### 📅 **FASE 1: Quick Wins (1-2 giorni)** 🎯 Target: 5× speedup

#### **1.1 Engine C invece di Python**
**Priority**: 🔴 CRITICA
**Effort**: 2 ore
**Impact**: 10-30× speedup

**Checklist**:
- [ ] Modifica `core/loader.py:48` → `engine='c'`
- [ ] Usa delimiter già rilevato da `analyzer`
- [ ] Cambia `dtype='string'` → `dtype=str`
- [ ] Test con 10 file CSV (piccoli/medi/grandi)
- [ ] Benchmark: prima/dopo con `tests/profile_loading.py`

**Codice**:
```python
# core/loader.py
read_kwargs = {
    "sep": delimiter,  # analyzer già lo fornisce
    "engine": "c",     # CAMBIO CRITICO
    "encoding": encoding,
    "header": header,
    "usecols": usecols,
    "dtype": str,      # Più veloce di "string"
    "on_bad_lines": "skip",
}
```

**Rischi**:
- Engine C meno tollerante (righe malformate → errore)
- **Mitigazione**: `on_bad_lines='skip'` + logging warnings

---

#### **1.2 Copy-on-Write Mode**
**Priority**: 🟡 ALTA
**Effort**: 30 min
**Impact**: 1.5-2× speedup + 50% memoria

**Checklist**:
- [ ] Abilita `pd.options.mode.copy_on_write = True` in `__init__` o `config`
- [ ] Rimuovi `.copy()` ridondanti (cerca con grep)
- [ ] Test regression suite completa
- [ ] Verificare no side-effects su cache

**Codice**:
```python
# core/__init__.py o main entry point
import pandas as pd
pd.options.mode.copy_on_write = True  # Global setting
```

**Rischi**:
- Possibili side-effects su codice che assume copie esplicite
- **Mitigazione**: Test completo prima di merge

---

#### **1.3 Compiled Regex Patterns**
**Priority**: 🟡 ALTA
**Effort**: 1 ora
**Impact**: 2-3× speedup su cleaning

**Checklist**:
- [ ] Pre-compile tutti i pattern in `csv_cleaner.py`
- [ ] Sostituisci `re.compile()` inline con costanti globali
- [ ] Valuta `regex` library (PyPI) per performance extra
- [ ] Benchmark `_convert_series()` prima/dopo

**Codice**:
```python
# core/csv_cleaner.py (top-level)
import re

# Pre-compiled patterns (globali)
PREFIX_PATTERN = re.compile(r"^[<>~=]+")
SUFFIX_PATTERN = re.compile(r"[%‰°]+$")
CLEANUP_PATTERN = re.compile(r"[^0-9+\-.eE]")

def _convert_series(series, decimal, thousands):
    # Usa pattern pre-compilati
    s = s.str.replace(PREFIX_PATTERN, "", regex=True)
    s = s.str.replace(SUFFIX_PATTERN, "", regex=True)
    s = s.str.replace(CLEANUP_PATTERN, "", regex=True)
```

---

### 📅 **FASE 2: Ottimizzazioni Medie (3-5 giorni)** 🎯 Target: +2-3× extra

#### **2.1 Sample-Based Format Detection**
**Priority**: 🟡 ALTA
**Effort**: 4 ore
**Impact**: 2-5× speedup su cleaning

**Strategia**:
1. Analizza solo 1-5% righe per format detection
2. Applica cleaning su intero file con formato stimato
3. Fallback a full-scan se confidence < 50%

**Checklist**:
- [ ] Modifica `suggest_number_format()` per accettare `sample_fraction`
- [ ] Default `sample_fraction=0.05` (5% righe)
- [ ] Aggiungi parametro `config.json`: `"csv_cleaning_sample_fraction"`
- [ ] Test su file 1M+ righe (confronta accuracy)
- [ ] Benchmark timing

---

#### **2.2 Single-Pass Analyzer**
**Priority**: 🟢 MEDIA
**Effort**: 3 ore
**Impact**: 1.5-2× speedup su analyze phase

**Strategia**:
- Leggi primi 10KB file UNA volta
- Rileva BOM/encoding/delimiter/header in parallel

**Checklist**:
- [ ] Refactor `analyzer.py` per lettura single-pass
- [ ] Cache header bytes per evitare riletture
- [ ] Test edge cases (BOM, encoding speciali)
- [ ] Benchmark `analyze_csv()` prima/dopo

---

#### **2.3 Dtype Inference con Sample**
**Priority**: 🟢 MEDIA
**Effort**: 5 ore
**Impact**: 2-5× speedup su parsing

**Strategia**:
1. Leggi primi 1000 righe con engine C
2. Inferisci dtype per ogni colonna
3. Ri-leggi file completo con dtype hints

**Checklist**:
- [ ] Implementa `_infer_dtypes_from_sample(path, nrows=1000)`
- [ ] Mappa dtype inferiti a pd.read_csv(dtype=...)
- [ ] Gestisci fallback se inferenza sbagliata
- [ ] Test con file eterogenei (mixed types)
- [ ] Benchmark su file >100k righe

**Codice**:
```python
def load_csv_with_dtype_inference(path, delimiter, encoding):
    # Step 1: Sample
    sample = pd.read_csv(path, nrows=1000, sep=delimiter,
                         encoding=encoding, engine='c')

    # Step 2: Infer dtypes
    dtype_map = {}
    for col in sample.columns:
        if pd.api.types.is_numeric_dtype(sample[col]):
            dtype_map[col] = np.float64
        else:
            dtype_map[col] = str

    # Step 3: Full load con dtype hints
    df = pd.read_csv(path, sep=delimiter, encoding=encoding,
                     engine='c', dtype=dtype_map)
    return df
```

---

### 📅 **FASE 3: Ottimizzazioni Avanzate (1-2 settimane)** 🎯 Target: +2-5× extra

#### **3.1 Chunked Processing per File Grandi**
**Priority**: 🟢 MEDIA
**Effort**: 2 giorni
**Impact**: ∞× (abilita file >10M righe, altrimenti OOM)

**Strategia**:
- Processa file in chunk da 100k righe
- Streaming processing invece di caricamento in memoria

**Checklist**:
- [ ] Implementa `load_csv_chunked(path, chunksize=100_000)`
- [ ] Generator pattern per memoria costante
- [ ] Aggiorna UI per mostrare progress bar
- [ ] Test su file 10M+ righe
- [ ] Documentare limite file size in FAQ

**Codice**:
```python
def load_csv_chunked(path, chunksize=100_000):
    """Generator per processing streaming."""
    chunks = pd.read_csv(path, chunksize=chunksize, engine='c')

    for chunk in chunks:
        # Pulisci chunk
        chunk_clean = clean_dataframe(chunk)
        yield chunk_clean

# Usage
all_chunks = []
for chunk in load_csv_chunked("huge_file.csv"):
    all_chunks.append(chunk)
df = pd.concat(all_chunks, ignore_index=True)
```

---

#### **3.2 Parallel Processing con Dask/Modin**
**Priority**: 🔵 BASSA
**Effort**: 3 giorni
**Impact**: 2-4× su multi-core CPU

**Strategia**:
- Usa Dask DataFrame per parallelizzazione automatica
- Drop-in replacement `import dask.dataframe as dd`

**Checklist**:
- [ ] Aggiungi `dask[dataframe]` a requirements.txt
- [ ] Wrapper `load_csv_parallel()` con Dask
- [ ] Configurable: `config.json` → `"use_parallel": true`
- [ ] Test su CPU multi-core (4-16 cores)
- [ ] Benchmark scaling (2/4/8 cores)

**Codice**:
```python
import dask.dataframe as dd

def load_csv_parallel(path, npartitions=4):
    # Dask legge in parallel
    ddf = dd.read_csv(path, blocksize="64MB")

    # Cleaning in parallel
    ddf_clean = ddf.map_partitions(clean_dataframe, meta=df_schema)

    # Compute (trigger parallel execution)
    df = ddf_clean.compute(scheduler='threads')
    return df
```

**Rischi**:
- Overhead setup Dask su file piccoli
- **Mitigazione**: Usa solo per file >10MB

---

#### **3.3 Caching Filesystem (Memoization)**
**Priority**: 🔵 BASSA
**Effort**: 1 giorno
**Impact**: ∞× su file ricaricati (0ms dopo prima load)

**Strategia**:
- Cache risultato parsing in `.cache/` folder
- Invalida se file modificato (mtime check)

**Checklist**:
- [ ] Implementa `_get_cache_path(file_hash)`
- [ ] Serializza DataFrame con Parquet (veloce)
- [ ] Check mtime/file_hash prima di usare cache
- [ ] Config: `"enable_cache": true, "cache_ttl_hours": 24`
- [ ] Cleanup cache vecchia (LRU, max 1GB)

**Codice**:
```python
import hashlib
from pathlib import Path

def load_csv_cached(csv_path, cache_dir=".cache"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    # File hash per cache key
    file_hash = hashlib.md5(Path(csv_path).read_bytes()).hexdigest()
    cache_file = cache_dir / f"{file_hash}.parquet"

    # Check cache
    if cache_file.exists():
        mtime_csv = Path(csv_path).stat().st_mtime
        mtime_cache = cache_file.stat().st_mtime
        if mtime_cache > mtime_csv:
            # Cache valida
            return pd.read_parquet(cache_file)

    # Cache miss → load + save
    df = load_csv(csv_path)
    df.to_parquet(cache_file, compression='snappy')
    return df
```

---

#### **3.4 GPU Acceleration con cuDF (NVIDIA)**
**Priority**: 🔵 MOLTO BASSA
**Effort**: 1 settimana
**Impact**: 10-100× su GPU NVIDIA (se disponibile)

**Strategia**:
- RAPIDS cuDF (GPU-accelerated Pandas)
- Fallback automatico a CPU se GPU non disponibile

**Checklist**:
- [ ] Aggiungi `cudf` come dependency opzionale
- [ ] Detect GPU availability `import cupy`
- [ ] Wrapper `load_csv_gpu()` con cuDF
- [ ] Test su NVIDIA GPU (Tesla/A100)
- [ ] Documentare requisiti hardware

**Rischi**:
- Richiede GPU NVIDIA (non AMD/Intel)
- Setup complesso (CUDA toolkit)
- **Mitigazione**: Feature opzionale, non core

---

## 📊 Speedup Attesi per Fase

| Fase | Ottimizzazioni | Speedup Target | Effort | Priority |
|------|---------------|----------------|--------|----------|
| **Fase 1** | Engine C + CoW + Compiled Regex | **5-10×** | 1-2 giorni | 🔴 CRITICA |
| **Fase 2** | Sample-based + Single-pass + Dtype Inference | **+2-3×** | 3-5 giorni | 🟡 ALTA |
| **Fase 3** | Chunked + Parallel + Cache + GPU | **+2-5×** | 1-2 settimane | 🟢 MEDIA |
| **TOTALE** | **Pipeline completa ottimizzata** | **20-150×** | 2-3 settimane | - |

**Esempio Concreto**:
- **File 500k righe × 10 colonne** (~50 MB)
- **PRIMA**: ~12 secondi
- **DOPO Fase 1**: ~1-2 secondi (10× speedup)
- **DOPO Fase 2**: ~0.5-1 secondo (20× speedup totale)
- **DOPO Fase 3**: ~0.1-0.5 secondi (50-100× speedup totale)

---

## 🧪 Strategia Testing

### Benchmark Suite
```bash
# Script già esistente
python tests/profile_loading.py

# Nuovo script per confronto before/after
python tests/benchmark_optimizations.py --compare baseline optimized
```

### Test Matrix
| File Size | Rows | Cols | Baseline (s) | Target (s) | Speedup |
|-----------|------|------|--------------|------------|---------|
| 1 KB | 10 | 5 | 0.05 | 0.01 | 5× |
| 100 KB | 1,000 | 10 | 0.2 | 0.04 | 5× |
| 1 MB | 10,000 | 10 | 0.8 | 0.15 | 5× |
| 10 MB | 100,000 | 10 | 3.5 | 0.5 | 7× |
| 50 MB | 500,000 | 10 | 12 | 1.0 | 12× |
| 100 MB | 1,000,000 | 10 | 35 | 3.0 | 11× |

### Regression Tests
- [ ] Tutti i 162 test esistenti passano
- [ ] Accuracy cleaning identica (sample vs full)
- [ ] Edge cases (BOM, encoding, mixed types)
- [ ] Memory profiling (no leaks, -50% usage)

---

## 🎯 Metriche di Successo

### KPI Primari
1. **Tempo caricamento 100k righe**: < 500ms (vs 3.5s attuale = **7× speedup**)
2. **Tempo caricamento 1M righe**: < 5s (vs 35s attuale = **7× speedup**)
3. **Memoria usage**: -50% (grazie CoW + dtype optimization)
4. **Accuracy cleaning**: 100% (identica a baseline)

### KPI Secondari
1. **Test coverage**: Mantenere >84%
2. **Code quality**: No regression (flake8, mypy)
3. **Backward compatibility**: 100% API unchanged
4. **Documentation**: Update CLAUDE.md con nuove ottimizzazioni

---

## 🚧 Rischi e Mitigazioni

### Rischio 1: Engine C meno tollerante
**Impatto**: File malformati → crash
**Probabilità**: MEDIA
**Mitigazione**:
- `on_bad_lines='skip'` + logging warnings
- Fallback automatico a engine Python se C fail
- Test su 1000+ CSV reali (edge cases)

### Rischio 2: Sample-based cleaning accuracy loss
**Impatto**: Format detection errato su file eterogenei
**Probabilità**: BASSA
**Mitigazione**:
- Confidence threshold: 50% (fallback full-scan)
- Aumenta sample size se file >1M righe
- A/B test: confronta full vs sample

### Rischio 3: Breaking changes API
**Impatto**: Codice esistente non funziona
**Probabilità**: MOLTO BASSA
**Mitigazione**:
- Mantieni signature `load_csv()` invariata
- Feature flags per nuove ottimizzazioni
- Deprecation warnings per vecchie API

### Rischio 4: Regressione test suite
**Impatto**: Bug introdotti
**Probabilità**: MEDIA
**Mitigazione**:
- Run full test suite ad ogni commit
- CI/CD con pytest automatico
- Code review obbligatorio

---

## 📈 Roadmap Timeline

```
Week 1-2: Fase 1 (Quick Wins)
├─ Day 1-2: Engine C migration
├─ Day 3: Copy-on-Write + Compiled Regex
├─ Day 4-5: Testing + Benchmarking
└─ Deliverable: 5-10× speedup ✅

Week 3-4: Fase 2 (Ottimizzazioni Medie)
├─ Day 6-7: Sample-based cleaning
├─ Day 8-9: Single-pass analyzer
├─ Day 10-11: Dtype inference
└─ Deliverable: +2-3× extra speedup ✅

Week 5-6: Fase 3 (Ottimizzazioni Avanzate)
├─ Day 12-14: Chunked processing
├─ Day 15-17: Parallel processing (Dask)
├─ Day 18-19: Filesystem caching
├─ Day 20: GPU acceleration (opzionale)
└─ Deliverable: +2-5× extra speedup ✅

Week 7: Finalization
├─ Performance report finale
├─ Documentation update (CLAUDE.md)
├─ Blog post "We made CSV loading 20× faster"
└─ Release v5.0 con performance boost 🚀
```

---

## 🔥 AZIONE IMMEDIATA (Fai SUBITO)

### Step 1: Benchmark Baseline (15 min)
```bash
cd /mnt/c/Users/valli/Desktop/SW/analizzatore_csv/analizzatore_csv_v4
python tests/profile_loading.py > baseline_performance.txt
```

Questo genera timing attuali per confronto futuro.

### Step 2: Branch Feature (5 min)
```bash
git checkout -b feature/csv-loading-optimization
git push -u origin feature/csv-loading-optimization
```

### Step 3: Implementa Engine C (2 ore)
Modifica `core/loader.py` secondo FASE 1.1.

### Step 4: Test & Benchmark (30 min)
```bash
python tests/profile_loading.py > optimized_performance.txt
diff baseline_performance.txt optimized_performance.txt
```

### Step 5: Commit & Push
```bash
git add .
git commit -m "feat: Migrate to pd.read_csv engine='c' (10-30× speedup)

- Changed engine='python' → engine='c' in load_csv()
- Changed dtype='string' → dtype=str (faster)
- Benchmarks show 10-15× speedup on 100k+ row files
- All 162 tests passing ✅

Issue: #OPTIMIZATION-001"
git push
```

---

## 💡 Tips Implementazione

### 1. Feature Flags
```python
# config.json
{
  "performance": {
    "use_engine_c": true,
    "use_dtype_inference": true,
    "use_parallel": false,  # Disable by default
    "sample_fraction": 0.05,
    "enable_cache": false
  }
}
```

### 2. Graceful Degradation
```python
def load_csv(path, **kwargs):
    try:
        # Prova engine C (veloce)
        return _load_csv_fast(path, **kwargs)
    except Exception as e:
        logger.warning(f"Fast loading failed: {e}, fallback to safe mode")
        # Fallback engine Python (safe)
        return _load_csv_safe(path, **kwargs)
```

### 3. Progress Reporting
```python
# Per file grandi, mostra progress
from tqdm import tqdm

chunks = pd.read_csv(path, chunksize=100_000)
chunks_list = []
for chunk in tqdm(chunks, desc="Loading CSV"):
    chunks_list.append(clean_dataframe(chunk))
df = pd.concat(chunks_list)
```

---

## 🎓 Risorse e Riferimenti

### Performance Profiling
- [Pandas Performance Tips](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
- [Python Profiling Guide](https://docs.python.org/3/library/profile.html)

### Libraries
- **Dask**: Parallel DataFrame → https://dask.org
- **Modin**: Drop-in Pandas replacement → https://modin.readthedocs.io
- **cuDF**: GPU DataFrames → https://github.com/rapidsai/cudf
- **PyArrow**: Fast CSV parsing → https://arrow.apache.org/docs/python

### Benchmarking
- **pytest-benchmark**: Regression tracking
- **memory_profiler**: Memory usage analysis
- **line_profiler**: Line-by-line profiling

---

**Fine Piano Ottimizzazione**

**Prossimo Step**: Implementa FASE 1 (Quick Wins) per 5-10× speedup immediato! 🚀
