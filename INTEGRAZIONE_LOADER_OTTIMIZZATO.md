# Integrazione Loader Ottimizzato - Completata

**Data**: 27 Novembre 2025
**Status**: ✅ Integrato e pronto per test

---

## 🎯 Modifiche Applicate

### 1. config.json

Aggiunte nuove opzioni nella sezione `performance`:

```json
{
  "performance": {
    "optimize_dtypes": true,
    "aggressive_dtype_optimization": false,
    "use_optimized_loader": true,           // ← NUOVO: abilita loader ottimizzato
    "chunked_loading_threshold_mb": 600,    // ← NUOVO: soglia MB per chunked loading
    "rows_threshold": 100000,               // ← NUOVO: soglia righe per chunked loading
    "chunk_size": 50000,                    // ← NUOVO: righe per chunk
    "sample_size": 50000                    // ← NUOVO: righe per sampling
  }
}
```

**Opzioni disponibili**:

- `use_optimized_loader`: `true` = usa loader_optimized.py, `false` = usa loader.py legacy
- `chunked_loading_threshold_mb`: File > N MB usano chunked loading (default: 50)
- `rows_threshold`: File con > N righe stimate usano chunked loading (default: 100000)
- `chunk_size`: Numero di righe per chunk (default: 50000)
- `sample_size`: Numero di righe da campionare in sampling mode (default: 50000)

**Nota**: Il loader usa chunked loading se **ALMENO UNA** delle condizioni è vera:
- File size > `chunked_loading_threshold_mb` MB
- Righe stimate > `rows_threshold`

### 2. web_app.py

**Import condizionale** basato su config.json:

```python
# Legge config e decide quale loader usare
_use_optimized = config.get("performance", {}).get("use_optimized_loader", True)

if _use_optimized:
    from core.loader_optimized import load_csv  # Loader ottimizzato
    LOADER_TYPE = "optimized"
else:
    from core.loader import load_csv  # Loader legacy
    LOADER_TYPE = "legacy"
```

**Badge nel footer** per mostrare quale loader è attivo:

```python
# In fondo alla pagina
loader_emoji = "🚀" if LOADER_TYPE == "optimized" else "📦"
loader_desc = "Ottimizzato (chunked)" if LOADER_TYPE == "optimized" else "Standard"
st.caption(f"{loader_emoji} Loader: {loader_desc}")
```

---

## 🚀 Come Funziona

### Comportamento con Loader Ottimizzato Abilitato

1. **File Piccoli (< 50 MB)**:
   - Usa loader legacy (più veloce per file piccoli)
   - Comportamento identico a prima

2. **File Grandi (> 50 MB o > 100k righe)**:
   - Usa caricamento chunked
   - Riduzione RAM del 45%
   - Overhead tempo ~4%

### Auto-Detection

Il loader ottimizzato sceglie automaticamente la strategia:

```
File uploaded
    ↓
Dimensione < 50 MB?
    ↓ YES
[Loader Legacy] → Fast load
    ↓ NO
[Chunked Loader] → Low RAM load
```

---

## 🧪 Come Testare

### Test 1: File Piccolo

```bash
# Crea CSV piccolo (dovrebbe usare loader legacy anche con optimized attivo)
python -c "
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': range(1000), 'B': np.random.rand(1000)})
df.to_csv('test_small.csv', index=False)
"

# Avvia app
streamlit run web_app.py

# Upload test_small.csv
# Footer dovrebbe mostrare: 🚀 Loader: Ottimizzato (chunked)
# Ma internamente usa loader legacy (auto-detection)
```

### Test 2: File Grande

```bash
# Usa uno dei CSV generati dal benchmark
# es. tests_csv/bench_large_500k.csv (170 MB)

streamlit run web_app.py

# Upload bench_large_500k.csv
# Dovrebbe usare chunked loading (auto-detection)
# RAM usage monitorabile con Task Manager
```

### Test 3: Disabilitare Loader Ottimizzato

Modifica `config.json`:

```json
{
  "performance": {
    "use_optimized_loader": false  // ← Cambia a false
  }
}
```

Riavvia app:
```bash
streamlit run web_app.py
```

Footer dovrebbe mostrare: `📦 Loader: Standard`

---

## 📊 Risultati Attesi

### Performance con File Grande (500k righe, 170 MB)

| Metrica | Loader Legacy | Loader Ottimizzato | Miglioramento |
|---------|---------------|-------------------|---------------|
| **RAM Picco** | 1626 MB | 890 MB | **-45%** |
| **Tempo** | 190s | 198s | -4% |
| **Crash OOM** | Possibile (RAM bassa) | No | ✅ |

### Comportamento Utente

**Prima** (loader legacy):
- File > 200 MB → spesso crash OOM
- File > RAM disponibile → impossibile caricare

**Dopo** (loader ottimizzato):
- File > 200 MB → caricamento lento ma funziona
- File > RAM disponibile → caricamento chunked, nessun limite teorico
- File < 50 MB → comportamento identico (usa legacy internamente)

---

## 🔧 Troubleshooting

### La configurazione non viene applicata

**Problema**: Modifichi `chunked_loading_threshold_mb` ma il loader usa ancora la soglia vecchia.

**Soluzione**:
1. **Riavvia Streamlit** - La configurazione viene letta al module load (una sola volta)
   ```bash
   # Ferma Streamlit (Ctrl+C)
   # Riavvia
   streamlit run web_app.py
   ```

2. **Verifica configurazione caricata** - Controlla il log:
   ```bash
   tail -f logs/analizzatore_YYYYMMDD.log | grep "Loader optimized config"
   ```

   Dovresti vedere:
   ```
   Loader optimized config: SIZE_THRESHOLD=600 MB, ROWS_THRESHOLD=100000, ...
   ```

3. **Test da Python**:
   ```bash
   python -c "from core.loader_optimized import SIZE_THRESHOLD_MB; print(SIZE_THRESHOLD_MB)"
   # Output atteso: 600 (se hai configurato 600 in config.json)
   ```

### Il loader ottimizzato non viene usato

**Verifica**:
1. `config.json` contiene `"use_optimized_loader": true`
2. Riavvia Streamlit dopo modifiche a config.json
3. Controlla footer: deve mostrare `🚀 Loader: Ottimizzato`
4. Controlla log per: `Loader optimized config: SIZE_THRESHOLD=...`

### File grandi ancora lenti

**Tuning**: Modifica `config.json`

```json
{
  "performance": {
    "chunk_size": 100000,  // Chunk più grandi = più veloce (ma più RAM)
    "chunked_loading_threshold_mb": 100  // Usa chunked solo per file molto grandi
  }
}
```

### Errori con loader ottimizzato

**Fallback a loader legacy**:

```json
{
  "performance": {
    "use_optimized_loader": false  // Torna al loader originale
  }
}
```

Segnala l'errore con:
- File CSV problematico (se possibile)
- Log da `logs/analizzatore_YYYYMMDD.log`

---

## 📈 Prossimi Miglioramenti (Opzionali)

### 1. Progress Bar in Streamlit

Il loader ottimizzato supporta già progress callback, basta integrarlo:

```python
# In web_app.py, nella funzione di caricamento:
from core.loader_optimized import LoadProgress

progress_bar = st.progress(0, text="Caricamento CSV...")

def update_progress(prog: LoadProgress):
    if prog.total > 0:
        pct = prog.current / prog.total
        progress_bar.progress(pct, text=prog.message)

df = load_csv(
    file_path,
    progress_callback=update_progress  # ← Aggiungi callback
)

progress_bar.empty()  # Rimuovi dopo caricamento
```

### 2. Sampling per Preview Rapide

Per file enormi (> 1M righe), offrire preview con sampling:

```python
from core.loader_optimized import load_csv_sampled

if file_size_mb > 500:  # File molto grande
    st.warning(f"File molto grande ({file_size_mb:.0f} MB). Carico preview di 50k righe...")

    df_preview, metadata = load_csv_sampled(
        file_path,
        sample_size=50_000
    )

    st.info(f"Preview: {metadata['sampled_rows']:,} righe da {metadata['total_rows']:,} totali")
    # Mostra preview...

    if st.button("Carica file completo"):
        df_full = load_csv(file_path)  # Carica tutto
```

### 3. Cache Parquet per Ricaricamenti Rapidi

Salvare CSV puliti in formato Parquet per ricaricamento istantaneo:

```python
import hashlib

# Hash del file CSV
file_hash = hashlib.md5(file_content).hexdigest()
cache_path = f".cache/{file_hash}.parquet"

if os.path.exists(cache_path):
    # Ricarica da cache (10-100× più veloce)
    df = pd.read_parquet(cache_path)
    st.success("Caricato da cache!")
else:
    # Prima volta: CSV → DataFrame → Parquet
    df = load_csv(file_path)
    df.to_parquet(cache_path)
```

---

## 📝 Note Implementative

### Compatibilità

✅ **Drop-in replacement**: API identica a `loader.py`
✅ **Nessuna modifica al resto del codice**: Solo cambio import
✅ **Configurabile**: Flag `use_optimized_loader` per facile rollback
✅ **Test suite**: 27 test passati, 90.71% coverage

### Limitazioni Note

⚠️ **Overhead per file medi**: File 50-100k righe hanno ~5% overhead
⚠️ **Concatenazione finale**: Può richiedere ~0.5s per file con molti chunks
⚠️ **No parallelizzazione**: Pulizia chunks è sequenziale (possibile miglioramento futuro)

### Design Decisions

**Perché import condizionale e non parametro?**
- Import time decision = più efficiente
- Evita if/else ad ogni chiamata load_csv()
- Facilita A/B testing (cambia config, riavvia)

**Perché default `true`?**
- Benefici superano svantaggi per la maggior parte degli utenti
- Fallback automatico a legacy per file piccoli
- Facile disabilitare se problemi

---

## ✅ Checklist Pre-Produzione

- [x] Loader ottimizzato implementato e testato
- [x] Test suite completa (27 test, 100% pass)
- [x] Benchmark comparativi eseguiti
- [x] Documentazione completa scritta
- [x] Integrazione in web_app.py completata
- [x] Config.json aggiornato con nuove opzioni
- [x] CLAUDE.md aggiornato
- [ ] Test manuali su CSV reali (da fare)
- [ ] Monitoraggio RAM in produzione (da fare)
- [ ] Feedback utenti (da raccogliere)

---

## 📚 Riferimenti

- **Documentazione tecnica**: [docs/OTTIMIZZAZIONE_CARICAMENTO_CSV.md](docs/OTTIMIZZAZIONE_CARICAMENTO_CSV.md)
- **Riepilogo completo**: [RIEPILOGO_OTTIMIZZAZIONE.md](RIEPILOGO_OTTIMIZZAZIONE.md)
- **Test suite**: [tests/test_loader_optimized.py](tests/test_loader_optimized.py)
- **Architettura**: [CLAUDE.md](CLAUDE.md) (sezioni "CSV Processing Pipeline")

---

**Status**: ✅ Pronto per testing utente
**Raccomandazione**: Testare con CSV reali prima di deploy in produzione
**Rollback**: Cambia `use_optimized_loader: false` in config.json
