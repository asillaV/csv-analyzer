# Remediation Roadmap — Performance & Security Review

**Creato**: 2025-10-16
**Repo**: asillaV/csv-analyzer
**Review completa**: `docs/agents/perf_sec_review_web_app.md`

---

## 📊 Overview Issue

| Milestone | Issue Count | Stima totale | Priorità |
|-----------|-------------|--------------|----------|
| 🔴 Security Hardening (Week 1) | 3 | 4h | Bloccante/Alta |
| 🟠 Performance Optimization (Sprint N+1) | 5 | 10h | Alta/Media |
| 🔵 Technical Debt (Backlog) | 2 | 3h | Bassa |
| **TOTALE** | **10** | **17h** | - |

---

## 🔴 Milestone 1: Security Hardening (Week 1)

**Deadline suggerita**: 7 giorni da oggi
**Obiettivo**: Eliminare vulnerabilità bloccanti (XSS, DoS, data leakage)

### Issue #47: HTML Injection (XSS) 🔴 BLOCCANTE
- **Stima**: 1h
- **Owner**: Frontend/Security team
- **Dipendenze**: Nessuna
- **Priority**: 1 (fare per primo)
- **Fix**:
  - Applica `html.escape()` su tutti i contenuti dinamici in markdown
  - Sostituisci spacer HTML con `st.write("")`
  - Audit: cerca tutti `unsafe_allow_html=True` nel codebase
- **Test**: `tests/test_web_app_security.py::test_xss_column_name`
- **Link**: https://github.com/asillaV/csv-analyzer/issues/47

### Issue #48: Upload DoS 🔴 BLOCCANTE
- **Stima**: 2h
- **Owner**: Backend/Security team
- **Dipendenze**: Nessuna
- **Priority**: 2
- **Fix**:
  - Limite file size: 100MB (configurabile)
  - Limite righe: 1M post-parse
  - Limite colonne: 500
  - Timeout parsing: 30s con `signal.alarm`
  - Temp file univoco: `tempfile.mkstemp()`
- **Test**: `test_reject_large_file()`, `test_parsing_timeout()`
- **Configurabilità**: Aggiungi `upload_limits` a `config.json`
- **Link**: https://github.com/asillaV/csv-analyzer/issues/48

### Issue #49: Cache isolation 🔴 ALTA
- **Stima**: 1h
- **Owner**: Backend team
- **Dipendenze**: Nessuna
- **Priority**: 3
- **Fix**:
  - Aggiungi `dataset_id = uuid4()` in `st.session_state`
  - Includi `session_id` in tutte le cache keys (DF, filter, FFT)
- **Test**: `test_cache_session_isolation()`
- **Link**: https://github.com/asillaV/csv-analyzer/issues/49

**Definition of Done (Milestone 1)**:
- [ ] Tutti i test di sicurezza passano
- [ ] Nessun warning da Bandit/safety check
- [ ] Smoke test: upload 200MB CSV → rejected
- [ ] Smoke test: CSV con `<script>` → escaped
- [ ] Deploy su staging e test manuale

---

## 🟠 Milestone 2: Performance Optimization (Sprint N+1)

**Deadline suggerita**: 2 settimane da inizio milestone
**Obiettivo**: Ridurre tempi rendering 70%, memoria 40%

### Issue #50: Downsampling optimization 🟠 ALTA
- **Stima**: 3h
- **Owner**: Performance team
- **Dipendenze**: Nessuna
- **Priority**: 1 (fare per primo)
- **Fix**:
  - Pre-decima DataFrame prima del loop plot
  - Applica a tutti i modal (Sovrapposto/Separati/Cascata)
- **Test**: `test_downsampling_efficiency()`
- **Benchmark atteso**: 500k × 10 cols: 12.5s → 2.1s (6× speedup)
- **Link**: https://github.com/asillaV/csv-analyzer/issues/50

### Issue #51: DataFrame copies 🟠 ALTA
- **Stima**: 2h
- **Owner**: Performance team
- **Dipendenze**: Nessuna
- **Priority**: 2
- **Fix**:
  - Aggiungi `.copy()` al cache load DF (protegge mutazioni)
  - Valuta rimozione `.copy()` da cache filter/FFT se immutabili
- **Test**: `test_cache_immutability()`
- **Link**: https://github.com/asillaV/csv-analyzer/issues/51

### Issue #52: Datetime conversions 🟡 MEDIA
- **Stima**: 2h
- **Owner**: Performance team
- **Dipendenze**: Issue #50 (stesso loop plot)
- **Priority**: 3
- **Fix**:
  - Crea `_parse_x_column_once()`
  - Pre-converti X prima loop plot
  - Modifica `_make_time_series()` per accettare x pre-processato
- **Test**: Verifica conversione singola con timer
- **Benchmark atteso**: 100k × 5 datetime: 1.2s → 0.2s (6× speedup)
- **Link**: https://github.com/asillaV/csv-analyzer/issues/52

### Issue #53: Dtype optimization 🟡 MEDIA
- **Stima**: 1h
- **Owner**: Performance team
- **Dipendenze**: Nessuna
- **Priority**: 4
- **Fix**:
  - Crea funzione `optimize_dtypes()`
  - Applica dopo `load_csv()` (linea 863)
  - Configurabile via `config.json`
- **Test**: Verifica valori invariati, dtype cambiato
- **Saving atteso**: 1M × 10 cols: 80MB → 40MB (50%)
- **Link**: https://github.com/asillaV/csv-analyzer/issues/53

### Issue #54: Logging context 🟡 MEDIA
- **Stima**: 2h
- **Owner**: DevOps/Observability team
- **Dipendenze**: Nessuna
- **Priority**: 5
- **Fix**:
  - Inizializza logger in `main()`
  - Fix exception handling: linee 877, 1686, 1794
  - User message generico, log tecnico con `exc_info=True`
- **Test**: Verifica log contiene traceback + context
- **Link**: https://github.com/asillaV/csv-analyzer/issues/54

**Definition of Done (Milestone 2)**:
- [ ] Benchmark su dataset 500k × 10 cols: <5s totale
- [ ] Peak memory <120MB (vs 214MB baseline)
- [ ] Cache hit rate >70% su rerun
- [ ] Logs strutturati con context (session_id, file_size)
- [ ] Deploy su staging, verifica metriche telemetria

---

## 🔵 Milestone 3: Technical Debt (Backlog)

**Deadline**: Quando bandwidth disponibile
**Obiettivo**: Migliorare observability e manutenibilità

### Issue #55: Cache telemetry 🔵 BASSA
- **Stima**: 2h
- **Owner**: Observability team
- **Dipendenze**: Nessuna
- **Priority**: Bassa
- **Fix**:
  - Aggiungi contatori `CACHE_STATS`
  - Instrumenta `_get_cached_filter()` e `_get_cached_fft()`
  - Log periodicamente hit rate
- **Link**: https://github.com/asillaV/csv-analyzer/issues/55

### Issue #56: Temp file cleanup 🔵 BASSA
- **Stima**: 1h
- **Owner**: Backend team
- **Dipendenze**: Issue #48 (usa tempfile.mkstemp)
- **Priority**: Bassa
- **Fix**:
  - Sostituisci `tmp_upload.csv` con `tempfile.mkstemp()`
  - Aggiungi `finally:` per cleanup garantito
- **Test**: Verifica file eliminato anche su exception
- **Link**: https://github.com/asillaV/csv-analyzer/issues/56

**Definition of Done (Milestone 3)**:
- [ ] Cache hit rate visible in logs
- [ ] No temp files leak (verifica dopo 1000 uploads)
- [ ] Documentazione aggiornata in `ARCHITECTURE.md`

---

## 📈 Metriche di Successo

### Baseline (pre-fix)
| Metrica | Valore attuale | Dataset |
|---------|----------------|---------|
| t_load_clean_ms | 4200ms | 500k × 10 |
| t_filter_total | 850ms | 5 cols |
| t_fft_total | 1200ms | 5 cols |
| t_plot_render | 8500ms | Sovrapposto |
| peak_mem_MB | 214MB | - |
| points_plotted | 2,500,000 | 500k × 5 |
| **TOTALE** | **14.75s** | - |

### Target (post-fix)
| Metrica | Target | % Miglioramento |
|---------|--------|-----------------|
| t_load_clean_ms | <5000ms | ✅ OK |
| t_filter_total | <1000ms | ✅ OK |
| t_plot_render | <3000ms | **65% ↓** |
| peak_mem_MB | <120MB | **44% ↓** |
| points_plotted | <50,000 | **98% ↓** |
| **TOTALE** | **<5s** | **66% ↓** |

### Metriche Security
- [ ] XSS test suite: 100% pass
- [ ] Upload validation: reject files >100MB
- [ ] Parsing timeout: <30s sempre
- [ ] Cache isolation: no cross-session leakage

---

## 🛠️ Workflow Remediation

### Setup
```bash
# Checkout branch per milestone
git checkout -b fix/milestone-1-security-hardening

# Installa dev dependencies (se necessario)
pip install -r requirements-dev.txt  # bandit, safety, pytest-cov
```

### Per ogni issue
1. **Assegna issue** su GitHub (assign to yourself)
2. **Crea branch** (opzionale, se preferisci feature branch per issue)
   ```bash
   git checkout -b fix/issue-47-xss
   ```
3. **Implementa fix** seguendo remediation nel report
4. **Scrivi test** (usa `tests/test_web_app_security.py` o `tests/test_web_app_performance.py`)
5. **Verifica test localmente**
   ```bash
   pytest tests/test_web_app_security.py -v
   ```
6. **Commit con riferimento issue**
   ```bash
   git add web_app.py tests/test_web_app_security.py
   git commit -m "fix: sanitize HTML in quality badge and metadata (#47)

   - Apply html.escape() to badge_text and issue_summary
   - Replace HTML spacers with st.write()
   - Add test_xss_column_name to verify escaping

   Fixes #47"
   ```
7. **Push e verifica CI** (se presente)
   ```bash
   git push origin fix/issue-47-xss
   ```
8. **Chiudi issue** (manualmente o via commit message "Fixes #47")

### Merge milestone
```bash
# Merge alla fine della milestone
git checkout main
git merge --no-ff fix/milestone-1-security-hardening
git tag v1.1.0-security-hardening
git push origin main --tags
```

---

## 📝 Note Implementazione

### Issue #48 (Upload DoS) - Nota su signal.alarm
**Problema**: `signal.alarm` funziona solo su Unix/Linux, non su Windows.

**Soluzione multi-platform**:
```python
import platform
import threading

def timeout_wrapper(func, timeout_sec, *args, **kwargs):
    """Cross-platform timeout wrapper."""
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        raise TimeoutError(f"Operation exceeded {timeout_sec}s")
    if exception[0]:
        raise exception[0]
    return result[0]

# Usage
try:
    df = timeout_wrapper(load_csv, CSV_PARSE_TIMEOUT_SEC, str(tmp_path), ...)
except TimeoutError:
    st.error("Parsing troppo lento. Verifica il formato del file.")
```

### Issue #50 (Downsampling) - Scelta metodo
**Attuale**: LTTB (Largest Triangle Three Buckets) — ottimo per preservare forma visuale.

**Alternative** (se LTTB troppo lento):
- **Uniform decimation**: `df.iloc[::factor]` — più veloce ma meno accurato
- **Min-Max**: mantiene picchi — ottimo per segnali
- **Random sampling**: `df.sample(n=10000)` — utile per scatter plot

**Benchmark**:
| Metodo | Tempo (100k→10k) | Qualità visuale |
|--------|------------------|-----------------|
| LTTB | 50ms | ⭐⭐⭐⭐⭐ |
| Uniform | 5ms | ⭐⭐⭐ |
| Min-Max | 30ms | ⭐⭐⭐⭐ |
| Random | 10ms | ⭐⭐ |

**Raccomandazione**: Mantieni LTTB, è già implementato e bilanciato.

### Issue #53 (Dtype) - Controllo overflow
Prima di convertire a float32/int32, verifica range:

```python
def safe_downcast_float(series: pd.Series) -> pd.Series:
    """Converti float64→float32 solo se safe."""
    max_val = series.abs().max()
    if pd.isna(max_val):
        return series.astype('float32')  # Tutti NaN, safe

    # float32: ±3.4e38, precision: ~7 decimali
    if max_val < 3.4e38:
        # Verifica precision loss
        original_sum = series.sum()
        converted = series.astype('float32')
        relative_error = abs(converted.sum() - original_sum) / abs(original_sum)

        if relative_error < 1e-6:  # <0.0001% error
            return converted

    return series  # Mantieni float64 se unsafe
```

---

## 🧪 Test Plan Completo

### Security Testing
```bash
# Run security test suite
pytest tests/test_web_app_security.py -v --cov=web_app --cov-report=html

# Static analysis
bandit -r web_app.py -ll  # Low confidence, Low severity
safety check  # Check dependencies vulnerabilities
```

### Performance Testing
```bash
# Baseline benchmark
python tests/profile_loading.py > baseline.txt

# Post-fix benchmark
python tests/profile_loading.py > post_fix.txt

# Compare
diff baseline.txt post_fix.txt
```

### Regression Testing
```bash
# Assicurati che tutti i test passino
pytest tests/ -v --maxfail=1

# Smoke test manuale
streamlit run web_app.py
# 1. Carica assets/sample_timeseries.csv
# 2. Applica filtro Butterworth
# 3. Genera FFT
# 4. Crea report visivo
# 5. Verifica: no crash, output corretto
```

---

## 🚀 Deploy Strategy

### Staging
1. Deploy branch `fix/milestone-1-security-hardening` su staging
2. Smoke test completo (checklist sopra)
3. Load test: 10 sessioni concorrenti × 50MB CSV
4. Monitoraggio: logs, metriche (tempo, memoria)
5. Se OK → merge a main

### Production
1. Deploy main branch
2. Canary release: 10% traffico su nuova versione
3. Monitor error rate, latency, memory usage
4. Se OK dopo 24h → 100% traffico
5. Se KO → rollback immediato

### Rollback Plan
```bash
# In caso di problemi critici in produzione
git revert <commit-hash>  # Revert merge milestone
git push origin main

# O rollback via deployment tool
kubectl rollout undo deployment/csv-analyzer  # K8s
```

---

## 📚 Riferimenti

- **Review completa**: `docs/agents/perf_sec_review_web_app.md`
- **Issue tracker**: https://github.com/asillaV/csv-analyzer/issues?q=label%3Asecurity+label%3Aperformance
- **OWASP Top 10**: https://owasp.org/www-project-top-ten/
- **Streamlit Security**: https://docs.streamlit.io/knowledge-base/deploy/authentication-best-practices

---

**Ultimo aggiornamento**: 2025-10-16
**Prossima review**: Dopo completamento Milestone 2
