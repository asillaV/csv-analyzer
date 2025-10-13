# Test Suite - Analizzatore CSV

Test suite completa per il progetto Analizzatore CSV, creata per la issue #40.

## Esecuzione Test

### Test completi con coverage
```bash
# Attiva ambiente virtuale
.venv\Scripts\activate

# Esegui tutti i test con coverage
pytest tests/ -v --cov=core --cov-report=term-missing

# Coverage HTML (visualizzazione dettagliata)
pytest tests/ --cov=core --cov-report=html
# Apri htmlcov/index.html nel browser
```

### Test specifici
```bash
# Test per modulo specifico
pytest tests/test_csv_cleaner.py -v

# Test singolo
pytest tests/test_signal_tools.py::TestResolveFs::test_manual_fs_priority -v

# Test con markers
pytest -m unit -v
pytest -m integration -v
```

## Struttura Test

```
tests/
├── __init__.py                     # Package marker
├── conftest.py                     # Fixtures condivise
├── fixtures/
│   └── synthetic_signals.py        # Generatori segnali sintetici
├── test_analyzer.py                # Test core/analyzer.py
├── test_csv_cleaner.py             # Test core/csv_cleaner.py
└── test_signal_tools.py            # Test core/signal_tools.py
```

## Fixtures Disponibili

### Segnali Sintetici
- `generate_sine_wave()` - Onda sinusoidale pura con metriche teoriche
- `generate_ramp()` - Segnale a rampa lineare
- `generate_white_noise()` - Rumore bianco gaussiano
- `generate_noisy_sine()` - Sinusoide con rumore additivo
- `generate_step_function()` - Funzione a gradini

### CSV Temporanei
- `temp_csv_path` - File CSV temporaneo
- `sample_csv_basic` - CSV base numerico
- `sample_csv_european_format` - Formato europeo (virgola decimale)
- `sample_csv_us_format` - Formato US (punto decimale)

## Coverage Target

**Target: 80%+ su core/**

Moduli prioritari:
- ✅ `core/signal_tools.py` - Filtri e FFT
- ✅ `core/csv_cleaner.py` - Pulizia dati numerici
- ✅ `core/analyzer.py` - Rilevamento metadati CSV
- ⏳ `core/loader.py` - Caricamento DataFrame
- ⏳ `core/report_manager.py` - Generazione report

## CI/CD

I test vengono eseguiti automaticamente su GitHub Actions:
- Python 3.10, 3.11, 3.12
- Ubuntu e Windows
- Coverage report su Codecov

## Note Sviluppo

### Test con Segnali Sintetici
I test usano segnali generati matematicamente con metriche note per validare:
- Correttezza filtri (MA, Butterworth)
- Accuratezza FFT
- Gestione frequenza campionamento

### Tolleranze
Test numerici usano `pytest.approx()` con tolleranze ragionevoli:
- Frequenze FFT: ±0.5 Hz
- Valori numerici: rel=0.01 (1%)

### Markers
- `@pytest.mark.slow` - Test lenti (>1 secondo)
- `@pytest.mark.integration` - Test integrazione
- `@pytest.mark.unit` - Test unitari
- `@pytest.mark.synthetic` - Test con segnali sintetici
