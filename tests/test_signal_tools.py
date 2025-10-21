"""Test per core/signal_tools.py - filtri e FFT."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.signal_tools import (
    FilterSpec,
    FFTSpec,
    estimate_fs,
    resolve_fs,
    validate_filter_spec,
    apply_filter,
    moving_average,
    compute_fft,
    _SCIPY_OK,
)
from tests.fixtures.synthetic_signals import (
    generate_sine_wave,
    generate_ramp,
    generate_white_noise,
    generate_noisy_sine,
)


class TestEstimateFs:
    """Test per estimate_fs()."""

    def test_uniform_numeric_series(self):
        """Serie numerica uniforme."""
        x = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4])  # dt = 0.1, fs = 10 Hz
        fs = estimate_fs(x)
        assert fs is not None
        assert fs == pytest.approx(10.0, rel=0.01)

    def test_uniform_datetime_series(self):
        """Serie datetime uniforme."""
        x = pd.date_range('2025-01-01', periods=10, freq='1s')
        fs = estimate_fs(pd.Series(x))
        assert fs is not None
        assert fs == pytest.approx(1.0, rel=0.01)  # 1 Hz (1 sample/sec)

    def test_datetime_milliseconds(self):
        """Serie datetime con millisecondi."""
        x = pd.date_range('2025-01-01', periods=10, freq='100ms')
        fs = estimate_fs(pd.Series(x))
        assert fs is not None
        assert fs == pytest.approx(10.0, rel=0.01)  # 10 Hz

    def test_irregular_sampling(self):
        """Campionamento irregolare (usa mediana)."""
        x = pd.Series([0.0, 0.1, 0.3, 0.4, 0.8])  # dt variabile
        fs = estimate_fs(x)
        # Mediana dei delta dovrebbe gestire irregolarità
        assert fs is not None
        assert fs > 0

    def test_single_value(self):
        """Serie con un solo valore."""
        x = pd.Series([1.0])
        fs = estimate_fs(x)
        assert fs is None  # Non può stimare con 1 valore

    def test_none_input(self):
        """Input None."""
        fs = estimate_fs(None)
        assert fs is None

    def test_all_nan(self):
        """Serie con tutti NaN."""
        x = pd.Series([np.nan, np.nan, np.nan])
        fs = estimate_fs(x)
        assert fs is None

    def test_negative_or_zero_deltas(self):
        """Delta negativi o zero (dati invalidi)."""
        x = pd.Series([1.0, 1.0, 1.0])  # dt = 0
        fs = estimate_fs(x)
        assert fs is None


class TestResolveFs:
    """Test per resolve_fs() - Single Source of Truth per fs."""

    def test_manual_fs_priority(self):
        x = pd.Series([0.0, 0.1, 0.2, 0.3])
        info = resolve_fs(x, manual_fs=100.0)
        assert info.value == pytest.approx(100.0)
        assert info.source == "manual"
        assert info.is_uniform is True
        assert info.warnings == []

    def test_numeric_estimation_on_index(self):
        x = pd.Series([0.0, 0.1, 0.2, 0.3])  # fs ~ 10 Hz
        info = resolve_fs(x, manual_fs=None)
        assert info.value == pytest.approx(10.0, rel=0.01)
        assert info.source == "index"
        assert info.is_uniform is True
        assert info.details.get("median_dt") == pytest.approx(0.1, rel=0.01)

    def test_datetime_estimation(self):
        x = pd.Series(pd.date_range('2025-01-01', periods=5, freq='1s'))
        info = resolve_fs(x, manual_fs=None)
        assert info.value == pytest.approx(1.0, rel=0.01)
        assert info.source == "datetime"
        assert info.is_uniform is True

    def test_zero_manual_fs_triggers_estimation(self):
        x = pd.Series([0.0, 0.5, 1.0])  # fs ~ 2 Hz
        info = resolve_fs(x, manual_fs=0.0)
        assert info.value == pytest.approx(2.0, rel=0.05)
        assert info.source == "index"

    def test_negative_manual_fs_ignored(self):
        x = pd.Series([0.0, 0.1, 0.2])
        info = resolve_fs(x, manual_fs=-10.0)
        assert info.source in {"index", "none"}

    def test_no_x_no_manual(self):
        info = resolve_fs(None, manual_fs=None)
        assert info.value is None
        assert info.source == "none"
        assert info.warnings

    def test_invalid_x_no_manual(self):
        x = pd.Series([np.nan, np.nan])
        info = resolve_fs(x, manual_fs=None)
        assert info.value is None
        assert info.source == "none"
        assert info.warnings

    def test_irregular_datetime_sampling_warning(self):
        times = pd.Series([
            pd.Timestamp('2025-01-01 00:00:00'),
            pd.Timestamp('2025-01-01 00:00:01'),
            pd.Timestamp('2025-01-01 00:00:03'),
            pd.Timestamp('2025-01-01 00:00:03.500'),
        ])
        info = resolve_fs(times, manual_fs=None)
        assert info.value is not None
        assert info.source == "datetime"
        assert info.is_uniform is False
        assert info.warnings

    def test_irregular_index_sampling_warning(self):
        x = pd.Series([0.0, 0.1, 0.4, 0.55, 1.5])
        info = resolve_fs(x, manual_fs=None)
        assert info.value is not None
        assert info.source == "index"
        assert info.is_uniform is False
        assert info.warnings


class TestValidateFilterSpec:
    """Test per validate_filter_spec()."""

    def test_disabled_filter_always_valid(self):
        """Filtro disabilitato è sempre valido."""
        spec = FilterSpec(kind="ma", enabled=False)
        ok, msg = validate_filter_spec(spec, fs=None)
        assert ok is True

    def test_ma_filter_valid(self):
        """Filtro MA valido."""
        spec = FilterSpec(kind="ma", enabled=True, ma_window=5)
        ok, msg = validate_filter_spec(spec, fs=None)
        assert ok is True

    def test_ma_filter_invalid_window(self):
        """Filtro MA con finestra < 1."""
        spec = FilterSpec(kind="ma", enabled=True, ma_window=0)
        ok, msg = validate_filter_spec(spec, fs=None)
        assert ok is False

    @pytest.mark.skipif(not _SCIPY_OK, reason="SciPy non disponibile")
    def test_butterworth_lp_valid(self):
        """Filtro Butterworth LP valido."""
        spec = FilterSpec(
            kind="butter_lp",
            enabled=True,
            order=4,
            cutoff=(10.0, None)  # 10 Hz
        )
        fs = 100.0  # fs = 100 Hz, Nyquist = 50 Hz
        ok, msg = validate_filter_spec(spec, fs)
        assert ok is True

    @pytest.mark.skipif(not _SCIPY_OK, reason="SciPy non disponibile")
    def test_butterworth_cutoff_above_nyquist(self):
        """Cutoff >= Nyquist non valida."""
        spec = FilterSpec(
            kind="butter_lp",
            enabled=True,
            order=4,
            cutoff=(60.0, None)  # 60 Hz > Nyquist (50 Hz)
        )
        fs = 100.0
        ok, msg = validate_filter_spec(spec, fs)
        assert ok is False
        assert "Nyquist" in msg

    @pytest.mark.skipif(not _SCIPY_OK, reason="SciPy non disponibile")
    def test_butterworth_bp_valid(self):
        """Filtro Butterworth BP valido."""
        spec = FilterSpec(
            kind="butter_bp",
            enabled=True,
            order=4,
            cutoff=(10.0, 40.0)
        )
        fs = 100.0  # Nyquist = 50 Hz
        ok, msg = validate_filter_spec(spec, fs)
        assert ok is True

    @pytest.mark.skipif(not _SCIPY_OK, reason="SciPy non disponibile")
    def test_butterworth_bp_inverted_cutoffs(self):
        """BP con cutoff invertite (hi < lo)."""
        spec = FilterSpec(
            kind="butter_bp",
            enabled=True,
            order=4,
            cutoff=(40.0, 10.0)  # Invertite!
        )
        fs = 100.0
        ok, msg = validate_filter_spec(spec, fs)
        assert ok is False

    def test_butterworth_without_scipy(self):
        """Butterworth senza SciPy installato."""
        if _SCIPY_OK:
            pytest.skip("SciPy disponibile, test non applicabile")

        spec = FilterSpec(
            kind="butter_lp",
            enabled=True,
            order=4,
            cutoff=(10.0, None)
        )
        ok, msg = validate_filter_spec(spec, fs=100.0)
        assert ok is False
        assert "SciPy" in msg

    def test_butterworth_without_fs(self):
        """Butterworth senza fs disponibile."""
        spec = FilterSpec(
            kind="butter_lp",
            enabled=True,
            order=4,
            cutoff=(10.0, None)
        )
        ok, msg = validate_filter_spec(spec, fs=None)
        assert ok is False


class TestMovingAverage:
    """Test per moving_average()."""

    def test_ma_smooth_signal(self):
        """MA riduce varianza di segnale rumoroso."""
        _, signal, _ = generate_white_noise(mean=0, std=1.0, n_samples=100, seed=42)

        filtered = moving_average(signal, window=10)

        # Varianza del segnale filtrato dovrebbe essere minore
        assert filtered.std() < signal.std()

    def test_ma_on_constant(self):
        """MA su costante non cambia il segnale."""
        signal = pd.Series([5.0] * 50)
        filtered = moving_average(signal, window=5)

        assert filtered.equals(signal)

    def test_ma_on_ramp(self):
        """MA su rampa."""
        _, signal, _ = generate_ramp(start=0, stop=100, n_samples=100)
        filtered = moving_average(signal, window=10)

        # Rampa liscia dovrebbe rimanere simile (con piccolo lag)
        # Confronta regione centrale
        assert np.corrcoef(signal[10:90], filtered[10:90])[0, 1] > 0.99

    def test_ma_window_1_is_identity(self):
        """MA con window=1 è identità."""
        signal = pd.Series([1, 2, 3, 4, 5])
        filtered = moving_average(signal, window=1)

        pd.testing.assert_series_equal(signal, filtered)

    def test_ma_preserves_length(self):
        """MA preserva lunghezza della serie."""
        signal = pd.Series(range(50))
        filtered = moving_average(signal, window=7)

        assert len(filtered) == len(signal)


class TestApplyFilter:
    """Test per apply_filter() orchestrator."""

    def test_apply_filter_disabled(self):
        """Filtro disabilitato ritorna originale."""
        signal = pd.Series([1, 2, 3, 4, 5])
        spec = FilterSpec(kind="ma", enabled=False)

        filtered, fs_used = apply_filter(signal, None, spec)

        pd.testing.assert_series_equal(signal, filtered)
        assert fs_used is None

    def test_apply_ma_filter(self):
        """Applica filtro MA."""
        signal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        spec = FilterSpec(kind="ma", enabled=True, ma_window=3)

        filtered, fs_used = apply_filter(signal, None, spec)

        assert len(filtered) == len(signal)
        assert fs_used is None  # MA non usa fs

    @pytest.mark.skipif(not _SCIPY_OK, reason="SciPy non disponibile")
    def test_apply_butterworth_with_fs(self):
        """Applica Butterworth con fs."""
        t, signal, _ = generate_sine_wave(frequency=5.0, fs=100.0, duration=1.0)

        spec = FilterSpec(
            kind="butter_lp",
            enabled=True,
            order=4,
            cutoff=(10.0, None)
        )

        filtered, fs_used = apply_filter(signal, t, spec, fs_override=100.0)

        assert len(filtered) == len(signal)
        assert fs_used == pytest.approx(100.0)

    def test_apply_filter_invalid_raises(self):
        """Filtro invalido solleva ValueError."""
        signal = pd.Series([1, 2, 3])
        spec = FilterSpec(kind="butter_lp", enabled=True, order=4, cutoff=(100.0, None))

        # fs=10 Hz, Nyquist=5 Hz, cutoff=100 Hz > Nyquist
        with pytest.raises(ValueError):
            apply_filter(signal, None, spec, fs_override=10.0)


class TestComputeFFT:
    """Test per compute_fft()."""

    def test_fft_on_sine_wave(self):
        """FFT su sinusoide pura trova picco alla frequenza corretta."""
        t, signal, metrics = generate_sine_wave(
            frequency=10.0,
            amplitude=2.0,
            fs=100.0,
            duration=2.0
        )

        freqs, amp = compute_fft(signal, fs=100.0, detrend=True)

        # Trova picco
        peak_idx = np.argmax(amp)
        peak_freq = freqs[peak_idx]

        # Dovrebbe essere a 10 Hz
        assert peak_freq == pytest.approx(10.0, abs=0.5)

    def test_fft_detrend_effect(self):
        """Detrend rimuove componente DC."""
        signal = pd.Series([1.0]*100 + np.random.randn(100)*0.01)  # DC + rumore

        freqs_no_detrend, amp_no_detrend = compute_fft(signal, fs=100.0, detrend=False)
        freqs_detrend, amp_detrend = compute_fft(signal, fs=100.0, detrend=True)

        # Con detrend, ampiezza a freq=0 dovrebbe essere minore
        assert amp_detrend[0] < amp_no_detrend[0]

    def test_fft_on_short_signal(self):
        """FFT su segnale troppo corto ritorna vuoto."""
        signal = pd.Series([1.0, 2.0])  # < 4 campioni

        freqs, amp = compute_fft(signal, fs=100.0)

        assert len(freqs) == 0
        assert len(amp) == 0

    def test_fft_without_fs(self):
        """FFT senza fs ritorna vuoto."""
        signal = pd.Series(np.random.randn(100))

        freqs, amp = compute_fft(signal, fs=None)

        assert len(freqs) == 0
        assert len(amp) == 0

    def test_fft_with_zero_fs(self):
        """FFT con fs=0 ritorna vuoto."""
        signal = pd.Series(np.random.randn(100))

        freqs, amp = compute_fft(signal, fs=0.0)

        assert len(freqs) == 0
        assert len(amp) == 0

    def test_fft_frequency_resolution(self):
        """Risoluzione frequenze FFT corretta."""
        fs = 100.0
        duration = 1.0
        n_samples = int(fs * duration)
        signal = pd.Series(np.random.randn(n_samples))

        freqs, amp = compute_fft(signal, fs=fs)

        # Risoluzione = fs / N
        expected_resolution = fs / n_samples
        actual_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 0

        assert actual_resolution == pytest.approx(expected_resolution, rel=0.01)

    def test_fft_nyquist_limit(self):
        """FFT non supera frequenza di Nyquist."""
        fs = 100.0
        signal = pd.Series(np.random.randn(200))

        freqs, amp = compute_fft(signal, fs=fs)

        # Max freq dovrebbe essere ~ fs/2
        assert freqs.max() <= fs / 2 + 1e-6


class TestFilterSpecDataclass:
    """Test per dataclass FilterSpec."""

    def test_filter_spec_defaults(self):
        """Valori di default FilterSpec."""
        spec = FilterSpec(kind="ma")
        assert spec.enabled is False
        assert spec.order == 4
        assert spec.cutoff is None
        assert spec.ma_window == 5

    def test_filter_spec_custom(self):
        """FilterSpec personalizzata."""
        spec = FilterSpec(
            kind="butter_lp",
            enabled=True,
            order=6,
            cutoff=(20.0, None),
            ma_window=10
        )
        assert spec.kind == "butter_lp"
        assert spec.enabled is True
        assert spec.order == 6
        assert spec.cutoff == (20.0, None)


class TestFFTSpecDataclass:
    """Test per dataclass FFTSpec."""

    def test_fft_spec_defaults(self):
        """Valori di default FFTSpec."""
        spec = FFTSpec()
        assert spec.enabled is False
        assert spec.detrend is True
        assert spec.window == "hann"

    def test_fft_spec_custom(self):
        """FFTSpec personalizzata."""
        spec = FFTSpec(enabled=True, detrend=False, window="hamming")
        assert spec.enabled is True
        assert spec.detrend is False
        assert spec.window == "hamming"


class TestIntegrationFiltersFFT:
    """Test di integrazione filtri + FFT."""

    def test_filter_then_fft_workflow(self):
        """Workflow completo: genera segnale → filtra → FFT."""
        # Genera segnale rumoroso con sinusoide
        t, signal, metrics = generate_noisy_sine(
            frequency=10.0,
            amplitude=2.0,
            noise_std=0.5,
            fs=100.0,
            duration=2.0,
            seed=42
        )

        # Applica MA per ridurre rumore
        spec = FilterSpec(kind="ma", enabled=True, ma_window=5)
        filtered, _ = apply_filter(signal, t, spec)

        # Calcola FFT
        freqs, amp = compute_fft(filtered, fs=100.0, detrend=True)

        # Trova picco
        peak_idx = np.argmax(amp)
        peak_freq = freqs[peak_idx]

        # Dovrebbe trovare la frequenza fondamentale
        assert peak_freq == pytest.approx(10.0, abs=1.0)

    @pytest.mark.skipif(not _SCIPY_OK, reason="SciPy non disponibile")
    def test_butterworth_removes_high_frequency(self):
        """Butterworth LP rimuove alte frequenze."""
        # Crea segnale con 2 componenti: 5 Hz + 40 Hz
        fs = 200.0
        duration = 2.0
        n = int(fs * duration)
        t = np.linspace(0, duration, n, endpoint=False)

        signal_low = np.sin(2 * np.pi * 5 * t)
        signal_high = 0.5 * np.sin(2 * np.pi * 40 * t)
        signal = pd.Series(signal_low + signal_high)

        # Filtro LP a 20 Hz (dovrebbe passare 5 Hz, bloccare 40 Hz)
        spec = FilterSpec(
            kind="butter_lp",
            enabled=True,
            order=6,
            cutoff=(20.0, None)
        )

        filtered, _ = apply_filter(signal, pd.Series(t), spec, fs_override=fs)

        # FFT su filtrato
        freqs, amp = compute_fft(filtered, fs=fs, detrend=True)

        # Trova picchi
        peak_indices = np.where(amp > amp.max() * 0.5)[0]
        peak_freqs = freqs[peak_indices]

        # Dovremmo vedere 5 Hz ma non 40 Hz
        assert any(abs(f - 5.0) < 2.0 for f in peak_freqs)
        assert not any(abs(f - 40.0) < 5.0 for f in peak_freqs)
