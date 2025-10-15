"""Generazione di segnali sintetici per test."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class SignalMetrics:
    """Metriche note per un segnale sintetico."""
    mean: float
    std: float
    min: float
    max: float
    rms: float
    frequency_hz: float | None = None  # Per segnali periodici


def generate_sine_wave(
    frequency: float = 1.0,
    amplitude: float = 1.0,
    offset: float = 0.0,
    fs: float = 100.0,
    duration: float = 1.0,
    phase: float = 0.0,
) -> Tuple[pd.Series, pd.Series, SignalMetrics]:
    """Genera un'onda sinusoidale pura.

    Args:
        frequency: Frequenza in Hz
        amplitude: Ampiezza picco-picco
        offset: Offset DC
        fs: Frequenza di campionamento
        duration: Durata in secondi
        phase: Fase iniziale in radianti

    Returns:
        (time_series, signal_series, metrics)
    """
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

    # Metriche teoriche
    mean = offset
    std = amplitude / np.sqrt(2)  # RMS per sinusoide
    rms = np.sqrt(offset**2 + (amplitude / np.sqrt(2))**2)
    min_val = offset - amplitude
    max_val = offset + amplitude

    metrics = SignalMetrics(
        mean=mean,
        std=std,
        min=min_val,
        max=max_val,
        rms=rms,
        frequency_hz=frequency,
    )

    time_series = pd.Series(t, name='time')
    signal_series = pd.Series(signal, name='signal')

    return time_series, signal_series, metrics


def generate_ramp(
    start: float = 0.0,
    stop: float = 10.0,
    n_samples: int = 100,
) -> Tuple[pd.Series, pd.Series, SignalMetrics]:
    """Genera un segnale a rampa lineare.

    Args:
        start: Valore iniziale
        stop: Valore finale
        n_samples: Numero di campioni

    Returns:
        (index_series, signal_series, metrics)
    """
    index = np.arange(n_samples)
    signal = np.linspace(start, stop, n_samples)

    # Metriche teoriche
    mean = (start + stop) / 2
    # Varianza di distribuzione uniforme continua: (b-a)²/12
    # Per rampa discreta è simile
    variance = ((stop - start) ** 2) / 12
    std = np.sqrt(variance)
    rms = np.sqrt(np.mean(signal ** 2))

    metrics = SignalMetrics(
        mean=mean,
        std=std,
        min=start,
        max=stop,
        rms=rms,
    )

    index_series = pd.Series(index, name='index')
    signal_series = pd.Series(signal, name='signal')

    return index_series, signal_series, metrics


def generate_white_noise(
    mean: float = 0.0,
    std: float = 1.0,
    n_samples: int = 1000,
    seed: int = 42,
) -> Tuple[pd.Series, pd.Series, SignalMetrics]:
    """Genera rumore bianco gaussiano.

    Args:
        mean: Media del rumore
        std: Deviazione standard
        n_samples: Numero di campioni
        seed: Seed per riproducibilità

    Returns:
        (index_series, signal_series, metrics)
    """
    rng = np.random.RandomState(seed)
    index = np.arange(n_samples)
    signal = rng.normal(loc=mean, scale=std, size=n_samples)

    # Metriche teoriche (nota: valori empirici saranno leggermente diversi)
    metrics = SignalMetrics(
        mean=mean,
        std=std,
        min=mean - 4*std,  # Approssimativo (99.99% dei dati)
        max=mean + 4*std,
        rms=np.sqrt(mean**2 + std**2),
    )

    index_series = pd.Series(index, name='index')
    signal_series = pd.Series(signal, name='signal')

    return index_series, signal_series, metrics


def generate_noisy_sine(
    frequency: float = 5.0,
    amplitude: float = 2.0,
    noise_std: float = 0.5,
    fs: float = 100.0,
    duration: float = 2.0,
    seed: int = 42,
) -> Tuple[pd.Series, pd.Series, SignalMetrics]:
    """Genera sinusoide con rumore additivo.

    Args:
        frequency: Frequenza sinusoide
        amplitude: Ampiezza sinusoide
        noise_std: Deviazione standard rumore
        fs: Frequenza campionamento
        duration: Durata in secondi
        seed: Seed per riproducibilità

    Returns:
        (time_series, signal_series, metrics_approx)
    """
    t, sine_clean, _ = generate_sine_wave(
        frequency=frequency,
        amplitude=amplitude,
        fs=fs,
        duration=duration,
    )

    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_std, len(sine_clean))
    signal = sine_clean + noise

    # Metriche approssimative (segnale + rumore)
    # Per rumore additivo indipendente: std_tot² = std_signal² + std_noise²
    std_signal = amplitude / np.sqrt(2)
    std_total = np.sqrt(std_signal**2 + noise_std**2)

    metrics = SignalMetrics(
        mean=0.0,
        std=std_total,
        min=-amplitude - 3*noise_std,
        max=amplitude + 3*noise_std,
        rms=std_total,  # Approssimazione (media ~ 0)
        frequency_hz=frequency,
    )

    signal_series = pd.Series(signal.values, name='noisy_signal')

    return t, signal_series, metrics


def generate_step_function(
    levels: list[float] = None,
    samples_per_level: int = 50,
) -> Tuple[pd.Series, pd.Series]:
    """Genera funzione a gradini.

    Args:
        levels: Livelli dei gradini
        samples_per_level: Campioni per ogni livello

    Returns:
        (index_series, signal_series)
    """
    if levels is None:
        levels = [0.0, 1.0, 0.0, -1.0, 0.0]

    signal = np.repeat(levels, samples_per_level)
    index = np.arange(len(signal))

    return pd.Series(index, name='index'), pd.Series(signal, name='signal')
