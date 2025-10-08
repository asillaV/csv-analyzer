import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def generate_all():
    outdir = "tests_csv"
    ensure_dir(outdir)

    # 1️ Test base: singola colonna numerica
    df1 = pd.DataFrame({
        "Value [V]": np.linspace(0, 10, 50)
    })
    df1.to_csv(f"{outdir}/01_basic.csv", index=False)

    # 2️ X numerica per test fs, FFT
    fs = 1000  # Hz
    t = np.arange(0, 1, 1/fs)
    y = np.sin(2*np.pi*10*t)
    df2 = pd.DataFrame({"Time [s]": t, "Signal [V]": y})
    df2.to_csv(f"{outdir}/02_with_x_numeric.csv", index=False)

    # 3️ X datetime (per test auto-fs e slicing datetime)
    start = datetime(2025, 1, 1, 12, 0, 0)
    times = [start + timedelta(milliseconds=i*10) for i in range(200)]
    df3 = pd.DataFrame({
        "Timestamp": times,
        "Temp [°C]": np.sin(np.linspace(0, 6*np.pi, 200)) * 5 + 20
    })
    df3.to_csv(f"{outdir}/03_with_x_datetime.csv", index=False)

    # 4️ Segnale rumoroso (per test filtro MA e Butterworth)
    t = np.linspace(0, 2, 2000)
    y = np.sin(2*np.pi*5*t) + 0.4*np.random.randn(len(t))
    df4 = pd.DataFrame({"Time [s]": t, "Noisy [V]": y})
    df4.to_csv(f"{outdir}/04_noise_signal.csv", index=False)

    # 5️ Multi-colonna (per test sovrapposti/separati)
    t = np.linspace(0, 1, 500)
    df5 = pd.DataFrame({
        "Time [s]": t,
        "Ch1 [V]": np.sin(2*np.pi*3*t),
        "Ch2 [V]": np.cos(2*np.pi*5*t),
        "Ch3 [V]": np.sin(2*np.pi*7*t)
    })
    df5.to_csv(f"{outdir}/05_multicolumn.csv", index=False)

    # 6️ Valori NaN / inf (test report e coercizione)
    data = np.linspace(0, 10, 50)
    data[10:15] = np.nan
    data[20] = np.inf
    data[25] = -np.inf
    df6 = pd.DataFrame({"Signal [V]": data})
    df6.to_csv(f"{outdir}/06_nan_and_inf.csv", index=False)

    # 7️ Serie troppo corta per FFT
    df7 = pd.DataFrame({
        "Time [s]": [0.0, 0.1, 0.2],
        "Value [V]": [1.0, 2.0, 3.0]
    })
    df7.to_csv(f"{outdir}/07_short_signal.csv", index=False)

    # 8️ Dataset grande per test prestazioni e downsampling
    n = 200_000
    t = np.linspace(0, 10, n)
    y = np.sin(2*np.pi*2*t) + 0.1*np.random.randn(n)
    df8 = pd.DataFrame({"Time [s]": t, "Signal [V]": y})
    df8.to_csv(f"{outdir}/08_big_signal.csv", index=False)

    print(f"Tutti i file creati in: {os.path.abspath(outdir)}")

if __name__ == "__main__":
    generate_all()
