import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def format_number_locale(
    value: float,
    decimals: int = 2,
    decimal: str = ",",
    thousands: Optional[str] = ".",
) -> str:
    """Formatta un numero usando separatori locali personalizzati."""
    sign = "-" if value < 0 else ""
    value = abs(value)
    fmt = f"{value:.{decimals}f}"
    if decimals == 0:
        int_part = fmt
        frac_part = ""
    else:
        int_part, frac_part = fmt.split(".")
    chunks = []
    while int_part:
        chunks.append(int_part[-3:])
        int_part = int_part[:-3]
    chunks = list(reversed(chunks)) or ["0"]
    if thousands:
        int_formatted = thousands.join(chunks)
    else:
        int_formatted = "".join(chunks)
    if decimals == 0:
        return f"{sign}{int_formatted}"
    return f"{sign}{int_formatted}{decimal}{frac_part}"

def generate_all():
    outdir = "tests_csv"
    ensure_dir(outdir)
    rng = np.random.default_rng(1234)

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

    # 9º Locale italiano: separatore ';', decimale ',' e migliaia '.'
    n9 = 80
    t9 = np.linspace(0, 240, n9)
    pressure9 = 1.8 + 0.25 * np.sin(np.linspace(0, 6 * np.pi, n9)) + rng.normal(0, 0.05, n9)
    flow9 = rng.normal(12345.6, 280.0, n9)
    status9 = ["OK" if i % 11 else "CHECK" for i in range(n9)]
    df9 = pd.DataFrame(
        {
            "Tempo [s]": [format_number_locale(v, decimals=1, decimal=",", thousands=".") for v in t9],
            "Pressione [bar]": [format_number_locale(v, decimals=2, decimal=",", thousands=".") for v in pressure9],
            "Portata [L/min]": [format_number_locale(v, decimals=1, decimal=",", thousands=".") for v in flow9],
            "Stato": status9,
        }
    )
    df9.to_csv(f"{outdir}/09_locale_it.csv", index=False, sep=";")

    # 10º Delimitatore TAB con migliaia spazio e simboli percentuali
    n10 = 96
    start10 = datetime(2025, 1, 1, 8, 0, 0)
    times10 = [start10 + timedelta(minutes=15 * i) for i in range(n10)]
    energy10 = rng.normal(1_250_000.0, 70_000.0, n10)
    temp10 = rng.normal(24.3, 1.3, n10)
    humidity10 = rng.normal(55.0, 4.5, n10)
    df10 = pd.DataFrame(
        {
            "Timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in times10],
            "Energia [Wh]": [
                format_number_locale(v, decimals=0, decimal=",", thousands="\u00A0") for v in energy10
            ],
            "Temp [°C]": [format_number_locale(v, decimals=1, decimal=",", thousands=None) for v in temp10],
            "Umidità [%]": [
                (
                    format_number_locale(v, decimals=1, decimal=",", thousands=None) + "%"
                    if i % 7 == 0
                    else format_number_locale(v, decimals=1, decimal=",", thousands=None)
                )
                for i, v in enumerate(humidity10)
            ],
        }
    )
    df10.to_csv(f"{outdir}/10_tab_space_thousands.csv", index=False, sep="\t")

    # 11º Valori misti con simboli '>', '~', 'n/a', migliaia ','
    n11 = 70
    length11 = rng.normal(1800.0, 120.0, n11)
    thickness11 = rng.normal(3.5, 0.45, n11)
    weight11 = rng.normal(245.0, 18.0, n11)
    mixed_thickness = []
    for val in thickness11:
        base = format_number_locale(val, decimals=2, decimal=".", thousands=",")
        r = rng.random()
        if r < 0.2:
            mixed_thickness.append(f">{base}")
        elif r < 0.35:
            mixed_thickness.append(f"~{base}")
        elif r < 0.45:
            mixed_thickness.append("n/a")
        else:
            mixed_thickness.append(base)
    df11 = pd.DataFrame(
        {
            "Lunghezza [mm]": [format_number_locale(v, decimals=1, decimal=".", thousands=",") for v in length11],
            "Spessore [mm]": mixed_thickness,
            "Peso [kg]": [
                format_number_locale(v, decimals=2, decimal=".", thousands=",") for v in weight11
            ],
        }
    )
    df11.to_csv(f"{outdir}/11_mixed_tokens.csv", index=False)

    # 12º Valori con simbolo € e margini percentuali
    n12 = 52
    start12 = datetime(2024, 1, 1)
    dates12 = [start12 + timedelta(weeks=i) for i in range(n12)]
    revenue = rng.normal(45678.9, 2200.0, n12)
    cost = revenue * rng.uniform(0.55, 0.9, n12)
    # Alcune righe con costi > ricavi per testare negativi
    for idx in range(0, n12, 9):
        cost[idx] = revenue[idx] * 1.08
    margin = revenue - cost
    margin_pct = np.divide(margin, revenue, out=np.zeros_like(margin), where=revenue != 0) * 100
    quantity = rng.integers(1_200, 8_500, n12)
    df12 = pd.DataFrame(
        {
            "Data": [d.strftime("%Y-%m-%d") for d in dates12],
            "Ricavo (€)": [f"€ {format_number_locale(v, decimals=2, decimal=',', thousands='.')}" for v in revenue],
            "Costo (€)": [
                (
                    f"({format_number_locale(v, decimals=2, decimal=',', thousands='.')})"
                    if i % 8 == 0
                    else f"{format_number_locale(v, decimals=2, decimal=',', thousands='.')} €"
                )
                for i, v in enumerate(cost)
            ],
            "Margine [%]": [
                f"{format_number_locale(v, decimals=1, decimal=',', thousands=None)}%"
                for v in margin_pct
            ],
            "Quantità": [
                format_number_locale(float(v), decimals=0, decimal=",", thousands=".")
                if i % 4 == 0
                else str(v)
                for i, v in enumerate(quantity)
            ],
        }
    )
    df12.to_csv(f"{outdir}/12_currency_euro.csv", index=False, sep=";")

    print(f"Tutti i file creati in: {os.path.abspath(outdir)}")

if __name__ == "__main__":
    generate_all()
