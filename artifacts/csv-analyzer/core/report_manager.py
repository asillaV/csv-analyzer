from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .logger import LogManager


log = LogManager("report").get_logger()


@dataclass
class ReportFormats:
    csv: bool = True
    md: bool = False
    html: bool = False

    @classmethod
    def from_token(cls, token: str) -> "ReportFormats":
        t = (token or "csv").strip().lower()
        if t == "csv":
            return cls(csv=True, md=False, html=False)
        if t == "csv+md":
            return cls(csv=True, md=True, html=False)
        if t == "csv+html":
            return cls(csv=True, md=False, html=True)
        if t == "csv+md+html":
            return cls(csv=True, md=True, html=True)
        # default
        return cls(csv=True, md=False, html=False)


class ReportManager:
    """
    Calcolo robusto di statistiche descrittive su colonne Y selezionate.
    - Gestione coercizione numerica (virgola decimale, inf, NaN)
    - Statistiche: count, NaN, min/max/range, mean/median/std/var, RMS, p2.5/25/75/97.5, IQR
    - first/last in base a X (se presente) o index
    - slope (retta di regressione) se X numerica
    - Output CSV (+ opzionale MD/HTML) in outputs/
    """

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.outputs_dir = self.project_root / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = self._load_config()

    def _load_config(self) -> dict:
        defaults = {
            "nan_strategy": "drop",  # drop | fill_zero | ffill (usato solo se in futuro servisse su DF in ingresso)
        }
        cfg_path = self.project_root / "config.json"
        if cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                for k in defaults:
                    if k in data:
                        defaults[k] = data[k]
                log.info("Config (report) caricata: %s", cfg_path)
            except Exception as e:
                log.warning("Config (report) non valida (%s). Uso defaults.", e)
        return defaults

    # ---------- utility coercizione ---------- #
    @staticmethod
    def _coerce_numeric(series: pd.Series) -> pd.Series:
        out = pd.to_numeric(series, errors="coerce")
        if series.dtype == "O":
            try:
                ss = series.astype(str).str.strip().str.replace(r"\s+", "", regex=True)
                sample = ss[ss.notna()].head(200)
                comma_count = sample.str.count(",").sum()
                dot_count = sample.str.count(r"\.").sum()
                if comma_count > dot_count:
                    ss2 = ss.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
                    out2 = pd.to_numeric(ss2, errors="coerce")
                    if out.isna().mean() > 0.5 and out2.isna().mean() < out.isna().mean():
                        out = out2
                else:
                    ss2 = ss.str.replace(",", "", regex=False)
                    out2 = pd.to_numeric(ss2, errors="coerce")
                    if out.isna().mean() > 0.5 and out2.isna().mean() < out.isna().mean():
                        out = out2
            except Exception:
                pass
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _linregress_slope(x: np.ndarray, y: np.ndarray) -> float:
        """Slope via OLS semplice. Restituisce NaN se var(x) ~ 0 o pochi punti."""
        if x.size < 2 or y.size < 2:
            return float("nan")
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        denom = float(np.sum((x - x_mean) ** 2))
        if denom == 0.0:
            return float("nan")
        num = float(np.sum((x - x_mean) * (y - y_mean)))
        return num / denom

    # ---------- core ---------- #
    def build_report_table(
        self,
        df: pd.DataFrame,
        x_col: Optional[str],
        y_cols: Iterable[str],
        percentiles: Tuple[float, float, float, float] = (2.5, 25.0, 75.0, 97.5),
    ) -> pd.DataFrame:
        """
        Restituisce una tabella di statistiche per ogni colonna Y.
        Non scrive su disco: usato sia da TUI che da web per download diretto.
        """
        y_list = [c for c in y_cols if c in df.columns]
        if not y_list:
            raise ValueError("Nessuna colonna Y valida trovata nel DataFrame.")

        # Prepara X (se fornita) per slope/first/last
        x_series = None
        x_kind = "none"
        if x_col and x_col in df.columns:
            x_try = df[x_col]
            # prova cast numerico (prioritario per slope)
            x_num = self._coerce_numeric(x_try)
            if x_num.notna().mean() >= 0.5:
                x_series = x_num
                x_kind = "number"
            else:
                # fallback datetime / altro
                x_dt = pd.to_datetime(x_try, errors="coerce")
                if x_dt.notna().mean() >= 0.5:
                    x_series = x_dt
                    x_kind = "datetime"
                else:
                    x_series = x_try
                    x_kind = "other"
        else:
            # usa l'indice
            idx = df.index
            try:
                idx_num = pd.to_numeric(idx, errors="coerce")
                if idx_num.notna().mean() >= 0.8:
                    x_series = idx_num
                    x_kind = "number"
                else:
                    idx_dt = pd.to_datetime(idx, errors="coerce")
                    if idx_dt.notna().mean() >= 0.8:
                        x_series = idx_dt
                        x_kind = "datetime"
                    else:
                        x_series = pd.Series(idx)
                        x_kind = "other"
            except Exception:
                x_series = pd.Series(idx)
                x_kind = "other"

        records: List[Dict[str, Union[str, float, int, None]]] = []
        p2, p25, p75, p97 = percentiles

        for y in y_list:
            raw = df[y]
            num = self._coerce_numeric(raw)
            n_total = int(len(num))
            n_valid = int(num.notna().sum())
            n_nan = int(num.isna().sum())
            nan_pct = (n_nan / n_total * 100.0) if n_total > 0 else 0.0

            note = ""
            if n_valid == 0:
                note = "Colonna non numerica o senza dati validi dopo coercizione."
                rec = {
                    "column": y,
                    "n_total": n_total,
                    "n_valid": n_valid,
                    "n_nan": n_nan,
                    "nan_pct": round(nan_pct, 3),
                    "min": np.nan,
                    "p2_5": np.nan,
                    "p25": np.nan,
                    "median": np.nan,
                    "p75": np.nan,
                    "p97_5": np.nan,
                    "max": np.nan,
                    "range": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                    "var": np.nan,
                    "rms": np.nan,
                    "first_x": None,
                    "first_y": np.nan,
                    "last_x": None,
                    "last_y": np.nan,
                    "slope": np.nan,
                    "note": note,
                }
                records.append(rec)
                continue

            y_valid = num.dropna()
            arr = y_valid.to_numpy(dtype=float)

            # statistiche base
            mn = float(np.nanmin(arr))
            mx = float(np.nanmax(arr))
            rng = mx - mn
            mean = float(np.nanmean(arr))
            # std/var con ddof=1 (campionaria) se possibile
            if arr.size >= 2:
                std = float(np.nanstd(arr, ddof=1))
                var = float(np.nanvar(arr, ddof=1))
            else:
                std = float("nan")
                var = float("nan")
            rms = float(np.sqrt(np.nanmean(np.square(arr))))
            q2_5 = float(np.nanpercentile(arr, p2))
            q25 = float(np.nanpercentile(arr, p25))
            med = float(np.nanpercentile(arr, 50.0))
            q75 = float(np.nanpercentile(arr, p75))
            q97_5 = float(np.nanpercentile(arr, p97))
            iqr = q75 - q25  # (puoi esporre come 'IQR' se vuoi una colonna separata)

            # first/last
            try:
                first_idx = y_valid.index[0]
                last_idx = y_valid.index[-1]
                first_x = x_series.loc[first_idx] if x_series is not None and hasattr(x_series, "loc") else None
                last_x = x_series.loc[last_idx] if x_series is not None and hasattr(x_series, "loc") else None
            except Exception:
                first_x = None
                last_x = None
            first_y = float(y_valid.iloc[0])
            last_y = float(y_valid.iloc[-1])

            # slope
            slope = float("nan")
            if x_kind == "number" and x_series is not None:
                # allinea sugli stessi indici
                try:
                    x_aligned = pd.Series(x_series).loc[y_valid.index]
                    x_arr = pd.to_numeric(x_aligned, errors="coerce").to_numpy(dtype=float)
                    # drop eventuali NaN in coppia
                    mask = ~np.isnan(x_arr) & ~np.isnan(arr)
                    if mask.sum() >= 2:
                        slope = self._linregress_slope(x_arr[mask], arr[mask])
                except Exception:
                    slope = float("nan")

            rec = {
                "column": y,
                "n_total": n_total,
                "n_valid": n_valid,
                "n_nan": n_nan,
                "nan_pct": round(nan_pct, 3),
                "min": mn,
                "p2_5": q2_5,
                "p25": q25,
                "median": med,
                "p75": q75,
                "p97_5": q97_5,
                "max": mx,
                "range": rng,
                "mean": mean,
                "std": std,
                "var": var,
                "rms": rms,
                "first_x": first_x if pd.notna(first_x) else None,
                "first_y": first_y,
                "last_x": last_x if pd.notna(last_x) else None,
                "last_y": last_y,
                "slope": slope,
                "note": note,
            }
            records.append(rec)

        result = pd.DataFrame.from_records(records)
        # opzionale: ordina per nome colonna
        try:
            result = result.sort_values(by="column").reset_index(drop=True)
        except Exception:
            pass
        return result

    # ---------- salvataggi ---------- #
    def _save_csv(self, df: pd.DataFrame, path: Path) -> Path:
        df.to_csv(path, index=False, encoding="utf-8")
        log.info("Report CSV salvato: %s", path)
        return path

    @staticmethod
    def _to_markdown_simple(df: pd.DataFrame) -> str:
        # Markdown senza dipendere da 'tabulate'
        cols = list(df.columns)
        header = "|" + "|".join(str(c) for c in cols) + "|\n"
        align = "|" + "|".join("---" for _ in cols) + "|\n"
        rows = []
        for _, row in df.iterrows():
            rows.append("|" + "|".join("" if pd.isna(v) else str(v) for v in row.tolist()) + "|")
        return header + align + "\n".join(rows) + "\n"

    def _save_md(self, df: pd.DataFrame, path: Path, title: str = "Report statistico") -> Path:
        content = f"# {title}\n\nGenerato il {datetime.now():%Y-%m-%d %H:%M:%S}\n\n"
        content += self._to_markdown_simple(df)
        path.write_text(content, encoding="utf-8")
        log.info("Report Markdown salvato: %s", path)
        return path

    def _save_html(self, df: pd.DataFrame, path: Path, title: str = "Report statistico") -> Path:
        html = f"""<!DOCTYPE html>
<html lang="it"><head><meta charset="UTF-8"><title>{title}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:20px;}} table{{border-collapse:collapse;width:100%;}} 
th,td{{border:1px solid #ddd;padding:6px;}} th{{background:#f4f4f4;}}</style></head>
<body>
<h1>{title}</h1>
<p>Generato il {datetime.now():%Y-%m-%d %H:%M:%S}</p>
{df.to_html(index=False, escape=False)}
</body></html>"""
        path.write_text(html, encoding="utf-8")
        log.info("Report HTML salvato: %s", path)
        return path

    # ---------- API ---------- #
    def generate_report(
        self,
        df: pd.DataFrame,
        x_col: Optional[str],
        y_cols: Iterable[str],
        formats: Union[str, ReportFormats] = "csv",
        base_name: Optional[str] = None,
    ) -> Dict[str, Optional[Path]]:
        """
        Costruisce la tabella di report e salva i formati richiesti in outputs/.
        Ritorna dizionario con i path creati: {'csv': Path|None, 'md': Path|None, 'html': Path|None}
        """
        if isinstance(formats, str):
            fmt = ReportFormats.from_token(formats)
        else:
            fmt = formats

        table = self.build_report_table(df, x_col, y_cols)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = (base_name or "report") + f"_{ts}"

        out: Dict[str, Optional[Path]] = {"csv": None, "md": None, "html": None}
        if fmt.csv:
            out["csv"] = self._save_csv(table, self.outputs_dir / f"{base}.csv")
        if fmt.md:
            out["md"] = self._save_md(table, self.outputs_dir / f"{base}.md", title="Report statistico")
        if fmt.html:
            out["html"] = self._save_html(table, self.outputs_dir / f"{base}.html", title="Report statistico")

        log.info("Report generato. Formati: %s", ", ".join(k for k, v in out.items() if v))
        return out
