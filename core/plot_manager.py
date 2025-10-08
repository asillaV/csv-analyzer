from __future__ import annotations

import json
import re
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .logger import LogManager

__all__ = ["PlotManager"]

log = LogManager("plot").get_logger()


class PlotManager:
    """Generatore di line-plot Plotly con pulizia dati, slicing, downsampling e salvataggio HTML."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.outputs_dir = self.project_root / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = self._load_config()
        log.info("PlotManager pronto. Output dir: %s", self.outputs_dir)

    # -------------------- config -------------------- #
    def _load_config(self) -> dict:
        """Carica config.json dalla root se presente; applica default robusti."""
        defaults = {
            "max_points_per_trace": 10_000,
            "nan_strategy": "drop",
            "open_mode": "html",  # html | show | both
        }
        cfg_path = self.project_root / "config.json"
        if cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                for k in defaults:
                    if k in data:
                        defaults[k] = data[k]
                log.info("Config caricata: %s", cfg_path)
            except Exception as e:
                log.warning("Config non valida (%s). Uso defaults.", e)
        return defaults

    # -------------------- utility base -------------------- #
    @staticmethod
    def _as_list(x: Union[str, Iterable[str]]) -> List[str]:
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        return [str(v) for v in x]

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_")

    @staticmethod
    def _is_number_dtype(dtype) -> bool:
        return np.issubdtype(dtype, np.number)

    @staticmethod
    def _is_datetime_dtype(series: pd.Series) -> bool:
        try:
            return pd.api.types.is_datetime64_any_dtype(series)
        except Exception:
            return False

    @staticmethod
    def _parse_open_mode(value: str) -> str:
        valid = {"html", "show", "both", "none"}
        v = (value or "html").strip().lower()
        if v not in valid:
            log.warning("open_mode '%s' non valido. Fallback a 'html'.", value)
            return "html"
        return v

    @staticmethod
    def _resolve_axis_range(
        given: Optional[Tuple[Optional[Union[str, float, int]], Optional[Union[str, float, int]]]],
        data_min,
        data_max,
        cast: str = "float",  # "float" | "datetime" | "raw"
    ) -> Optional[List]:
        if not given:
            return None

        lo, hi = given
        lo_final = data_min if (lo is None or lo == "") else lo
        hi_final = data_max if (hi is None or hi == "") else hi

        try:
            if cast == "float":
                lo_val = float(lo_final)
                hi_val = float(hi_final)
                if lo_val > hi_val:
                    lo_val, hi_val = hi_val, lo_val
                return [lo_val, hi_val]
            elif cast == "datetime":
                if pd.isna(lo_final) or pd.isna(hi_final):
                    return None
                if lo_final > hi_final:
                    lo_final, hi_final = hi_final, lo_final
                return [lo_final, hi_final]
            else:
                if lo_final > hi_final:
                    lo_final, hi_final = hi_final, lo_final
                return [lo_final, hi_final]
        except Exception:
            return None

    # -------------------- coercizioni -------------------- #
    @staticmethod
    def _coerce_numeric(series: pd.Series) -> pd.Series:
        """Converte in numerico gestendo anche la virgola decimale e Inf."""
        s = series
        out = pd.to_numeric(s, errors="coerce")

        if s.dtype == "O":
            try:
                ss = s.astype(str).str.strip()
                ss = ss.str.replace(r"\s+", "", regex=True)

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

    def _clean_dataframe(
        self,
        df: pd.DataFrame,
        x_col: Optional[str],
        y_cols: Sequence[str],
        nan_strategy: str = "drop",
    ) -> pd.DataFrame:
        work = df.copy()

        # X: se object prova numerico (solo se migliora); gestisci inf
        if x_col is not None:
            if work[x_col].dtype == "O":
                trial = self._coerce_numeric(work[x_col])
                if trial.notna().mean() >= 0.5:
                    work[x_col] = trial
            if np.issubdtype(work[x_col].dtype, np.number):
                work[x_col] = work[x_col].replace([np.inf, -np.inf], np.nan)

        # Y sempre numeriche (con virgola decimale)
        for y in y_cols:
            work[y] = self._coerce_numeric(work[y])

        cols_to_check = list(y_cols) + ([x_col] if x_col is not None else [])
        if nan_strategy == "drop":
            work = work.dropna(subset=cols_to_check)
        elif nan_strategy == "fill_zero":
            work[cols_to_check] = work[cols_to_check].fillna(0)
        elif nan_strategy == "ffill":
            work[cols_to_check] = work[cols_to_check].ffill().bfill()
        else:
            log.warning("nan_strategy '%s' non riconosciuta. Uso 'drop'.", nan_strategy)
            work = work.dropna(subset=cols_to_check)

        if work.empty:
            log.warning("Dati vuoti dopo la pulizia (nan_strategy=%s).", nan_strategy)
        return work

    # -------------------- slicing X -------------------- #
    @staticmethod
    def _parse_datetime_bound(v: Optional[Union[str, float, int]]) -> Optional[pd.Timestamp]:
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            val = float(v)
            try:
                if val > 1e15:  # ns
                    return pd.to_datetime(val, unit="ns", errors="coerce")
                elif val > 1e12:  # ms
                    return pd.to_datetime(val, unit="ms", errors="coerce")
                elif val > 1e9:  # s
                    return pd.to_datetime(val, unit="s", errors="coerce")
                else:
                    return pd.to_datetime(val, unit="s", errors="coerce")
            except Exception:
                return pd.NaT
        try:
            return pd.to_datetime(v, errors="coerce")
        except Exception:
            return pd.NaT

    def _slice_by_x_range(
        self,
        df_clean: pd.DataFrame,
        x_col: Optional[str],
        x_range: Optional[Tuple[Optional[Union[str, float, int]], Optional[Union[str, float, int]]]],
    ) -> Tuple[pd.DataFrame, Optional[List], pd.Series]:
        if df_clean.empty:
            return df_clean, None, (df_clean.index if x_col is None else df_clean[x_col])

        # Decide la serie X
        if x_col is None:
            x_series_orig = df_clean.index
            if isinstance(x_series_orig, pd.DatetimeIndex):
                typed_x = x_series_orig
                kind = "datetime"
            else:
                try:
                    typed_num = pd.to_numeric(x_series_orig, errors="coerce")
                    if typed_num.notna().mean() >= 0.8:
                        typed_x = typed_num
                        kind = "number"
                    else:
                        typed_dt = pd.to_datetime(x_series_orig, errors="coerce")
                        if typed_dt.notna().mean() >= 0.8:
                            typed_x = typed_dt
                            kind = "datetime"
                        else:
                            typed_x = x_series_orig
                            kind = "other"
                except Exception:
                    typed_x = x_series_orig
                    kind = "other"
        else:
            x_series_orig = df_clean[x_col]
            if self._is_datetime_dtype(x_series_orig):
                typed_x = x_series_orig
                kind = "datetime"
            elif self._is_number_dtype(x_series_orig.dtype):
                typed_x = x_series_orig.astype(float)
                kind = "number"
            elif x_series_orig.dtype == "O":
                typed_num = self._coerce_numeric(x_series_orig)
                if typed_num.notna().mean() >= 0.5:
                    typed_x = typed_num
                    kind = "number"
                else:
                    typed_x = x_series_orig
                    kind = "other"
            else:
                typed_x = x_series_orig
                kind = "other"

        if not x_range or (x_range[0] in (None, "") and x_range[1] in (None, "")):
            return df_clean, None, typed_x

        xmin_raw, xmax_raw = x_range

        if kind == "datetime":
            xmin = self._parse_datetime_bound(xmin_raw)
            xmax = self._parse_datetime_bound(xmax_raw)

            mask = pd.Series([True] * len(df_clean), index=df_clean.index)
            if xmin is not None and not pd.isna(xmin):
                mask &= (typed_x >= xmin)
            if xmax is not None and not pd.isna(xmax):
                mask &= (typed_x <= xmax)

            sliced = df_clean[mask]
            if sliced.empty:
                log.warning("Filtro X ha svuotato i dati (datetime): xmin=%s, xmax=%s", xmin_raw, xmax_raw)
                return sliced, None, typed_x

            x_min_data = typed_x[mask].min()
            x_max_data = typed_x[mask].max()
            xaxis_range = self._resolve_axis_range(
                (xmin, xmax), x_min_data, x_max_data, cast="datetime"
            )
            return sliced, xaxis_range, typed_x[mask]

        elif kind == "number":
            def _to_float(v):
                if v is None or v == "":
                    return None
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return float(v)
                try:
                    return float(str(v).replace(".", "").replace(",", ".")) \
                        if (str(v).count(",") > str(v).count(".")) else float(str(v).replace(",", ""))
                except Exception:
                    try:
                        return float(v)
                    except Exception:
                        return None

            xmin = _to_float(xmin_raw)
            xmax = _to_float(xmax_raw)

            mask = pd.Series([True] * len(df_clean), index=df_clean.index)
            if xmin is not None:
                mask &= (typed_x >= xmin)
            if xmax is not None:
                mask &= (typed_x <= xmax)

            sliced = df_clean[mask]
            if sliced.empty:
                log.warning("Filtro X ha svuotato i dati (numerico): xmin=%s, xmax=%s", xmin_raw, xmax_raw)
                return sliced, None, typed_x

            x_min_data = float(pd.Series(typed_x[mask]).min())
            x_max_data = float(pd.Series(typed_x[mask]).max())
            xaxis_range = self._resolve_axis_range((xmin, xmax), x_min_data, x_max_data, cast="float")
            return sliced, xaxis_range, typed_x[mask]

        else:
            log.info("X non numerica/datetime. Nessuno slicing su righe; imposto solo range asse se possibile.")
            x_min_data = typed_x.min() if hasattr(typed_x, "min") else None
            x_max_data = typed_x.max() if hasattr(typed_x, "max") else None
            if x_min_data is None or x_max_data is None:
                return df_clean, None, typed_x
            xaxis_range = self._resolve_axis_range(x_range, x_min_data, x_max_data, cast="raw")
            return df_clean, xaxis_range, typed_x

    # -------------------- downsampling -------------------- #
    @staticmethod
    def _choose_indices(n: int, max_points: int) -> np.ndarray:
        """Restituisce indici equispaziati [0..n-1] ridotti a max_points."""
        if n <= max_points:
            return np.arange(n, dtype=int)
        return np.linspace(0, n - 1, max_points, dtype=int)

    def _apply_downsampling(
        self,
        x_values: pd.Series,
        df: pd.DataFrame,
        y_list: Sequence[str],
        max_points: int,
    ) -> tuple[pd.Series, pd.DataFrame]:
        n = len(df)
        if n <= max_points:
            return x_values, df
        idx = self._choose_indices(n, max_points)
        log.info("Downsampling: %d -> %d punti per traccia.", n, len(idx))
        # x_values può essere Series/Index; convertiamo a Series indicizzabile
        if hasattr(x_values, "iloc"):
            x_ds = x_values.iloc[idx]
        else:
            x_ds = pd.Series(x_values).iloc[idx]
        df_ds = df.copy()
        df_ds = df_ds.iloc[idx]
        # Mantieni solo colonne interessate (evita gonfiare memoria)
        cols = list(dict.fromkeys(y_list))  # unique preserving order
        if any(c not in df_ds.columns for c in cols):
            return x_ds, df_ds
        return x_ds, df_ds

    # -------------------- salvataggio -------------------- #
    def _save_and_open(
        self,
        fig: go.Figure,
        html_path: Path,
        open_mode: str,
        what_label: str = "plot",
    ) -> None:
        """Salva sempre l'HTML; apri in base a open_mode; logga chiaramente le azioni."""
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        log.info("%s salvato: %s", what_label.capitalize(), html_path)

        if open_mode in ("html", "both"):
            try:
                webbrowser.open(html_path.as_uri(), new=2)
                log.info("Aperto nel browser: %s", html_path)
            except Exception as e:
                log.warning("Impossibile aprire nel browser: %s", e)

        if open_mode in ("show", "both"):
            try:
                fig.show()
                log.info("Aperto con fig.show()")
            except Exception as e:
                log.warning("fig.show() non riuscito: %s", e)

    # -------------------- API principale -------------------- #
    def plot_lines(
        self,
        df: pd.DataFrame,
        x_col: Optional[str],
        y_cols: Union[str, Iterable[str]],
        title: Optional[str] = None,
        nan_strategy: str = "drop",
        open_mode: str = "html",                     # html | show | both | none
        x_range: Optional[Tuple[Optional[Union[str, float, int]], Optional[Union[str, float, int]]]] = None,
        y_range: Optional[Tuple[Optional[Union[str, float, int]], Optional[Union[str, float, int]]]] = None,
        mode: str = "overlay",                       # "overlay" | "separate"
    ) -> Optional[Union[Path, List[Path]]]:
        """
        Genera grafici Plotly secondo i parametri richiesti.

        Ritorno:
            - overlay  -> Path dell'HTML
            - separate -> List[Path] degli HTML creati
            - None     -> nessun grafico creato
        """
        y_list = self._as_list(y_cols)
        if not y_list:
            log.error("Nessuna colonna Y fornita per il plot.")
            return None

        if x_col is not None and x_col not in df.columns:
            log.error("Colonna X '%s' non trovata nel DataFrame.", x_col)
            return None

        missing = [y for y in y_list if y not in df.columns]
        if missing:
            log.error("Colonne Y mancanti: %s", ", ".join(missing))
            return None

        # Pulizia comune
        clean = self._clean_dataframe(df, x_col, y_list, nan_strategy=nan_strategy)
        if clean.empty:
            log.warning("Nessun dato da plottare dopo la pulizia.")
            return None

        # Slicing X + range X consigliato
        clean2, xaxis_range, x_values = self._slice_by_x_range(clean, x_col, x_range)
        if clean2.empty:
            log.warning("Nessun dato dopo filtro X. Abort.")
            return None

        # Downsampling
        max_pts = int(self.cfg.get("max_points_per_trace", 10_000))
        x_values_ds, clean2_ds = self._apply_downsampling(x_values, clean2, y_list, max_pts)

        # Range Y: completa bound mancanti con min/max dei dati
        y_min_data = float(clean2_ds[y_list].min(numeric_only=True).min())
        y_max_data = float(clean2_ds[y_list].max(numeric_only=True).max())
        yaxis_range = None
        if y_range and not (y_range[0] in (None, "") and y_range[1] in (None, "")):
            yaxis_range = self._resolve_axis_range(y_range, y_min_data, y_max_data, cast="float")
            if yaxis_range:
                log.info("Range Y applicato: %s", yaxis_range)

        # open_mode validato
        omode = self._parse_open_mode(open_mode)
        title_base = title or "Analisi CSV"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ---------------- Overlay ----------------
        if mode.lower().strip() == "overlay":
            fig = go.Figure()
            for y in y_list:
                series = clean2_ds[y].dropna()
                if series.empty:
                    log.warning("Serie '%s' vuota dopo pulizia. Salto.", y)
                    continue
                # Allinea x su index della serie filtrata
                x_aligned = x_values_ds.loc[series.index] if hasattr(x_values_ds, "loc") else x_values_ds
                fig.add_trace(go.Scatter(x=x_aligned, y=series, mode="lines", name=y))

            if not fig.data:
                log.warning("Tutte le serie vuote: nessun plot generato.")
                return None

            x_label = x_col if x_col is not None else "Index"
            fig.update_layout(
                title=title_base,
                xaxis_title=x_label,
                yaxis_title="Valore",
                template="plotly_white",
                legend_title="Serie",
                margin=dict(l=50, r=30, t=60, b=50),
            )

            if xaxis_range:
                fig.update_xaxes(range=xaxis_range)
                log.info("Range X applicato: %s", xaxis_range)
            if yaxis_range:
                fig.update_yaxes(range=yaxis_range)

            out_path = self.outputs_dir / f"plot_{ts}.html"
            self._save_and_open(fig, out_path, omode, "plot")
            return out_path

        # ---------------- Separate ----------------
        elif mode.lower().strip() == "separate":
            saved: List[Path] = []
            x_label = x_col if x_col is not None else "Index"

            for y in y_list:
                series = clean2_ds[y].dropna()
                if series.empty:
                    log.warning("Serie '%s' vuota dopo pulizia. Salto.", y)
                    continue

                x_aligned = x_values_ds.loc[series.index] if hasattr(x_values_ds, "loc") else x_values_ds
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_aligned, y=series, mode="lines", name=y))

                fig.update_layout(
                    title=f"{title_base} — {y}",
                    xaxis_title=x_label,
                    yaxis_title="Valore",
                    template="plotly_white",
                    showlegend=False,
                    margin=dict(l=50, r=30, t=60, b=50),
                )

                if xaxis_range:
                    fig.update_xaxes(range=xaxis_range)
                if yaxis_range:
                    fig.update_yaxes(range=yaxis_range)

                safe_y = self._sanitize_name(y)
                out_path = self.outputs_dir / f"plot_{ts}_{safe_y}.html"
                self._save_and_open(fig, out_path, omode, f"plot '{y}'")
                saved.append(out_path)

            if not saved:
                log.warning("Nessuna figura generata in modalità 'separate'.")
                return None

            log.info("Create %d figure in modalità 'separate'.", len(saved))
            return saved

        else:
            log.warning("mode '%s' non riconosciuta. Uso 'overlay'.", mode)
            # fallback overlay
            fig = go.Figure()
            for y in y_list:
                series = clean2_ds[y].dropna()
                if series.empty:
                    continue
                x_aligned = x_values_ds.loc[series.index] if hasattr(x_values_ds, "loc") else x_values_ds
                fig.add_trace(go.Scatter(x=x_aligned, y=series, mode="lines", name=y))

            if not fig.data:
                log.warning("Tutte le serie vuote: nessun plot generato.")
                return None

            x_label = x_col if x_col is not None else "Index"
            fig.update_layout(
                title=title_base,
                xaxis_title=x_label,
                yaxis_title="Valore",
                template="plotly_white",
                legend_title="Serie",
                margin=dict(l=50, r=30, t=60, b=50),
            )

            if xaxis_range:
                fig.update_xaxes(range=xaxis_range)
                log.info("Range X applicato: %s", xaxis_range)
            if yaxis_range:
                fig.update_yaxes(range=yaxis_range)

            out_path = self.outputs_dir / f"plot_{ts}.html"
            self._save_and_open(fig, out_path, omode, "plot")
            return out_path
