from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .logger import LogManager

__all__ = ["VisualPlotSpec", "VisualReportManager"]

log = LogManager("visual_report").get_logger()


@dataclass
class VisualPlotSpec:
    """Specifica per un singolo grafico nel report visivo."""

    y_column: str
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None


class VisualReportManager:
    """Crea report visivi composti da massimo 4 grafici impilati."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        project_root = Path(__file__).resolve().parents[1]
        base_dir = output_dir or (project_root / "outputs" / "visual_reports")
        base_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = base_dir

    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        import re

        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_._")
        return sanitized or "visual_report"

    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_xy(
        df: pd.DataFrame,
        y_column: str,
        x_column: Optional[str],
    ) -> tuple[pd.Series, pd.Series]:
        if y_column not in df.columns:
            raise ValueError(f"Colonna Y '{y_column}' non trovata nel DataFrame.")

        y_series = pd.to_numeric(df[y_column], errors="coerce")

        x_series: Optional[pd.Series] = None
        if x_column and x_column in df.columns:
            raw_x = df[x_column]
            if pd.api.types.is_datetime64_any_dtype(raw_x) or pd.api.types.is_timedelta64_dtype(raw_x):
                x_series = pd.to_datetime(raw_x, errors="coerce")
            else:
                x_numeric = pd.to_numeric(raw_x, errors="coerce")
                if x_numeric.notna().mean() >= 0.5:
                    x_series = x_numeric
                else:
                    x_dt = pd.to_datetime(raw_x, errors="coerce")
                    if x_dt.notna().any():
                        x_series = x_dt
            if x_series is None:
                # fallback: uso stringhe per conservare informazione
                x_series = raw_x.astype(str)
        else:
            x_series = pd.Series(df.index, name=x_column or "index")

        paired = pd.DataFrame({"x": x_series, "y": y_series}).dropna()
        if paired.empty:
            raise ValueError(
                f"Colonna '{y_column}' non contiene dati numerici validi in combinazione con X selezionata."
            )
        return paired["x"], paired["y"]

    # ------------------------------------------------------------------
    @staticmethod
    def _build_figure(
        df: pd.DataFrame,
        specs: Sequence[VisualPlotSpec],
        x_column: Optional[str],
        report_title: Optional[str],
        template: str,
        height_per_plot: int,
        show_legend: bool,
        x_range: Optional[Sequence] = None,
        y_range: Optional[Sequence[float]] = None,
    ) -> go.Figure:
        rows = len(specs)
        subplot_titles = [spec.title or spec.y_column for spec in specs]
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=False, vertical_spacing=0.08, subplot_titles=subplot_titles)

        for idx, spec in enumerate(specs, start=1):
            x_series, y_series = VisualReportManager._prepare_xy(df, spec.y_column, x_column)
            trace_name = spec.title or spec.y_column
            fig.add_trace(
                go.Scatter(x=x_series, y=y_series, mode="lines", name=trace_name),
                row=idx,
                col=1,
            )
            fig.update_yaxes(title_text=spec.y_label or spec.y_column, row=idx, col=1)
            if y_range is not None:
                fig.update_yaxes(range=list(y_range), row=idx, col=1)
            default_x_label = x_column if x_column else "Index"
            fig.update_xaxes(title_text=spec.x_label or default_x_label, row=idx, col=1)
            if x_range is not None:
                fig.update_xaxes(range=list(x_range), row=idx, col=1)

        height = max(1, rows) * height_per_plot
        fig.update_layout(
            title=report_title,
            template=template,
            showlegend=show_legend,
            height=height,
            margin=dict(l=70, r=40, t=80 if report_title else 50, b=50),
        )
        if show_legend:
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        return fig

    # ------------------------------------------------------------------
    def generate_report(
        self,
        df: pd.DataFrame,
        specs: Iterable[VisualPlotSpec],
        x_column: Optional[str] = None,
        title: Optional[str] = None,
        base_name: Optional[str] = None,
        file_format: str = "png",
        height_per_plot: int = 420,
        template: str = "plotly_white",
        show_legend: bool = False,
        scale: float = 2.0,
        x_range: Optional[Sequence] = None,
        y_range: Optional[Sequence[float]] = None,
    ) -> dict:
        """Genera il report visivo e lo salva come PNG o PDF.

        Restituisce un dict con figure Plotly, percorso del file e contenuto binario.
        """

        spec_list: List[VisualPlotSpec] = [s for s in specs]
        if not spec_list:
            raise ValueError("Specificare almeno un grafico per il report visivo.")
        if len(spec_list) > 4:
            raise ValueError("Il report visivo supporta al massimo 4 grafici.")

        fmt = (file_format or "png").strip().lower()
        if fmt not in {"png", "pdf"}:
            raise ValueError("Formato non supportato. Usare 'png' oppure 'pdf'.")

        fig = self._build_figure(
            df=df,
            specs=spec_list,
            x_column=x_column,
            report_title=title,
            template=template,
            height_per_plot=height_per_plot,
            show_legend=show_legend,
            x_range=x_range,
            y_range=y_range,
        )

        base = base_name.strip() if base_name else ""
        if not base:
            base = f"visual_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filename = f"{self._sanitize_filename(base)}.{fmt}"
        output_path = self.output_dir / filename

        width = 1280
        height = height_per_plot * len(spec_list)

        try:
            image_bytes = fig.to_image(format=fmt, width=width, height=height, scale=scale)
        except Exception as exc:
            raise RuntimeError(
                "Impossibile esportare il report visivo. Assicurarsi che il pacchetto 'kaleido' sia installato."
            ) from exc

        output_path.write_bytes(image_bytes)
        log.info("Report visivo salvato in %s", output_path)

        return {"figure": fig, "path": output_path, "bytes": image_bytes, "format": fmt}
