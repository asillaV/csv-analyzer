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
    def _ensure_kaleido_available() -> bool:
        """Tenta di garantire che il backend Kaleido sia importabile.

        Restituisce ``True`` se il modulo è disponibile, ``False`` in caso
        contrario. In ambienti "serverless" (es. Streamlit Cloud) l'installazione
        dichiarata nei requirements potrebbe non essere già attiva al primo
        avvio: in quel caso viene provata un'installazione automatica tramite
        ``pip``. Eventuali errori vengono loggati e segnalati al chiamante via
        ritorno ``False`` anziché generare un'eccezione bloccante.
        """

        import importlib.util
        import subprocess
        import sys

        if importlib.util.find_spec("kaleido") is not None:
            return True

        log.warning("Pacchetto 'kaleido' non trovato, tentativo di installazione runtime…")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "kaleido>=0.2.1", "--quiet"],
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                log.debug("pip install kaleido stdout: %s", result.stdout.strip())
            if result.stderr:
                log.debug("pip install kaleido stderr: %s", result.stderr.strip())
        except Exception as exc:  # pragma: no cover - dipende dall'ambiente runtime
            log.warning("Installazione automatica di 'kaleido' fallita: %s", exc, exc_info=True)
            return False

        if importlib.util.find_spec("kaleido") is None:
            log.error("'kaleido' non risulta disponibile dopo l'installazione automatica.")
            return False

        return True

    # ------------------------------------------------------------------
    @staticmethod
    def _tune_kaleido_scope() -> None:
        """Configura Kaleido per ambienti containerizzati/senza sandbox."""

        try:
            import plotly.io as pio  # pylint: disable=import-outside-toplevel
        except Exception:  # pragma: no cover - dipende da runtime esterni
            return

        extra_args = ("--disable-gpu", "--no-sandbox")

        def _merge_args(raw_value):
            current = tuple(raw_value or ())
            merged = tuple(dict.fromkeys(current + extra_args))
            if merged == current:
                return None
            if isinstance(raw_value, list):
                return list(merged)
            return merged

        defaults = getattr(pio, "defaults", None)
        for candidate in (
            getattr(getattr(defaults, "to_image", None), "kaleido", None),
            getattr(defaults, "kaleido", None),
            defaults,
        ):
            if candidate is None:
                continue
            try:
                current_args = getattr(candidate, "chromium_args")
            except AttributeError:
                continue

            merged_args = _merge_args(current_args)
            if merged_args is not None:
                try:
                    setattr(candidate, "chromium_args", merged_args)
                    return
                except AttributeError:
                    # Alcuni wrapper possono impedire l'assegnazione diretta
                    log.debug("Impossibile aggiornare chromium_args sul namespace defaults Kaleido.")
                    break

        kaleido_module = getattr(pio, "kaleido", None)
        scope = getattr(kaleido_module, "scope", None)
        if scope is None:
            return

        try:
            current_args = getattr(scope, "chromium_args")
        except AttributeError:
            log.debug(
                "Kaleido scope non espone chromium_args; salto configurazione sandbox (Plotly >= 6?)."
            )
            return

        merged_args = _merge_args(current_args)
        if merged_args is not None:
            try:
                setattr(scope, "chromium_args", merged_args)
            except AttributeError:
                log.debug("Impossibile impostare chromium_args su Kaleido scope; ignorato.")

    # ------------------------------------------------------------------
    @staticmethod
    def _describe_kaleido_failure(error: Exception | str) -> str:
        """Genera un messaggio user-friendly per gli errori Kaleido."""

        message = str(error).strip()
        if not message:
            return (
                "Esportazione Kaleido non riuscita per un motivo sconosciuto. "
                "Utilizzare il formato HTML come alternativa."
            )

        lowered = message.lower()
        if "requires google chrome" in lowered or "chromium" in lowered:
            return (
                "Kaleido richiede Google Chrome/Chromium preinstallato. "
                "In ambienti gestiti (es. Streamlit Cloud) non è possibile installarlo: "
                "utilizzare il download HTML oppure eseguire l'app in locale con un browser disponibile."
            )
        if "kaleido" in lowered and "install" in lowered:
            return (
                "La dipendenza Kaleido non è disponibile. "
                "Verificare l'installazione (pip install kaleido) oppure usare l'esportazione HTML."
            )

        return message

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
        """Genera il report visivo e lo salva come PNG, PDF oppure HTML.

        Restituisce un dict con figure Plotly, percorso del file e contenuto
        binario. In assenza del backend Kaleido (necessario per PNG/PDF) viene
        effettuato un fallback automatico su HTML interattivo.
        """

        spec_list: List[VisualPlotSpec] = [s for s in specs]
        if not spec_list:
            raise ValueError("Specificare almeno un grafico per il report visivo.")
        if len(spec_list) > 4:
            raise ValueError("Il report visivo supporta al massimo 4 grafici.")

        fmt = (file_format or "png").strip().lower()
        if fmt not in {"png", "pdf", "html"}:
            raise ValueError("Formato non supportato. Usare 'png', 'pdf' oppure 'html'.")

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
        base_name = self._sanitize_filename(base)

        width = 1280
        height = height_per_plot * len(spec_list)

        requested_fmt = fmt
        fallback_reason: Optional[str] = None

        if fmt == "html":
            export_bytes = fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
            final_fmt = "html"
        else:
            kaleido_available = self._ensure_kaleido_available()
            if kaleido_available:
                self._tune_kaleido_scope()
                try:
                    export_bytes = fig.to_image(
                        format=fmt,
                        width=width,
                        height=height,
                        scale=scale,
                    )
                    final_fmt = fmt
                except Exception as exc:  # pragma: no cover - dipende da ambiente kaleido
                    log.warning("Esportazione Kaleido fallita, uso fallback HTML: %s", exc, exc_info=True)
                    fallback_reason = self._describe_kaleido_failure(exc)
                    kaleido_available = False
            else:
                fallback_reason = self._describe_kaleido_failure(
                    "Kaleido non disponibile nell'ambiente runtime."
                )

            if fmt != "html" and not kaleido_available:
                export_bytes = fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
                final_fmt = "html"
        filename = f"{base_name}.{final_fmt}"
        output_path = self.output_dir / filename

        output_path.write_bytes(export_bytes)
        log.info("Report visivo salvato in %s", output_path)

        result = {
            "figure": fig,
            "path": output_path,
            "bytes": export_bytes,
            "format": final_fmt,
        }
        if requested_fmt != final_fmt:
            result["requested_format"] = requested_fmt
        if fallback_reason:
            result["fallback_reason"] = fallback_reason

        return result
