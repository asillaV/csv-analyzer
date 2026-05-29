"""
Spike UI — Reflex.

Stessa schermata della versione NiceGUI, ma con il modello Reflex: uno `State`
esplicito (classe con variabili reattive) e handler di evento. Questo è il
modello di stato che rende Reflex adatto al multiutente — lo stato è per-sessione
e tipizzato, non un dizionario globale ricostruito a ogni rerun.

Riusa `spike/shared.py` (e quindi `core/`) esattamente come la versione NiceGUI.

Toolchain: a differenza di NiceGUI, Reflex compila un frontend Next.js sotto il
cofano. Il primo `reflex run` scarica un toolchain JS (bun/node) — vedi README
per le implicazioni sulla scelta del framework.
"""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import reflex as rx

# `spike/` nel path per importare il modulo condiviso.
_SPIKE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SPIKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPIKE_DIR))

import shared  # noqa: E402


class State(rx.State):
    columns: list[str] = []
    x_col: str = ""
    y_col: str = ""
    filter_kind: str = "butter_lp"
    cutoff: float = 10.0
    info: str = ""
    warning: str = ""
    time_fig: go.Figure = go.Figure()
    fft_fig: go.Figure = go.Figure()

    # Setter espliciti: in questa versione di Reflex i setter `set_<var>`
    # non sono più auto-generati di default (dato di confronto DX).
    def set_x_col(self, v: str):
        self.x_col = v

    def set_y_col(self, v: str):
        self.y_col = v

    def set_filter_kind(self, v: str):
        self.filter_kind = v

    def set_cutoff(self, v: str):
        try:
            self.cutoff = float(v)
        except (TypeError, ValueError):
            self.cutoff = 0.0

    def on_load(self):
        self.columns = shared.list_columns(shared.DEFAULT_CSV)
        self.x_col = self.columns[0]
        self.y_col = self.columns[3] if len(self.columns) > 3 else self.columns[-1]
        self.run()

    def run(self):
        try:
            res = shared.analyze(
                shared.DEFAULT_CSV, x_col=self.x_col, y_col=self.y_col,
                filter_kind=self.filter_kind, cutoff_hz=float(self.cutoff or 10.0),
            )
        except Exception as exc:
            self.warning = f"Errore: {exc}"
            return
        self.info = res.info
        self.warning = res.warning
        self.time_fig = res.time_fig
        self.fft_fig = res.fft_fig


def index() -> rx.Component:
    return rx.vstack(
        rx.heading("CSV Analyzer — spike Reflex", size="7"),
        rx.text("Core riusato da spike/shared.py (nessuna dipendenza da Streamlit).",
                color_scheme="gray", size="2"),
        rx.hstack(
            rx.select(State.columns, value=State.x_col, on_change=State.set_x_col,
                      placeholder="Colonna X"),
            rx.select(State.columns, value=State.y_col, on_change=State.set_y_col,
                      placeholder="Colonna Y"),
            rx.select(shared.FILTER_CHOICES, value=State.filter_kind,
                      on_change=State.set_filter_kind),
            rx.input(value=State.cutoff.to_string(), on_change=State.set_cutoff,
                     type="number", width="8em"),
            rx.button("Analizza", on_click=State.run),
            spacing="3", align="end",
        ),
        rx.text(State.info, font_family="monospace", size="2"),
        rx.cond(State.warning != "", rx.callout(State.warning, color_scheme="orange")),
        rx.plotly(data=State.time_fig, width="100%"),
        rx.plotly(data=State.fft_fig, width="100%"),
        spacing="4", padding="1.5em", width="100%",
    )


app = rx.App()
app.add_page(index, on_load=State.on_load)
