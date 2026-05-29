"""
Spike UI — NiceGUI.

Schermata singola: scegli colonna Y + filtro, calcola FFT, mostra serie
temporale e spettro. Usa `spike/shared.py` (quindi `core/`) senza Streamlit e
senza aprire il browser dal core.

Stato per-client: la funzione decorata con `@ui.page("/")` viene eseguita una
volta per ogni connessione, quindi le variabili locali sono isolate per utente.
È il modello di concorrenza che ci serve per il SaaS (al contrario del rerun
globale di Streamlit).

Avvio:
    .venv-spike/bin/python spike/nicegui_app.py
Poi apri http://localhost:8080
"""
from __future__ import annotations

from nicegui import ui

import shared


@ui.page("/")
def index() -> None:
    state = {"path": str(shared.DEFAULT_CSV)}
    columns = shared.list_columns(state["path"])

    ui.label("CSV Analyzer — spike NiceGUI").classes("text-2xl font-bold")
    ui.label("Core riusato da spike/shared.py (nessuna dipendenza da Streamlit).") \
        .classes("text-sm text-gray-500")

    with ui.row().classes("items-end gap-4"):
        x_sel = ui.select(columns, value=columns[0], label="Colonna X").classes("w-48")
        y_sel = ui.select(columns, value=columns[3], label="Colonna Y").classes("w-48")
        f_sel = ui.select(shared.FILTER_CHOICES, value="butter_lp",
                          label="Filtro").classes("w-40")
        cut = ui.number("Cutoff [Hz]", value=10.0, min=0.1, step=0.5).classes("w-32")

    info = ui.label("").classes("text-sm font-mono")
    warn = ui.label("").classes("text-sm text-orange-600")
    time_plot = ui.plotly(shared.go.Figure()).classes("w-full")
    fft_plot = ui.plotly(shared.go.Figure()).classes("w-full")

    def run() -> None:
        try:
            res = shared.analyze(
                state["path"], x_col=x_sel.value, y_col=y_sel.value,
                filter_kind=f_sel.value, cutoff_hz=float(cut.value or 10.0),
            )
        except Exception as exc:  # mostra errore senza crashare la sessione
            warn.text = f"Errore: {exc}"
            return
        info.text = res.info
        warn.text = res.warning
        time_plot.update_figure(res.time_fig)
        fft_plot.update_figure(res.fft_fig)

    def on_upload(e) -> None:
        path = shared.write_temp_upload(e.content.read())
        state["path"] = path
        cols = shared.list_columns(path)
        x_sel.options = cols
        y_sel.options = cols
        x_sel.value, y_sel.value = cols[0], cols[min(1, len(cols) - 1)]
        x_sel.update(); y_sel.update()
        run()

    ui.upload(label="Carica un CSV (opzionale)", on_upload=on_upload, auto_upload=True) \
        .classes("max-w-md")
    ui.button("Analizza", on_click=run).props("color=primary")

    run()  # render iniziale con il CSV di esempio


ui.run(title="CSV Analyzer — spike NiceGUI", port=8080, reload=False, show=False)
