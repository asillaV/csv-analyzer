"""
Spike Reflex — Analizzatore CSV
================================
Obiettivo: verificare la portabilità di core/ e misurare i pattern Reflex
rispetto ai 208 st.session_state e 7 st.rerun() di web_app.py.

Copertura dello spike:
  ✅ File upload          → rx.upload + handler asincrono
  ✅ Parsing CSV          → core/analyzer.py (zero modifiche)
  ✅ Selezione colonne    → rx.select + on_change
  ✅ Campionamento fs     → core/signal_tools.resolve_fs (zero modifiche)
  ✅ Filtro Butterworth   → core/signal_tools.apply_filter (zero modifiche)
  ✅ Grafico Plotly       → rx.plotly
  ✅ Sidebar layout       → rx.box CSS-based (hstack sidebar+main)
  ✅ State management     → rx.State classe tipizzata
  ✅ Gestione errori      → rx.cond + callout

Pattern NON coperti (fuori scope spike):
  ⬜ Preset (preset_manager.py) — richiede persistenza DB per multi-utente
  ⬜ FFT                  — stessa logica di filtro, pattern identico
  ⬜ Trasformazioni       — pattern identico
  ⬜ Report/export        — richiede object storage per multi-utente
"""

import io
import sys

# ── Accesso al core esistente (spike only).
# In produzione: installare csv-analyzer-core come package (pyproject.toml).
sys.path.insert(0, "/home/runner/workspace/artifacts/csv-analyzer")

import reflex as rx
import pandas as pd
import plotly.graph_objects as go

from core.analyzer import CsvAnalyzer
from core.signal_tools import resolve_fs, apply_filter, FilterSpec


# ═══════════════════════════════════════════════════════════
# STATE
# Sostituisce: st.session_state (208 riferimenti in web_app.py)
# Pattern Streamlit: dizionario implicito disperso in 2988 righe
# Pattern Reflex:    classe tipizzata, tutti i campi visibili qui
# ═══════════════════════════════════════════════════════════

FILTER_KINDS = ["Butterworth LP", "Butterworth HP", "Butterworth BP", "Media mobile (MA)"]
KIND_MAP = {
    "Butterworth LP": "butter_lp",
    "Butterworth HP": "butter_hp",
    "Butterworth BP": "butter_bp",
    "Media mobile (MA)": "ma",
}


class CsvState(rx.State):
    # ── File / DataFrame
    file_name: str = ""
    df_json: str = ""
    columns: list[str] = []
    num_rows: int = 0
    status_msg: str = "Carica un file CSV per iniziare."
    is_error: bool = False

    # ── Selezione colonne
    x_col: str = ""
    y_col: str = ""

    # ── Sidebar: campionamento
    manual_fs_input: str = "0"
    fs_display: str = ""

    # ── Sidebar: filtro
    enable_filter: bool = False
    filter_kind: str = "Butterworth LP"
    f_order_input: str = "4"
    f_lo: str = ""
    f_hi: str = ""

    # ── Output
    figure: dict = {}
    has_plot: bool = False

    # ────────────────────────────────────────
    # EVENT HANDLERS
    # Confronto con Streamlit:
    #   Streamlit: ogni widget modifica session_state e triggera un rerun globale.
    #   Reflex:    ogni evento aggiorna solo i campi necessari, nessun rerun.
    # ────────────────────────────────────────

    async def handle_upload(self, files: list[rx.UploadFile]):
        """
        Caricamento CSV.
        Streamlit: st.file_uploader, poi gestione nel corpo principale con rerun.
        Reflex:    handler asincrono esplicito. Il DataFrame viene serializzato
                   in JSON perché rx.State richiede tipi serializzabili.
        SPIKE FINDING: serializzazione JSON del DataFrame è un overhead (~2×).
                       Soluzione produzione: database o object storage per-utente,
                       non session_state.
        """
        for file in files:
            try:
                data = await file.read()
                analyzer = CsvAnalyzer(io.BytesIO(data))
                result = analyzer.analyze()
                df = pd.read_csv(
                    io.BytesIO(data),
                    delimiter=result.delimiter,
                    encoding=result.encoding,
                    header=result.header_row,
                )
                self.df_json = df.to_json(orient="split")
                self.columns = list(df.columns)
                self.num_rows = len(df)
                self.x_col = self.columns[0] if self.columns else ""
                self.y_col = self.columns[1] if len(self.columns) > 1 else ""
                self.file_name = file.filename or "file.csv"
                self.status_msg = f"✅  {self.file_name}  —  {self.num_rows:,} righe × {len(self.columns)} colonne"
                self.is_error = False
                self.figure = {}
                self.has_plot = False
                self.fs_display = ""
            except Exception as e:
                self.status_msg = f"❌ Errore parsing: {e}"
                self.is_error = True

    def set_x_col(self, value: str):
        self.x_col = value
        self.has_plot = False

    def set_y_col(self, value: str):
        self.y_col = value
        self.has_plot = False

    def set_manual_fs(self, value: str):
        self.manual_fs_input = value

    def set_filter_kind(self, value: str):
        self.filter_kind = value

    def set_f_order(self, value: str):
        self.f_order_input = value

    def set_f_lo(self, value: str):
        self.f_lo = value

    def set_f_hi(self, value: str):
        self.f_hi = value

    def toggle_filter(self, value: bool):
        self.enable_filter = value

    def plot(self):
        """
        Applica / Plot.
        Streamlit: st.form_submit_button + nonce trick per reset widget.
        Reflex:    event handler diretto, nessun nonce, nessun rerun.
        SPIKE FINDING: il nonce trick di Streamlit non serve in Reflex.
                       Reset dei widget = resettare i campi di state.
        """
        if not self.df_json:
            self.status_msg = "Nessun file caricato."
            return

        df = pd.read_json(io.StringIO(self.df_json), orient="split")

        if not self.x_col or not self.y_col:
            self.status_msg = "⚠️ Seleziona le colonne X e Y."
            return

        x = df[self.x_col]
        y_raw = pd.to_numeric(df[self.y_col], errors="coerce")

        # ── Risolvi fs — stesso core, zero modifiche al modulo
        x_num = pd.to_numeric(x, errors="coerce")
        x_for_fs = x_num if x_num.notna().mean() > 0.8 else None
        try:
            manual_fs_val = float(self.manual_fs_input)
        except (ValueError, TypeError):
            manual_fs_val = 0.0
        fs_info = resolve_fs(x_for_fs, manual_fs_val if manual_fs_val > 0 else None)
        fs_value = fs_info.value if fs_info.value and fs_info.value > 0 else None

        source_map = {
            "manual": "manuale",
            "datetime": "da timestamp",
            "index": "da indice",
        }
        self.fs_display = (
            f"fs = {fs_value:.4g} Hz ({source_map.get(fs_info.source, fs_info.source)})"
            if fs_value
            else "fs non disponibile"
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x.tolist(),
            y=y_raw.tolist(),
            mode="lines",
            name=f"{self.y_col} (originale)",
            line=dict(color="#2563eb", width=1.5),
        ))

        # ── Filtro — stesso core apply_filter, zero modifiche al modulo
        if self.enable_filter and fs_value:
            try:
                f_order = max(1, int(self.f_order_input)) if self.f_order_input else 4
                fspec = FilterSpec(
                    kind=KIND_MAP.get(self.filter_kind, "butter_lp"),
                    order=f_order,
                    f_lo=float(self.f_lo) if self.f_lo.strip() else None,
                    f_hi=float(self.f_hi) if self.f_hi.strip() else None,
                    fs=fs_value,
                )
                y_filtered = apply_filter(y_raw, fspec)
                if y_filtered is not None:
                    fig.add_trace(go.Scatter(
                        x=x.tolist(),
                        y=y_filtered.tolist(),
                        mode="lines",
                        name=f"{self.y_col} (filtrato)",
                        line=dict(color="#dc2626", width=1.8),
                    ))
            except Exception as e:
                self.status_msg = f"⚠️ Filtro non applicato: {e}"

        fig.update_layout(
            title=f"{self.y_col} vs {self.x_col}",
            xaxis_title=self.x_col,
            yaxis_title=self.y_col,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=50, b=40),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#ffffff",
            height=420,
        )

        self.figure = fig.to_dict()
        self.has_plot = True


# ═══════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════

def sidebar() -> rx.Component:
    """
    Sidebar: equivalente di `with st.sidebar:` in Streamlit.
    Differenza chiave: layout CSS esplicito (box con width fissa).
    Vantaggio: nessun limite di Streamlit su cosa mettere nella sidebar.
    """
    return rx.box(
        rx.vstack(
            rx.heading("⚙️ Campionamento", size="3", weight="bold"),
            rx.text("Frequenza di campionamento (Hz)", size="1", color_scheme="gray"),
            rx.input(
                placeholder="0 = auto",
                value=CsvState.manual_fs_input,
                on_change=CsvState.set_manual_fs,
                type="number",
                min="0",
                step="0.1",
                width="100%",
                size="2",
            ),
            rx.cond(
                CsvState.fs_display != "",
                rx.callout(
                    CsvState.fs_display,
                    color="blue",
                    size="1",
                    width="100%",
                ),
            ),

            rx.divider(width="100%"),

            rx.heading("🔧 Filtro", size="3", weight="bold"),
            rx.hstack(
                rx.checkbox(
                    on_change=CsvState.toggle_filter,
                    checked=CsvState.enable_filter,
                ),
                rx.text("Abilita filtro", size="2"),
                align="center",
                spacing="2",
            ),
            rx.cond(
                CsvState.enable_filter,
                rx.vstack(
                    rx.select(
                        FILTER_KINDS,
                        value=CsvState.filter_kind,
                        on_change=CsvState.set_filter_kind,
                        width="100%",
                        size="2",
                    ),
                    rx.text("Ordine Butterworth", size="1", color_scheme="gray"),
                    rx.input(
                        value=CsvState.f_order_input,
                        on_change=CsvState.set_f_order,
                        type="number",
                        min="1",
                        max="10",
                        width="100%",
                        size="2",
                    ),
                    rx.hstack(
                        rx.vstack(
                            rx.text("Cutoff low (Hz)", size="1", color_scheme="gray"),
                            rx.input(
                                placeholder="es. 5",
                                value=CsvState.f_lo,
                                on_change=CsvState.set_f_lo,
                                size="2",
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("Cutoff high (Hz)", size="1", color_scheme="gray"),
                            rx.input(
                                placeholder="es. 20",
                                value=CsvState.f_hi,
                                on_change=CsvState.set_f_hi,
                                size="2",
                            ),
                            spacing="1",
                        ),
                        width="100%",
                        spacing="2",
                    ),
                    width="100%",
                    align="start",
                    spacing="2",
                ),
            ),

            rx.divider(width="100%"),

            rx.badge(
                "Spike — solo layout & core",
                color_scheme="orange",
                variant="soft",
                size="1",
            ),
            rx.text(
                "Preset, FFT, trasformazioni e report richiedono persistenza "
                "DB e object storage per il multi-utente.",
                size="1",
                color_scheme="gray",
            ),

            width="100%",
            align="start",
            spacing="3",
        ),
        width="270px",
        min_width="270px",
        padding="1.2em",
        background="#f8fafc",
        border_right="1px solid #e2e8f0",
        height="100vh",
        overflow_y="auto",
    )


def upload_zone() -> rx.Component:
    """
    Zona upload drag-and-drop.
    Streamlit: st.file_uploader (widget con UI propria).
    Reflex:    rx.upload componibile — i figli sono l'area cliccabile.
    """
    return rx.upload(
        rx.vstack(
            rx.icon("upload", size=36, color="#94a3b8"),
            rx.text(
                "Trascina un CSV qui oppure clicca per sfogliare",
                color="#64748b",
                size="2",
                text_align="center",
            ),
            rx.text("Max 200 MB", size="1", color="#94a3b8"),
            align="center",
            spacing="3",
            padding="2em",
        ),
        id="csv-upload",
        accept={".csv": ["text/csv", "application/csv"]},
        on_drop=CsvState.handle_upload(rx.upload_files(upload_id="csv-upload")),
        border="2px dashed #cbd5e1",
        border_radius="10px",
        width="100%",
        cursor="pointer",
        _hover={"border_color": "#2563eb", "background": "#f0f7ff"},
    )


def column_selectors() -> rx.Component:
    """
    Selettori colonne X e Y.
    Streamlit: st.selectbox — rerun completo ad ogni cambio.
    Reflex:    on_change chiama handler preciso, aggiorna solo x_col/y_col.
    """
    return rx.hstack(
        rx.vstack(
            rx.text("Colonna X (asse tempo / indice)", size="1", weight="bold", color_scheme="gray"),
            rx.select(
                CsvState.columns,
                value=CsvState.x_col,
                on_change=CsvState.set_x_col,
                placeholder="Seleziona X…",
                width="220px",
                size="2",
            ),
            spacing="1",
        ),
        rx.vstack(
            rx.text("Colonna Y (segnale)", size="1", weight="bold", color_scheme="gray"),
            rx.select(
                CsvState.columns,
                value=CsvState.y_col,
                on_change=CsvState.set_y_col,
                placeholder="Seleziona Y…",
                width="220px",
                size="2",
            ),
            spacing="1",
        ),
        spacing="5",
        wrap="wrap",
    )


def main_area() -> rx.Component:
    return rx.box(
        rx.vstack(
            # ── Header
            rx.hstack(
                rx.vstack(
                    rx.heading("Analizzatore CSV", size="6"),
                    rx.text(
                        "Spike Reflex 0.9.3 — verifica portabilità da Streamlit",
                        size="1",
                        color_scheme="gray",
                    ),
                    align="start",
                    spacing="0",
                ),
                justify="start",
                width="100%",
            ),
            rx.divider(width="100%"),

            # ── Status bar
            rx.cond(
                CsvState.is_error,
                rx.callout(CsvState.status_msg, color="red", size="2"),
                rx.text(CsvState.status_msg, color="#475569", size="2"),
            ),

            # ── Upload o contenuto
            rx.cond(
                CsvState.df_json == "",
                upload_zone(),
                rx.vstack(
                    # Re-upload
                    rx.upload(
                        rx.button(
                            rx.icon("refresh-cw", size=14),
                            "Cambia file",
                            variant="outline",
                            size="2",
                            color_scheme="gray",
                        ),
                        id="csv-reupload",
                        accept={".csv": ["text/csv"]},
                        on_drop=CsvState.handle_upload(
                            rx.upload_files(upload_id="csv-reupload")
                        ),
                    ),

                    # Column selectors
                    column_selectors(),

                    # Plot button
                    rx.button(
                        rx.icon("chart-line", size=16),
                        "Applica / Plot",
                        on_click=CsvState.plot,
                        color_scheme="blue",
                        size="3",
                        width="100%",
                    ),

                    # Chart
                    rx.cond(
                        CsvState.has_plot,
                        rx.box(
                            rx.plotly(
                                data=CsvState.figure,
                                width="100%",
                                height="450",
                                use_resize_handler=True,
                            ),
                            width="100%",
                            border="1px solid #e2e8f0",
                            border_radius="8px",
                            overflow="hidden",
                        ),
                    ),

                    width="100%",
                    align="start",
                    spacing="4",
                ),
            ),

            width="100%",
            align="start",
            spacing="4",
            padding="1.5em",
        ),
        flex="1",
        overflow_y="auto",
        height="100vh",
    )


def index() -> rx.Component:
    """
    Layout root: sidebar fissa a sinistra + area principale scrollabile.
    Streamlit: gestito automaticamente da st.sidebar.
    Reflex:    rx.hstack con CSS esplicito. Più controllo, stessa semplicità.
    """
    return rx.hstack(
        sidebar(),
        main_area(),
        width="100%",
        spacing="0",
        align="start",
        height="100vh",
        overflow="hidden",
    )


# ═══════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════
app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="blue",
        radius="medium",
    ),
)
app.add_page(index, route="/", title="CSV Analyzer — Spike Reflex")
