from __future__ import annotations

from core.logger import LogManager
from ui.main_app import CSVAnalyzerApp


def main() -> None:
    """Avvia l'interfaccia TUI basata su Textual."""
    logger = LogManager("analizzatore.tui").get_logger()
    try:
        CSVAnalyzerApp().run()
    except Exception as exc:
        logger.error("Errore critico nella TUI: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
