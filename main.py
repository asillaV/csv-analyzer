from __future__ import annotations

from core.logger import LogManager

def main() -> None:
    logger = LogManager("analizzatore.main").get_logger()
    try:
        from ui.main_app import CSVAnalyzerApp
        app = CSVAnalyzerApp()
        app.run()
    except Exception as e:
        logger.error("Errore critico in main: %s", e, exc_info=True)
        print(f"[ERRORE] {e}")

if __name__ == "__main__":
    main()
