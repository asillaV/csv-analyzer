from __future__ import annotations

from core.logger import LogManager
from ui.desktop_app import DesktopAppTk


def main() -> None:
    """Entry point per l'interfaccia desktop Tkinter."""
    logger = LogManager("analizzatore.desktop").get_logger()
    try:
        app = DesktopAppTk()
        app.mainloop()
    except Exception as exc:
        logger.error("Errore critico nell'app desktop: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
