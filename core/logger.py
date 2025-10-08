from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from datetime import datetime
from typing import Optional


class LogManager:
    """
    Gestisce un logger gerarchico 'analizzatore.*' con:
    - cartella logs/ creata sempre accanto alla root del progetto,
    - file UTF-8 giornaliero 'analizzatore_YYYYMMDD.log',
    - StreamHandler su console,
    - prevenzione handler duplicati,
    - livello default INFO (configurabile).
    """

    _configured: bool = False
    _base_logger_name: str = "analizzatore"
    _logfile_path: Optional[Path] = None

    def __init__(self, component: str = "app", level: int = logging.INFO) -> None:
        self.component = component.strip() or "app"
        self.level = level
        self._ensure_configured()

    @classmethod
    def _project_root(cls) -> Path:
        # .../core/logger.py -> project_root = parent of 'core'
        return Path(__file__).resolve().parents[1]

    @classmethod
    def _ensure_configured(cls) -> None:
        if cls._configured:
            return

        project_root = cls._project_root()
        logs_dir = project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_name = f"analizzatore_{datetime.now():%Y%m%d}.log"
        cls._logfile_path = logs_dir / log_name

        base_logger = logging.getLogger(cls._base_logger_name)
        base_logger.setLevel(logging.INFO)
        base_logger.propagate = False  # Evita doppie stampe sul root

        # Formati
        common_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        file_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s"

        # Evita duplicati controllando gli handler già presenti
        existing_file = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == str(cls._logfile_path)
            for h in base_logger.handlers
        )
        existing_stream = any(isinstance(h, logging.StreamHandler) for h in base_logger.handlers)

        if not existing_file:
            fh = logging.FileHandler(cls._logfile_path, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter(file_fmt))
            base_logger.addHandler(fh)

        if not existing_stream:
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(logging.Formatter(common_fmt))
            base_logger.addHandler(sh)

        cls._configured = True
        base_logger.info("Logger configurato. File: %s", cls._logfile_path)

    def get_logger(self, level: Optional[int] = None) -> Logger:
        base = logging.getLogger(self._base_logger_name)
        logger = base.getChild(self.component)
        logger.setLevel(level if level is not None else self.level)
        return logger

    @classmethod
    def logfile_path(cls) -> Optional[Path]:
        return cls._logfile_path
