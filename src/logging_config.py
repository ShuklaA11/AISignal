"""Centralized logging configuration.

Uvicorn calls ``logging.config.dictConfig()`` with
``disable_existing_loggers=True``, which disables all pre-existing loggers.
We patch dictConfig to prevent this and ensure our file handler survives.
"""

import logging
import logging.config
import logging.handlers
from pathlib import Path

from src.config import DATA_DIR

_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_file_handler: logging.Handler | None = None

# ---------------------------------------------------------------------------
# Patch dictConfig IMMEDIATELY at import time so even uvicorn's first call
# is intercepted.  Note: this module must be imported before uvicorn's
# Config.__init__ runs — which it is, since app.py imports us at the top.
# ---------------------------------------------------------------------------
_original_dictConfig = logging.config.dictConfig


def _safe_dictConfig(config):
    """Wrapper that forces disable_existing_loggers=False."""
    if isinstance(config, dict):
        config = {**config, "disable_existing_loggers": False}
    _original_dictConfig(config)


logging.config.dictConfig = _safe_dictConfig


def setup_logging(level: str = "INFO", log_to_file: bool = True) -> None:
    """Configure logging for the entire application. Safe to call multiple times."""
    global _file_handler

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)
    log_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)

    # Ensure our file handler is on root (remove stale ones first)
    root.handlers = [
        h for h in root.handlers
        if not isinstance(h, logging.handlers.RotatingFileHandler)
    ]

    if log_to_file:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        log_path = DATA_DIR / "newsletter.log"
        if _file_handler is None:
            _file_handler = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=10_000_000, backupCount=3,
            )
            _file_handler.setFormatter(formatter)
        root.addHandler(_file_handler)

    # Re-enable any src.* loggers
    for name, lgr in logging.Logger.manager.loggerDict.items():
        if isinstance(lgr, logging.Logger) and name.startswith("src"):
            lgr.disabled = False
            lgr.setLevel(logging.NOTSET)

    src_logger = logging.getLogger("src")
    src_logger.disabled = False
    src_logger.setLevel(log_level)

    # Quiet noisy third-party loggers
    for name in ("httpx", "httpcore", "litellm", "apscheduler", "alembic"):
        logging.getLogger(name).setLevel(logging.WARNING)
