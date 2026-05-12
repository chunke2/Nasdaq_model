"""Structured logging configuration.

All modules should use `get_logger(__name__)` to obtain a configured logger.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

_log_initialized: bool = False


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """Configure root logger with structured formatting.

    Call once at application startup. Subsequent calls are no-ops.
    """
    global _log_initialized
    if _log_initialized:
        return

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root.addHandler(console)

    # Optional file handler
    if log_file is not None:
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        root.addHandler(file_handler)

    # Silence overly chatty third-party loggers
    for noisy in ("urllib3", "yfinance", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _log_initialized = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for `name`, ensuring logging is set up."""
    if not _log_initialized:
        setup_logging()
    return logging.getLogger(name)
