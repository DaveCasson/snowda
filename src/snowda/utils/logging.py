from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


_LOGGER_NAME = "snowda"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger scoped under the snowda namespace.

    Examples:
        get_logger()            -> snowda
        get_logger(__name__)    -> snowda.snowda.core.model
    """
    if name is None:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


def configure_logging(
    *,
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
) -> None:
    """
    Configure logging for the snowda package.

    Should be called ONCE, typically in workflows/run_case.py.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times (e.g. notebooks)
    if logger.handlers:
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stdout)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Optional file handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent logs from being duplicated by root logger
    logger.propagate = False