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
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    has_console = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    if not has_console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    if log_file is None:
        return

    # A notebook kernel keeps the snowda logger alive across runs. Point the
    # file handler at this run so later runs are not stuck writing the first log.
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)