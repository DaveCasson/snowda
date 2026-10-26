"""Logger setup across repeated notebook runs."""

from __future__ import annotations

import logging
from pathlib import Path

from snowda.utils.logging import configure_logging


def _file_handlers() -> list[logging.FileHandler]:
    logger = logging.getLogger("snowda")
    return [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]


def test_configure_logging_follows_the_latest_run_directory(tmp_path: Path) -> None:
    logger = logging.getLogger("snowda")
    saved_handlers = list(logger.handlers)
    logger.handlers.clear()
    try:
        first = tmp_path / "run1" / "summa.log"
        second = tmp_path / "run2" / "summa.log"
        configure_logging(level=logging.INFO, log_file=first)
        configure_logging(level=logging.INFO, log_file=second)
        handlers = _file_handlers()
        assert len(handlers) == 1
        assert Path(handlers[0].baseFilename) == second.resolve()
        logger.info("second run")
        assert "second run" in second.read_text(encoding="utf-8")
        assert not first.exists() or "second run" not in first.read_text(encoding="utf-8")
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        logger.handlers.extend(saved_handlers)
