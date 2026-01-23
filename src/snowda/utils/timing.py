from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from snowda.utils.logging import get_logger

F = TypeVar("F", bound=Callable[..., Any])


def timer(func: F) -> F:
    """Log the runtime of the decorated function."""

    @functools.wraps(func)
    def wrapper_timer(*args: Any, **kwargs: Any) -> Any:
        log = get_logger(__name__)
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        log.info("Function %s completed in %.4f seconds", func.__name__, run_time)
        return value

    return wrapper_timer  # type: ignore[return-value]
