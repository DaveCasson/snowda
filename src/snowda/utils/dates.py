"""Calendar helpers shared by the assimilation and ESP schedules."""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta
from typing import Sequence, TypeVar

Stamp = TypeVar("Stamp", bound=datetime)


def replace_year(stamp: Stamp, year: int) -> Stamp:
    """``stamp`` in ``year``. 29 February becomes 28 February in a non-leap year."""
    last_day = calendar.monthrange(year, stamp.month)[1]
    return stamp.replace(year=year, day=min(stamp.day, last_day))


def scheduled_dates(
    start: datetime,
    end: datetime,
    explicit_dates: Sequence[datetime],
    frequency_days: int | None,
    months: Sequence[int] | None = None,
) -> list[datetime]:
    """Explicit dates plus every ``frequency_days`` from ``start`` to ``end``.

    ``months`` keeps only dates in those calendar months. Explicit dates are
    not clipped to ``[start, end]``.
    """
    dates: set[datetime] = set(explicit_dates)
    if frequency_days:
        current = start
        while current <= end:
            dates.add(current)
            current += timedelta(days=frequency_days)
    if months:
        month_set = set(months)
        dates = {date for date in dates if date.month in month_set}
    return sorted(dates)
