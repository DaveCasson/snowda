"""Select lumped model-equivalent SWE for dated assimilation updates.

This module does not read files and does not rerun quantile mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np
import pandas as pd

USABLE_MAPPING_FLAGS = frozenset({0, 1})


@dataclass(frozen=True)
class SweSample:
    time: datetime
    station_id: str
    value: float


@dataclass(frozen=True)
class LumpedSwe:
    date: datetime
    value: float
    samples: tuple[SweSample, ...]


def _as_datetime(value: object) -> datetime:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.to_pydatetime()


def lumped_swe_by_date(
    obs_time: Sequence[object],
    station_ids: Sequence[object],
    swe: np.ndarray,
    flags: np.ndarray,
    dates: Sequence[datetime],
    window_days: int,
) -> dict[datetime, LumpedSwe]:
    """Average usable model-equivalent SWE assigned to each assimilation date.

    A sample is usable when ``flags`` is 0 (mapped) or 1 (observed zero) and
    the value is finite. It is kept when it falls within ``window_days`` of a
    date. A sample inside two windows is assigned only to the nearest date.
    Equal distances use the earlier date.
    """
    if window_days < 0:
        raise ValueError("`obs_window_days` must be >= 0.")
    values = np.asarray(swe, dtype=float)
    flag_values = np.asarray(flags)
    if values.ndim != 2 or flag_values.shape != values.shape:
        raise ValueError(
            "swe and mapping_flag must be 2-D with shape (time, station)."
        )
    times = [_as_datetime(value) for value in obs_time]
    stations = [str(station) for station in station_ids]
    if values.shape != (len(times), len(stations)):
        raise ValueError(
            "swe shape must be (time, station); "
            f"got {values.shape} for {len(times)} times and {len(stations)} stations."
        )
    ordered_dates = sorted(dates)
    window = timedelta(days=window_days)
    assigned: dict[datetime, list[SweSample]] = {date: [] for date in ordered_dates}

    for time_index, obs_stamp in enumerate(times):
        for station_index, station_id in enumerate(stations):
            flag = int(flag_values[time_index, station_index])
            value = float(values[time_index, station_index])
            if flag not in USABLE_MAPPING_FLAGS or not np.isfinite(value):
                continue
            candidates = [
                date
                for date in ordered_dates
                if abs(obs_stamp - date) <= window
            ]
            if not candidates:
                continue
            nearest = min(candidates, key=lambda date: (abs(obs_stamp - date), date))
            assigned[nearest].append(
                SweSample(time=obs_stamp, station_id=station_id, value=value)
            )

    lumped: dict[datetime, LumpedSwe] = {}
    for date, samples in assigned.items():
        if not samples:
            continue
        mean_value = float(np.mean([sample.value for sample in samples]))
        lumped[date] = LumpedSwe(date=date, value=mean_value, samples=tuple(samples))
    return lumped
