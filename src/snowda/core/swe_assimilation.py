"""Select lumped model-equivalent SWE for dated assimilation updates.

This module does not read files and does not rerun quantile mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping, Sequence

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


@dataclass(frozen=True)
class HruSwe:
    hru_id: int
    value: float
    samples: tuple[SweSample, ...]


@dataclass(frozen=True)
class DatedHruSwe:
    """Usable SWE on one assimilation date, averaged within each mapped HRU."""

    date: datetime
    hrus: tuple[HruSwe, ...]

    @property
    def samples(self) -> tuple[SweSample, ...]:
        return tuple(sample for hru in self.hrus for sample in hru.samples)

    @property
    def value(self) -> float:
        """Unweighted mean of the per-HRU means."""
        return float(np.mean([hru.value for hru in self.hrus]))


def _as_datetime(value: object) -> datetime:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.to_pydatetime()


def _assign_samples(
    obs_time: Sequence[object],
    station_ids: Sequence[object],
    swe: np.ndarray,
    flags: np.ndarray,
    dates: Sequence[datetime],
    window_days: int,
    *,
    included_stations: set[str] | None = None,
) -> dict[datetime, list[SweSample]]:
    """Assign usable samples to the nearest assimilation date.

    A sample is usable when ``flags`` is 0 (mapped) or 1 (observed zero) and
    the value is finite. It is kept when it falls within ``window_days`` of a
    date. A sample inside two windows is assigned only to the nearest date.
    Equal distances use the earlier date. Stations outside
    ``included_stations`` are ignored when that set is provided.
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
            if included_stations is not None and station_id not in included_stations:
                continue
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
    return assigned


def lumped_swe_by_date(
    obs_time: Sequence[object],
    station_ids: Sequence[object],
    swe: np.ndarray,
    flags: np.ndarray,
    dates: Sequence[datetime],
    window_days: int,
) -> dict[datetime, LumpedSwe]:
    """Average usable model-equivalent SWE assigned to each assimilation date."""
    assigned = _assign_samples(obs_time, station_ids, swe, flags, dates, window_days)
    lumped: dict[datetime, LumpedSwe] = {}
    for date, samples in assigned.items():
        if not samples:
            continue
        mean_value = float(np.mean([sample.value for sample in samples]))
        lumped[date] = LumpedSwe(date=date, value=mean_value, samples=tuple(samples))
    return lumped


def mapped_swe_by_date(
    obs_time: Sequence[object],
    station_ids: Sequence[object],
    swe: np.ndarray,
    flags: np.ndarray,
    dates: Sequence[datetime],
    window_days: int,
    station_to_hru: Mapping[str, int],
) -> dict[datetime, DatedHruSwe]:
    """Average usable SWE for each HRU named in ``station_to_hru``.

    Stations absent from the map are ignored. Several stations may share one
    HRU; their samples on a date are averaged. An HRU with no usable sample
    on a date is omitted from that date.
    """
    if not station_to_hru:
        raise ValueError("station_to_hru must name at least one station.")
    assigned = _assign_samples(
        obs_time,
        station_ids,
        swe,
        flags,
        dates,
        window_days,
        included_stations=set(station_to_hru),
    )
    hru_order: list[int] = []
    seen: set[int] = set()
    for hru_id in station_to_hru.values():
        if hru_id not in seen:
            seen.add(hru_id)
            hru_order.append(int(hru_id))
    mapped: dict[datetime, DatedHruSwe] = {}
    for date, samples in assigned.items():
        grouped: dict[int, list[SweSample]] = {hru_id: [] for hru_id in hru_order}
        for sample in samples:
            grouped[int(station_to_hru[sample.station_id])].append(sample)
        hrus_list: list[HruSwe] = []
        for hru_id in hru_order:
            hru_samples = grouped[hru_id]
            if not hru_samples:
                continue
            hrus_list.append(
                HruSwe(
                    hru_id=hru_id,
                    value=float(np.mean([sample.value for sample in hru_samples])),
                    samples=tuple(hru_samples),
                )
            )
        hrus = tuple(hrus_list)
        if hrus:
            mapped[date] = DatedHruSwe(date=date, hrus=hrus)
    return mapped
