from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Sequence

import pandas as pd

from snowda.config.pysummada import PysummadaConfig
from snowda.utils.logging import get_logger

log = get_logger(__name__)


def restart_filename(prefix: str, when: datetime) -> str:
    """SUMMA restart name ``{prefix}_restart_{YYYYMMDDHH}_{YYYY_MM_DD}.nc``."""
    return (
        f"{prefix}_restart_{when.strftime('%Y%m%d%H')}_"
        f"{when.strftime('%Y_%m_%d')}.nc"
    )


def resolve_assimilation_dates(
    start_time: datetime,
    end_time: datetime,
    explicit_dates: Sequence[datetime],
    frequency_days: int | None,
    months: Sequence[int] | None = None,
) -> list[datetime]:
    """Resolve assimilation stops inside ``(start_time, end_time]``.

    Explicit dates and an optional day frequency are combined the same way as
    ESP triggers. A date equal to ``start_time`` is rejected. Dates outside
    the run are dropped.
    """
    if not explicit_dates and not frequency_days:
        raise ValueError(
            "Assimilation requires `assimilation.dates` or `assimilation.frequency_days`."
        )
    if any(date == start_time for date in explicit_dates):
        raise ValueError(
            "Assimilation dates must be after start_time so a forecast segment precedes them."
        )
    dates: set[datetime] = set(explicit_dates)
    if frequency_days:
        if frequency_days <= 0:
            raise ValueError("`assimilation.frequency_days` must be > 0.")
        current = start_time
        while current <= end_time:
            dates.add(current)
            current += timedelta(days=frequency_days)
    if months:
        month_set = set(months)
        dates = {date for date in dates if date.month in month_set}
    resolved = sorted(date for date in dates if start_time < date <= end_time)
    if not resolved:
        raise ValueError("No assimilation dates fall inside (start_time, end_time].")
    return resolved


def assimilation_segments(
    start_time: datetime,
    end_time: datetime,
    dates: Sequence[datetime],
) -> list[tuple[datetime, datetime, bool]]:
    """Return ``(segment_start, segment_end, assimilate)`` covering the run.

    Assimilate only when the segment ends on an assimilation date. A tail
    from the last date to ``end_time`` is open-loop.
    """
    ordered = sorted(dates)
    if not ordered:
        raise ValueError("Assimilation segments require at least one date.")
    bounds = [start_time, *ordered]
    if end_time > ordered[-1]:
        bounds.append(end_time)
    date_set = set(ordered)
    return [(left, right, right in date_set) for left, right in zip(bounds, bounds[1:])]


@dataclass
class DataAssimilationCore:
    config: PysummadaConfig
    ens_config: Dict[str, Dict[str, Dict[str, str]]] = field(default_factory=dict)

    def set_run_period(self) -> pd.DatetimeIndex:
        log.info(
            "Setting run period from %s to %s", self.config.start_time, self.config.end_time
        )
        self.run_period = pd.date_range(
            self.config.start_time,
            self.config.end_time - self.config.time_delta,
            freq="d",
        )
        return self.run_period

    def create_ens_config(self, ens_file_managers: list[str]) -> Dict[str, Dict[str, str]]:
        file_manager_dicts = [{"file_manager": fm} for fm in ens_file_managers]
        config_zip = zip(self.config.ens_label_array, file_manager_dicts)
        config_dict = dict(config_zip)
        self.ens_config = config_dict
        return config_dict

    def update_run_time(self, start_time: datetime, end_time: datetime) -> Dict[str, Dict[str, str]]:
        for ens in self.config.ens_label_array:
            entry = self.ens_config.setdefault(ens, {})
            manager = entry.setdefault("manager", {})
            manager["simStartTime"] = start_time.strftime("%Y-%m-%d %H:%M")
            manager["simEndTime"] = end_time.strftime("%Y-%m-%d %H:%M")
        return self.ens_config

    def update_warm_state(self, end_time: datetime) -> Dict[str, Dict[str, str]]:
        warm_states = self.create_warm_state_list(end_time)
        config_list = [
            {ens: {"manager": {"initConditionFile": warm_state}}}
            for ens, warm_state in zip(self.config.ens_label_array, warm_states)
        ]
        config_dict = self._merge_dicts(config_list)
        self.ens_config = self.merge_dict(self.ens_config, config_dict)
        return self.ens_config

    @staticmethod
    def merge_dict(a: Dict, b: Dict, path: list[str] | None = None) -> Dict:
        if path is None:
            path = []
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    DataAssimilationCore.merge_dict(a[key], b[key], path + [str(key)])
                elif a[key] != b[key]:
                    raise ValueError(f"Conflict at {'.'.join(path + [str(key)])}")
            else:
                a[key] = b[key]
        return a

    def create_warm_state_list(self, end_time: datetime) -> list[str]:
        return [
            restart_filename(run_prefix, end_time)
            for run_prefix in self.config.ens_label_array
        ]

    @staticmethod
    def _merge_dicts(dicts: list[Dict]) -> Dict:
        merged: Dict = {}
        for item in dicts:
            merged = DataAssimilationCore.merge_dict(merged, item)
        return merged
