from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from snowda.utils.logging import get_logger

log = get_logger(__name__)


def _resolve_date_array(ds: xr.Dataset) -> pd.DatetimeIndex:
    if "time" not in ds.coords and "time" not in ds:
        raise ValueError("Archive dataset is missing a 'time' coordinate.")
    return pd.DatetimeIndex(ds["time"].values)


def _wraparound_distance(a: np.ndarray, b: int) -> np.ndarray:
    raw = np.abs(a - b)
    return np.minimum(raw, 366 - raw)


@dataclass
class Snow5ArchiveAdapter:
    """
    Archive adapter for snow_model output datasets with dimensions ``time`` × ``hru``.
    """

    archive_path: Path | str
    variable: str
    time_index: pd.DatetimeIndex | None = field(default=None, init=False)
    dataset: xr.Dataset | None = field(default=None, init=False)

    def _load_dataset(self) -> xr.Dataset:
        if self.dataset is None:
            path = Path(self.archive_path)
            if not path.exists():
                raise FileNotFoundError(f"Schaake archive not found: {path}")
            log.info("Opening Schaake archive: %s", path)
            self.dataset = xr.open_dataset(path, decode_times=True)
            if self.variable not in self.dataset:
                raise ValueError(
                    f"Archive variable '{self.variable}' missing from {path.name}"
                )
        return self.dataset

    def _get_time_index(self) -> pd.DatetimeIndex:
        if self.time_index is None:
            ds = self._load_dataset()
            self.time_index = _resolve_date_array(ds)
        return self.time_index

    def get_candidate_indices(
        self, date: datetime, window_days: int
    ) -> np.ndarray:
        times = self._get_time_index()
        if window_days < 0:
            raise ValueError("window_days must be >= 0")
        doy = int(date.timetuple().tm_yday)
        candidate_mask = _wraparound_distance(times.dayofyear.to_numpy(), doy) <= window_days
        indices = np.nonzero(candidate_mask)[0]
        if indices.size == 0:
            raise ValueError(
                f"No archive samples within ±{window_days} days of {date.date()}."
            )
        return indices

    def sample_Y(
        self,
        date: datetime,
        n_members: int,
        window_days: int,
        rng_seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        indices = self.get_candidate_indices(date, window_days)
        if len(indices) < n_members:
            raise ValueError(
                "Not enough archive samples near %s (found %d, needed %d)"
                % (date.date(), len(indices), n_members)
            )
        generator = rng or np.random.default_rng(rng_seed)
        chosen = generator.choice(indices, size=n_members, replace=False)
        ds = self._load_dataset()
        selection = ds[self.variable].isel(time=chosen)
        return np.asarray(selection.values, dtype="float64")
