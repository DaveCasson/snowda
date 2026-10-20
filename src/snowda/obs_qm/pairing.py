"""Align point SWE observations with colocated midnight model SWE.

Observed NorSWE is daily at 00:00. Modeled SUMMA SWE is typically hourly.
Pairing uses the model value at 00:00 on the observation date only:

- no daily averaging
- no interpolation
- no gap filling

If a midnight model sample is missing, that date is unpaired (NaN).
Times of day more than ``midnight_tolerance`` after 00:00 (default
1 minute) are never treated as midnight, so a 01:00 step is not used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_MIDNIGHT_TOLERANCE = pd.Timedelta(minutes=1)


def _naive_datetime_index(values) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if times.tz is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    return times.rename("time")


def extract_midnight_swe(
    da: xr.DataArray,
    *,
    tolerance: pd.Timedelta | str = DEFAULT_MIDNIGHT_TOLERANCE,
) -> xr.DataArray:
    """Keep model samples whose time-of-day is within ``tolerance`` of 00:00.

    The returned array has a calendar-date ``time`` coordinate (midnight
    normalized). Duplicate midnights on the same date raise; they are not
    averaged.
    """
    if "time" not in da.dims:
        raise ValueError(f"Expected a 'time' dimension (dims={list(da.dims)}).")
    extra = [d for d in da.dims if d != "time"]
    if extra:
        raise ValueError(
            f"Model SWE still has extra dimensions {extra}; select a single "
            "HRU/location before extracting midnight values."
        )

    tol = pd.Timedelta(tolerance)
    if tol < pd.Timedelta(0):
        raise ValueError("midnight tolerance must be >= 0.")

    times = _naive_datetime_index(da["time"].values)
    time_of_day = times - times.normalize()
    midnight_mask = np.asarray(time_of_day <= tol)
    n_keep = int(midnight_mask.sum())
    if n_keep == 0:
        empty = da.isel(time=slice(0, 0)).copy()
        empty = empty.assign_coords(time=("time", np.array([], dtype="datetime64[ns]")))
        empty.attrs["model_time_selection"] = "midnight"
        return empty

    selected = da.isel(time=np.flatnonzero(midnight_mask)).copy()
    dates = times[midnight_mask].normalize()
    if dates.duplicated().any():
        n_dup = int(dates.duplicated().sum())
        raise ValueError(
            f"Found {n_dup} duplicate midnight sample(s) on the same date. "
            "Refusing to average; inspect the model time axis."
        )
    selected = selected.assign_coords(time=("time", dates.to_numpy()))
    selected.attrs.update(da.attrs)
    selected.attrs["model_time_selection"] = "midnight"
    selected.attrs["midnight_tolerance"] = str(tol)
    return selected


def align_model_to_obs_time(
    obs: xr.DataArray,
    model: xr.DataArray,
) -> xr.DataArray:
    """Reindex midnight model SWE onto the observation time axis.

    Missing model dates remain NaN. No interpolation or filling.
    """
    if "time" not in obs.dims:
        raise ValueError(f"Expected obs 'time' dimension (dims={list(obs.dims)}).")
    if "time" not in model.dims:
        raise ValueError(f"Expected model 'time' dimension (dims={list(model.dims)}).")

    obs_times = _naive_datetime_index(obs["time"].values)
    model_times = _naive_datetime_index(model["time"].values)
    obs = obs.assign_coords(time=("time", obs_times.to_numpy()))
    model = model.assign_coords(time=("time", model_times.to_numpy()))
    aligned = model.reindex(time=obs["time"], method=None)
    aligned.name = model.name or "swe_model"
    aligned.attrs.update(model.attrs)
    aligned.attrs["alignment"] = "reindex_to_obs_time_no_fill"
    return aligned


def pair_obs_model(
    obs: xr.DataArray,
    model: xr.DataArray,
) -> xr.Dataset:
    """Return obs and colocated model on the observation time axis.

    The dataset always has the observation times. Model values are NaN
    where no midnight sample exists. A ``paired`` boolean marks dates
    where both values are finite.
    """
    extra_obs = [d for d in obs.dims if d != "time"]
    if extra_obs:
        raise ValueError(
            f"obs must be 1-D in time for pairing (extra dims={extra_obs})."
        )
    extra_model = [d for d in model.dims if d != "time"]
    if extra_model:
        raise ValueError(
            f"model must be 1-D in time for pairing (extra dims={extra_model})."
        )

    model_aligned = align_model_to_obs_time(obs, model)
    obs_time = _naive_datetime_index(obs["time"].values)
    obs = obs.assign_coords(time=("time", obs_time.to_numpy()))
    obs_vals = np.asarray(obs.values, dtype=np.float64)
    model_vals = np.asarray(model_aligned.values, dtype=np.float64)
    paired = np.isfinite(obs_vals) & np.isfinite(model_vals)

    ds = xr.Dataset(
        {
            "swe_obs": ("time", obs_vals),
            "swe_model": ("time", model_vals),
            "paired": ("time", paired),
        },
        coords={"time": obs["time"]},
    )
    ds["swe_obs"].attrs.update(obs.attrs)
    ds["swe_obs"].attrs.setdefault("long_name", "observed SWE")
    ds["swe_model"].attrs.update(model_aligned.attrs)
    ds["swe_model"].attrs.setdefault("long_name", "colocated midnight model SWE")
    ds["paired"].attrs["long_name"] = "both observed and midnight model SWE are finite"
    ds.attrs["n_obs_times"] = int(obs_time.size)
    ds.attrs["n_paired"] = int(paired.sum())
    ds.attrs["n_unpaired_missing_model"] = int(np.isfinite(obs_vals).sum() - paired.sum())
    return ds


def squeeze_single_location(da: xr.DataArray, *, hru: int | None = None, hru_id: Any | None = None) -> xr.DataArray:
    """Reduce model SWE to a 1-D time series.

    Lumped files with one HRU are squeezed automatically. Multiple HRUs
    require ``hru`` (positional index) or ``hru_id``.
    """
    out = da
    if "hru" not in out.dims:
        extra = [d for d in out.dims if d != "time"]
        if extra:
            out = out.squeeze(drop=True)
        extra = [d for d in out.dims if d != "time"]
        if extra:
            raise ValueError(f"Cannot squeeze model SWE with extra dims {extra}.")
        return out

    if hru_id is not None:
        if "hruId" in out.coords:
            matching = np.asarray(out["hruId"].values) == hru_id
            hits = np.flatnonzero(np.atleast_1d(matching))
            if hits.size == 0:
                raise ValueError(f"hru_id={hru_id!r} not found in hruId.")
            if hits.size > 1:
                raise ValueError(f"hru_id={hru_id!r} matches {hits.size} HRUs.")
            out = out.isel(hru=int(hits[0]))
        else:
            out = out.sel(hru=hru_id)
    elif hru is not None:
        out = out.isel(hru=int(hru))
    elif out.sizes["hru"] == 1:
        out = out.isel(hru=0)
    else:
        raise ValueError(
            f"Model SWE has {out.sizes['hru']} HRUs; pass hru or hru_id "
            "to select the colocated location."
        )
    extra = [d for d in out.dims if d != "time"]
    if extra:
        out = out.squeeze(drop=True)
    extra = [d for d in out.dims if d != "time"]
    if extra:
        raise ValueError(f"Model SWE still has extra dimensions {extra}.")
    return out
