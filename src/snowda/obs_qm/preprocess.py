"""Station-specific observation-to-model SWE quantile mapping.

This is observation preprocessing for later data assimilation. The model
state is never modified. Each station has its own empirical mapping.

Flag values on ``mapping_flag``:

    0  mapped with empirical quantile mapping
    1  observed SWE is exactly zero; model-equivalent set to zero
    2  fewer than min_samples paired positive SWE values in the window
    3  missing or invalid observation
    4  reserved: no midnight model sample (unpaired dates are NaN in
       swe_model; mapping a target observation does not require a
       same-day model value)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr

from snowda.obs_qm.mapping import PLOTTING_POSITION, quantile_map_obs_to_model
from snowda.obs_qm.pairing import align_model_to_obs_time
from snowda.obs_qm.seasonal import (
    DEFAULT_WATER_YEAR_START_MONTH,
    DEFAULT_WINDOW_DAYS,
    get_seasonal_samples,
    observation_year_coverage,
    positive_paired_samples,
    water_year,
)

FLAG_MAPPED = np.int8(0)
FLAG_ZERO_OBS = np.int8(1)
FLAG_INSUFFICIENT_SAMPLES = np.int8(2)
FLAG_MISSING_OBS = np.int8(3)
FLAG_MISSING_MODEL = np.int8(4)

FLAG_MEANINGS = (
    "mapped zero_obs insufficient_samples missing_obs missing_model"
)
FLAG_VALUES = np.array([0, 1, 2, 3, 4], dtype=np.int8)

DEFAULT_MIN_SAMPLES = 20
DEFAULT_MIN_YEARS = 5


def _as_1d_time_series(da: xr.DataArray, name: str) -> xr.DataArray:
    if "time" not in da.dims:
        raise ValueError(f"{name} must have a 'time' dimension (dims={list(da.dims)}).")
    extra = [d for d in da.dims if d != "time"]
    if extra:
        da = da.squeeze(drop=True)
    extra = [d for d in da.dims if d != "time"]
    if extra:
        raise ValueError(f"{name} must be 1-D in time (extra dims={extra}).")
    times = pd.DatetimeIndex(pd.to_datetime(da["time"].values, errors="coerce"))
    if times.tz is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    return da.assign_coords(time=("time", times.to_numpy()))


def zero_frequency_diagnostics(
    obs: np.ndarray,
    model: np.ndarray,
) -> dict[str, float]:
    """Zero-SWE frequencies on dates where both series are finite."""
    obs_vals = np.asarray(obs, dtype=np.float64)
    model_vals = np.asarray(model, dtype=np.float64)
    paired = np.isfinite(obs_vals) & np.isfinite(model_vals)
    n_paired = int(paired.sum())
    if n_paired == 0:
        return {
            "n_paired": 0.0,
            "obs_zero_fraction": np.nan,
            "model_zero_fraction": np.nan,
            "both_positive_fraction": np.nan,
        }
    obs_p = obs_vals[paired]
    model_p = model_vals[paired]
    return {
        "n_paired": float(n_paired),
        "obs_zero_fraction": float(np.mean(obs_p == 0.0)),
        "model_zero_fraction": float(np.mean(model_p == 0.0)),
        "both_positive_fraction": float(np.mean((obs_p > 0.0) & (model_p > 0.0))),
    }


def preprocess_station(
    obs: xr.DataArray,
    model: xr.DataArray,
    station_id: Any = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    cross_validate: bool = True,
    water_year_start_month: int = DEFAULT_WATER_YEAR_START_MONTH,
    calibration_start=None,
    calibration_end=None,
) -> xr.Dataset:
    """Quantile-map one station's observations into model SWE space.

    ``obs`` and ``model`` are 1-D time series. ``model`` should already be
    midnight-extracted; it is reindexed onto ``obs`` time with no filling.
    Same-day model values are not required to transform a target
    observation; they are only used when building seasonal CDFs.
    """
    if window_days < 0:
        raise ValueError("window_days must be >= 0.")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")

    obs = _as_1d_time_series(obs, "obs")
    model = _as_1d_time_series(model, "model")
    model_aligned = align_model_to_obs_time(obs, model)

    times = pd.DatetimeIndex(obs["time"].values)
    obs_vals = np.asarray(obs.values, dtype=np.float64)
    model_vals = np.asarray(model_aligned.values, dtype=np.float64)
    wy = water_year(times, start_month=water_year_start_month)

    y_star = np.full(times.size, np.nan, dtype=np.float64)
    q_obs = np.full(times.size, np.nan, dtype=np.float64)
    n_samples = np.zeros(times.size, dtype=np.int32)
    flags = np.full(times.size, FLAG_MISSING_OBS, dtype=np.int8)
    paired = np.isfinite(obs_vals) & np.isfinite(model_vals)

    cache: dict[tuple[int, Optional[int]], tuple[np.ndarray, np.ndarray]] = {}

    for i in range(times.size):
        y = obs_vals[i]
        if not np.isfinite(y) or y < 0.0:
            flags[i] = FLAG_MISSING_OBS
            continue
        if y == 0.0:
            y_star[i] = 0.0
            flags[i] = FLAG_ZERO_OBS
            continue

        exclude = int(wy[i]) if cross_validate else None
        target_doy = int(times[i].dayofyear)
        key = (target_doy, exclude)
        if key not in cache:
            obs_s, mod_s = get_seasonal_samples(
                times,
                obs_vals,
                model_vals,
                times[i],
                window_days=window_days,
                exclude_water_year=exclude,
                water_year_start_month=water_year_start_month,
                calibration_start=calibration_start,
                calibration_end=calibration_end,
            )
            cache[key] = positive_paired_samples(obs_s, mod_s)
        obs_pos, mod_pos = cache[key]
        n_samples[i] = int(obs_pos.size)
        if obs_pos.size < min_samples:
            flags[i] = FLAG_INSUFFICIENT_SAMPLES
            continue
        y_star[i], q_obs[i] = quantile_map_obs_to_model(y, obs_pos, mod_pos)
        flags[i] = FLAG_MAPPED

    zeros = zero_frequency_diagnostics(obs_vals, model_vals)
    years = observation_year_coverage(
        times, obs_vals, start_month=water_year_start_month
    )
    ds = xr.Dataset(
        {
            "swe_obs": ("time", obs_vals),
            "swe_model": ("time", model_vals),
            "swe_model_equivalent": ("time", y_star),
            "quantile_obs": ("time", q_obs),
            "n_samples_mapping": ("time", n_samples),
            "mapping_flag": ("time", flags),
            "paired": ("time", paired),
        },
        coords={"time": times.to_numpy()},
    )
    if station_id is not None:
        ds = ds.assign_coords(station_id=station_id)

    units = obs.attrs.get("units") or model.attrs.get("units") or "kg m-2"
    ds["swe_obs"].attrs.update(
        {
            "long_name": "Observed surface snow water equivalent",
            "standard_name": "surface_snow_amount",
            "units": units,
        }
    )
    ds["swe_model"].attrs.update(
        {
            "long_name": "Colocated midnight model surface snow water equivalent",
            "standard_name": "surface_snow_amount",
            "units": units,
        }
    )
    ds["swe_model_equivalent"].attrs.update(
        {
            "long_name": (
                "Model-equivalent SWE derived from empirical quantile mapping"
            ),
            "standard_name": "surface_snow_amount",
            "units": units,
            "mapping_direction": "observation_to_model",
        }
    )
    ds["quantile_obs"].attrs.update(
        {
            "long_name": "Empirical percentile of the observation in F_obs",
            "units": "1",
            "plotting_position": PLOTTING_POSITION,
            "comment": (
                "Defined only when mapping_flag=0. Clipped to the supported "
                "empirical range; no tail extrapolation."
            ),
        }
    )
    ds["n_samples_mapping"].attrs.update(
        {
            "long_name": (
                "Number of paired positive SWE samples used for the mapping"
            ),
            "units": "1",
        }
    )
    ds["mapping_flag"].attrs.update(
        {
            "long_name": "Quantile-mapping status flag",
            "flag_values": FLAG_VALUES,
            "flag_meanings": FLAG_MEANINGS,
        }
    )
    ds["paired"].attrs["long_name"] = (
        "both observed SWE and midnight model SWE are finite on this date"
    )
    ds.attrs.update(
        {
            "mapping_method": "empirical_quantile_mapping",
            "mapping_direction": "observation_to_model",
            "seasonal_window_days": int(window_days),
            "cross_validation": (
                "leave_one_water_year_out" if cross_validate else "disabled"
            ),
            "min_samples": int(min_samples),
            "water_year_start_month": int(water_year_start_month),
            "plotting_position": PLOTTING_POSITION,
            "model_time_selection": "midnight",
            "n_paired": int(zeros["n_paired"]),
            "obs_zero_fraction": zeros["obs_zero_fraction"],
            "model_zero_fraction": zeros["model_zero_fraction"],
            "both_positive_fraction": zeros["both_positive_fraction"],
            "n_calendar_years": years["n_calendar_years"],
            "n_water_years": years["n_water_years"],
            "n_water_years_positive": years["n_water_years_positive"],
            "first_year": years["first_year"],
            "last_year": years["last_year"],
        }
    )
    if station_id is not None:
        ds.attrs["station_id"] = str(station_id)
    if calibration_start is not None:
        ds.attrs["calibration_start"] = str(pd.Timestamp(calibration_start))
    if calibration_end is not None:
        ds.attrs["calibration_end"] = str(pd.Timestamp(calibration_end))
    return ds


def station_record_table(
    obs: xr.DataArray,
    *,
    station_dim: str = "station_id",
    min_years: int = DEFAULT_MIN_YEARS,
    water_year_start_month: int = DEFAULT_WATER_YEAR_START_MONTH,
) -> pd.DataFrame:
    """Year-coverage screening table for every station.

    ``n_water_years`` is the number of water years with at least one finite
    observation. Stations with ``n_water_years < min_years`` fail the screen.
    """
    if station_dim not in obs.dims:
        raise ValueError(
            f"obs must have a {station_dim!r} dimension (dims={list(obs.dims)})."
        )
    rows = []
    for sid in obs[station_dim].values:
        series = obs.sel({station_dim: sid})
        stats = observation_year_coverage(
            series["time"].values,
            series.values,
            start_month=water_year_start_month,
        )
        n_wy = stats["n_water_years"]
        rows.append(
            {
                "station_id": sid,
                "n_finite": int(stats["n_finite"]),
                "n_positive": int(stats["n_positive"]),
                "n_calendar_years": int(stats["n_calendar_years"]),
                "n_water_years": int(n_wy),
                "n_water_years_positive": int(stats["n_water_years_positive"]),
                "first_year": stats["first_year"],
                "last_year": stats["last_year"],
                "passes_year_screen": bool(n_wy >= min_years),
            }
        )
    return pd.DataFrame(rows)


def combine_stations(
    ds: xr.Dataset,
    station_ids,
    *,
    new_id: str,
    new_name: str | None = None,
    station_dim: str = "station_id",
) -> xr.Dataset:
    """Merge selected stations into one record before quantile mapping.

    Finite values from either station are kept. Where two or more stations
    have a finite value on the same date, the arithmetic mean is used.
    Other dates are not interpolated or filled. Numeric station attributes
    (lat, lon, elevation) are averaged; ``station_name`` uses ``new_name``
    when given.
    """
    ids = [str(s) for s in station_ids]
    if len(ids) < 2:
        raise ValueError("combine_stations requires at least two station ids.")
    available = [str(s) for s in ds[station_dim].values]
    missing = [s for s in ids if s not in available]
    if missing:
        raise ValueError(f"Station id(s) not in dataset: {missing}.")
    if new_id in available and new_id not in ids:
        raise ValueError(f"new_id {new_id!r} already exists in {station_dim}.")

    subset = ds.sel({station_dim: ids})
    keep_ids = [s for s in available if s not in ids]
    others = ds.sel({station_dim: keep_ids}) if keep_ids else None

    combined_vars: dict[str, xr.DataArray] = {}
    for name, da in subset.data_vars.items():
        if station_dim not in da.dims:
            combined_vars[name] = da
            continue
        combined_vars[name] = _expand_combined(
            _reduce_combined_station(
                da,
                station_dim=station_dim,
                name=name,
                new_name=new_name,
            ),
            station_dim=station_dim,
            new_id=new_id,
            original_dims=da.dims,
        )

    combined = xr.Dataset(combined_vars)
    for coord_name, coord in subset.coords.items():
        if coord_name in (station_dim, "time"):
            continue
        if station_dim in coord.dims:
            combined = combined.assign_coords(
                {
                    coord_name: _expand_combined(
                        _reduce_combined_station(
                            coord,
                            station_dim=station_dim,
                            name=coord_name,
                            new_name=new_name,
                        ),
                        station_dim=station_dim,
                        new_id=new_id,
                        original_dims=coord.dims,
                    )
                }
            )
        else:
            combined = combined.assign_coords({coord_name: coord})
    combined.attrs.update(ds.attrs)
    combined.attrs["combined_from"] = ", ".join(ids)
    combined.attrs["combine_overlap"] = "arithmetic_mean"

    if others is None or others.sizes.get(station_dim, 0) == 0:
        return combined
    out = xr.concat([others, combined], dim=station_dim)
    out.attrs.update(ds.attrs)
    out.attrs["combined_from"] = ", ".join(ids)
    out.attrs["combine_overlap"] = "arithmetic_mean"
    return out


def _reduce_combined_station(
    da: xr.DataArray,
    *,
    station_dim: str,
    name: str,
    new_name: str | None,
) -> xr.DataArray:
    extra = [d for d in da.dims if d != station_dim]
    if name == "station_name" and new_name is not None:
        if extra:
            raise ValueError("station_name should be 1-D in station_id.")
        out = xr.DataArray(str(new_name))
        out.attrs.update(da.attrs)
        return out
    if np.issubdtype(da.dtype, np.number):
        out = da.mean(dim=station_dim, skipna=True)
        out.attrs.update(da.attrs)
        return out
    # Non-numeric (flags, source strings): keep the first station; no fill.
    out = da.isel({station_dim: 0}, drop=True)
    out.attrs.update(da.attrs)
    return out


def _expand_combined(
    da: xr.DataArray,
    *,
    station_dim: str,
    new_id: str,
    original_dims,
) -> xr.DataArray:
    expanded = da.expand_dims({station_dim: [new_id]})
    orig = [d for d in original_dims if d in expanded.dims]
    extra = [d for d in expanded.dims if d not in orig]
    order = orig + extra
    if list(expanded.dims) != order:
        expanded = expanded.transpose(*order)
    return expanded


def stations_to_plot(
    mapped: xr.Dataset,
    *,
    include_screened: bool = False,
    station_dim: str = "station_id",
) -> list:
    """Station ids to draw. Screened records are omitted unless requested."""
    ids = list(mapped[station_dim].values)
    if include_screened or "passes_year_screen" not in mapped:
        return ids
    passing = []
    for sid in ids:
        flag = mapped["passes_year_screen"].sel({station_dim: sid})
        if bool(np.asarray(flag).reshape(-1)[0]):
            passing.append(sid)
    return passing


def preprocess_all_stations(
    obs: xr.DataArray,
    model: xr.DataArray,
    *,
    station_dim: str = "station_id",
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_years: int = DEFAULT_MIN_YEARS,
    cross_validate: bool = True,
    water_year_start_month: int = DEFAULT_WATER_YEAR_START_MONTH,
    calibration_start=None,
    calibration_end=None,
) -> xr.Dataset:
    """Map every station independently.

    ``obs`` is ``time × station``. ``model`` is a 1-D midnight series
    colocated with every station (lumped HRU) or ``time × station``.
    Stations are never pooled.

    ``min_years`` screens on water years with at least one finite
    observation. Short records are still mapped so diagnostics remain
    available; ``passes_year_screen`` marks whether they pass.
    """
    if min_years < 1:
        raise ValueError("min_years must be >= 1.")
    if "time" not in obs.dims:
        raise ValueError(f"obs must have a time dimension (dims={list(obs.dims)}).")
    if station_dim not in obs.dims:
        raise ValueError(
            f"obs must have a {station_dim!r} dimension (dims={list(obs.dims)})."
        )

    model_by_station = None
    if station_dim in model.dims:
        model_by_station = model
    else:
        model_1d = _as_1d_time_series(model, "model")

    pieces: list[xr.Dataset] = []
    n_paired = []
    obs_zero_fraction = []
    model_zero_fraction = []
    both_positive_fraction = []
    n_calendar_years = []
    n_water_years = []
    n_water_years_positive = []
    first_year = []
    last_year = []
    station_ids = list(obs[station_dim].values)
    for sid in station_ids:
        obs_s = obs.sel({station_dim: sid})
        if model_by_station is not None:
            model_s = model_by_station.sel({station_dim: sid})
        else:
            model_s = model_1d
        mapped = preprocess_station(
            obs_s,
            model_s,
            station_id=sid,
            window_days=window_days,
            min_samples=min_samples,
            cross_validate=cross_validate,
            water_year_start_month=water_year_start_month,
            calibration_start=calibration_start,
            calibration_end=calibration_end,
        )
        n_paired.append(mapped.attrs.get("n_paired", np.nan))
        obs_zero_fraction.append(mapped.attrs.get("obs_zero_fraction", np.nan))
        model_zero_fraction.append(mapped.attrs.get("model_zero_fraction", np.nan))
        both_positive_fraction.append(mapped.attrs.get("both_positive_fraction", np.nan))
        n_calendar_years.append(mapped.attrs.get("n_calendar_years", np.nan))
        n_water_years.append(mapped.attrs.get("n_water_years", np.nan))
        n_water_years_positive.append(mapped.attrs.get("n_water_years_positive", np.nan))
        first_year.append(mapped.attrs.get("first_year", np.nan))
        last_year.append(mapped.attrs.get("last_year", np.nan))
        if station_dim in mapped.coords and station_dim not in mapped.dims:
            mapped = mapped.drop_vars(station_dim)
        mapped = mapped.expand_dims({station_dim: [sid]})
        pieces.append(mapped)

    out = xr.concat(pieces, dim=station_dim)
    for key in (
        "n_paired",
        "obs_zero_fraction",
        "model_zero_fraction",
        "both_positive_fraction",
        "n_calendar_years",
        "n_water_years",
        "n_water_years_positive",
        "first_year",
        "last_year",
        "station_id",
        "calibration_start",
        "calibration_end",
    ):
        out.attrs.pop(key, None)
    n_wy = np.asarray(n_water_years, dtype=np.float64)
    out["n_paired"] = (station_dim, np.asarray(n_paired, dtype=np.float64))
    out["obs_zero_fraction"] = (station_dim, np.asarray(obs_zero_fraction, dtype=np.float64))
    out["model_zero_fraction"] = (
        station_dim,
        np.asarray(model_zero_fraction, dtype=np.float64),
    )
    out["both_positive_fraction"] = (
        station_dim,
        np.asarray(both_positive_fraction, dtype=np.float64),
    )
    out["n_calendar_years"] = (station_dim, np.asarray(n_calendar_years, dtype=np.float64))
    out["n_water_years"] = (station_dim, n_wy)
    out["n_water_years_positive"] = (
        station_dim,
        np.asarray(n_water_years_positive, dtype=np.float64),
    )
    out["first_year"] = (station_dim, np.asarray(first_year, dtype=np.float64))
    out["last_year"] = (station_dim, np.asarray(last_year, dtype=np.float64))
    out["passes_year_screen"] = (station_dim, n_wy >= int(min_years))
    out["n_paired"].attrs["long_name"] = "Number of dates with finite obs and midnight model SWE"
    out["obs_zero_fraction"].attrs["long_name"] = "Fraction of paired dates with observed SWE == 0"
    out["model_zero_fraction"].attrs["long_name"] = "Fraction of paired dates with modeled SWE == 0"
    out["both_positive_fraction"].attrs["long_name"] = (
        "Fraction of paired dates with both observed and modeled SWE > 0"
    )
    out["n_calendar_years"].attrs["long_name"] = (
        "Calendar years with at least one finite SWE observation"
    )
    out["n_water_years"].attrs["long_name"] = (
        "Water years with at least one finite SWE observation"
    )
    out["n_water_years_positive"].attrs["long_name"] = (
        "Water years with at least one positive SWE observation"
    )
    out["first_year"].attrs["long_name"] = "First calendar year with a finite observation"
    out["last_year"].attrs["long_name"] = "Last calendar year with a finite observation"
    out["passes_year_screen"].attrs["long_name"] = (
        f"True if n_water_years >= min_years ({int(min_years)})"
    )
    out.attrs.update(
        {
            "mapping_method": "empirical_quantile_mapping",
            "mapping_direction": "observation_to_model",
            "seasonal_window_days": int(window_days),
            "cross_validation": (
                "leave_one_water_year_out" if cross_validate else "disabled"
            ),
            "min_samples": int(min_samples),
            "min_years": int(min_years),
            "water_year_start_month": int(water_year_start_month),
            "plotting_position": PLOTTING_POSITION,
            "model_time_selection": "midnight",
            "station_pooling": "none",
        }
    )
    return out
