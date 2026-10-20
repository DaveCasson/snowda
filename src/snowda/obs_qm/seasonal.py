"""Seasonal windows and water-year helpers for SWE quantile mapping.

Snow SWE distributions are strongly seasonal, so empirical CDFs are
estimated in a moving day-of-year window rather than from the full year.

The window uses circular calendar day-of-year distance

    min(|d1 - d2|, 366 - |d1 - d2|)

so a window around 1 October includes late September. Water year is used
for leave-one-year-out exclusion, not for the window itself.

Default water year starts on 1 October: 1 Oct YYYY belongs to water year
YYYY+1.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_WINDOW_DAYS = 30
DEFAULT_WATER_YEAR_START_MONTH = 10


def water_year(times, start_month: int = DEFAULT_WATER_YEAR_START_MONTH) -> np.ndarray:
    """Return the water year for each timestamp.

    Dates in ``start_month`` through December belong to the following
    calendar year's water year. With the default October start,
    2015-10-01 is water year 2016 and 2015-09-30 is water year 2015.
    """
    if not 1 <= int(start_month) <= 12:
        raise ValueError(f"start_month must be in 1..12 (got {start_month}).")
    index = pd.DatetimeIndex(times)
    year = index.year.to_numpy()
    month = index.month.to_numpy()
    return np.where(month >= start_month, year + 1, year).astype(np.int32)


def water_year_day(
    times,
    start_month: int = DEFAULT_WATER_YEAR_START_MONTH,
) -> np.ndarray:
    """Return day of water year (1 October = 1 with the default start).

    Calendar day-of-year splits winter across day 1 and day 365, which makes
    February–April surveys look like a mid-year blob. Water-year day keeps
    the snow season contiguous (October through May) and puts summer at
    the end of the axis.
    """
    index = pd.DatetimeIndex(times)
    wy = water_year(index, start_month=start_month)
    starts = pd.DatetimeIndex(
        [pd.Timestamp(year=int(year) - 1, month=int(start_month), day=1) for year in wy]
    )
    return ((index - starts).days + 1).to_numpy().astype(np.int32)


def observation_year_coverage(
    times,
    obs,
    start_month: int = DEFAULT_WATER_YEAR_START_MONTH,
) -> dict[str, float]:
    """Count calendar years and water years with at least one finite observation."""
    index = pd.DatetimeIndex(times)
    values = np.asarray(obs, dtype=np.float64)
    if values.shape != (index.size,):
        raise ValueError("times and obs must be 1-D arrays of the same length.")
    finite = np.isfinite(values)
    positive = finite & (values > 0.0)
    if not finite.any():
        return {
            "n_finite": 0.0,
            "n_positive": 0.0,
            "n_calendar_years": 0.0,
            "n_water_years": 0.0,
            "n_water_years_positive": 0.0,
            "first_year": np.nan,
            "last_year": np.nan,
            "first_water_year": np.nan,
            "last_water_year": np.nan,
        }
    wy = water_year(index, start_month=start_month)
    return {
        "n_finite": float(finite.sum()),
        "n_positive": float(positive.sum()),
        "n_calendar_years": float(np.unique(index[finite].year).size),
        "n_water_years": float(np.unique(wy[finite]).size),
        "n_water_years_positive": float(np.unique(wy[positive]).size) if positive.any() else 0.0,
        "first_year": float(index[finite].year.min()),
        "last_year": float(index[finite].year.max()),
        "first_water_year": float(wy[finite].min()),
        "last_water_year": float(wy[finite].max()),
    }


def circular_doy_distance(doy: np.ndarray, target_doy: int) -> np.ndarray:
    """Circular day-of-year distance using a 366-day wraparound."""
    raw = np.abs(np.asarray(doy, dtype=np.int64) - int(target_doy))
    return np.minimum(raw, 366 - raw)


def in_seasonal_window(
    times,
    target_date,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> np.ndarray:
    """Boolean mask of timestamps within ±window_days of target day-of-year."""
    if window_days < 0:
        raise ValueError("window_days must be >= 0.")
    index = pd.DatetimeIndex(times)
    target = pd.Timestamp(target_date)
    target_doy = int(target.dayofyear)
    distance = circular_doy_distance(index.dayofyear.to_numpy(), target_doy)
    return distance <= int(window_days)


def get_seasonal_samples(
    times,
    obs,
    model,
    target_date,
    window_days: int = DEFAULT_WINDOW_DAYS,
    exclude_water_year: Optional[int] = None,
    water_year_start_month: int = DEFAULT_WATER_YEAR_START_MONTH,
    calibration_start=None,
    calibration_end=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired finite obs/model samples in the seasonal window.

    Only dates where **both** observed and modeled SWE are finite are
    returned. Missing values are not interpolated or filled. The two
    arrays share the same dates; the caller sorts them independently
    when forming empirical CDFs.

    If ``exclude_water_year`` is set, every sample from that water year
    is dropped (leave-one-water-year-out). Optional ``calibration_start``
    / ``calibration_end`` further restrict the pool (inclusive).
    """
    index = pd.DatetimeIndex(times)
    obs_vals = np.asarray(obs, dtype=np.float64)
    model_vals = np.asarray(model, dtype=np.float64)
    if obs_vals.shape != model_vals.shape or obs_vals.shape != (index.size,):
        raise ValueError(
            "times, obs, and model must be 1-D arrays of the same length "
            f"(got times={index.size}, obs={obs_vals.shape}, model={model_vals.shape})."
        )

    mask = in_seasonal_window(index, target_date, window_days=window_days)
    mask &= np.isfinite(obs_vals) & np.isfinite(model_vals)

    if calibration_start is not None:
        mask &= index >= pd.Timestamp(calibration_start)
    if calibration_end is not None:
        mask &= index <= pd.Timestamp(calibration_end)

    if exclude_water_year is not None:
        wy = water_year(index, start_month=water_year_start_month)
        mask &= wy != int(exclude_water_year)

    return obs_vals[mask], model_vals[mask]


def positive_paired_samples(
    obs_samples: np.ndarray,
    model_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep samples where both observed and modeled SWE are strictly positive."""
    obs_vals = np.asarray(obs_samples, dtype=np.float64)
    model_vals = np.asarray(model_samples, dtype=np.float64)
    keep = (obs_vals > 0.0) & (model_vals > 0.0)
    return obs_vals[keep], model_vals[keep]


def water_year_axis(
    start_month: int = DEFAULT_WATER_YEAR_START_MONTH,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Reference non-leap water year (365 days) for seasonal diagnostic axes."""
    start = pd.Timestamp(year=2001, month=int(start_month), day=1)
    ref = pd.date_range(start, periods=365, freq="D")
    return ref, water_year_day(ref, start_month=start_month)


def calibration_sample_count_by_water_year_day(
    times,
    obs,
    model,
    window_days: int = DEFAULT_WINDOW_DAYS,
    water_year_start_month: int = DEFAULT_WATER_YEAR_START_MONTH,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Count positive paired samples in the ±window around each water-year day.

    Uses the same circular *calendar* day-of-year window as the mapper, but
    the returned x-axis is water-year day so October–May is contiguous.
    """
    index = pd.DatetimeIndex(times)
    obs_vals = np.asarray(obs, dtype=np.float64)
    model_vals = np.asarray(model, dtype=np.float64)
    keep = (
        np.isfinite(obs_vals)
        & np.isfinite(model_vals)
        & (obs_vals > 0.0)
        & (model_vals > 0.0)
    )
    sample_doy = index.dayofyear.to_numpy()[keep]
    ref, wyd = water_year_axis(start_month=water_year_start_month)
    target_doy = ref.dayofyear.to_numpy()
    counts = np.array(
        [
            int(np.sum(circular_doy_distance(sample_doy, int(day)) <= int(window_days)))
            for day in target_doy
        ],
        dtype=np.int32,
    )
    return wyd, counts, ref
