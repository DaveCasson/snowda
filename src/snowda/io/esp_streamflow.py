"""Load ESP routed ensembles, observational climatology, and return-period thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from snowda.io.streamflow import (
    DEFAULT_RESAMPLE,
    STREAMFLOW_UNIT,
    _ensure_unique_sorted_time,
    _maybe_convert_units,
    _normalize_resample,
    _resample_mean,
    read_routed_streamflow,
    read_streamflow_observations,
)
from snowda.utils.logging import get_logger

log = get_logger(__name__)

CLIMO_MEMBER_DIM = "climo_year"
ENSEMBLE_DIM = "member"
DEFAULT_RETURN_PERIODS = (5, 10, 20)
SECONDS_PER_DAY = 86400.0
VOLUME_UNIT = "m3"


@dataclass(frozen=True)
class EspPairedForecast:
    """One ESP initiation: ensemble and observations aligned on ``lead``."""

    fcst: xr.DataArray
    obs: xr.DataArray
    init_time: pd.Timestamp
    init_month: str
    n_pairs: int
    source_files: tuple[str, ...]

    @property
    def valid_time(self) -> xr.DataArray:
        return self.fcst["valid_time"]

    @property
    def n_members(self) -> int:
        return int(self.fcst.sizes.get(ENSEMBLE_DIM, 0))


def parse_state_date_key(key: str) -> pd.Timestamp:
    """Parse ESP folder names such as ``2014020100`` (YYYYMMDDHH)."""
    text = str(key).strip()
    if len(text) == 10 and text.isdigit():
        return pd.Timestamp(pd.to_datetime(text, format="%Y%m%d%H"))
    return pd.Timestamp(pd.to_datetime(text))


def init_month_key(init_time: pd.Timestamp | str) -> str:
    return pd.Timestamp(init_time).strftime("%m-%d")


def _replace_year(ts: pd.Timestamp, year: int) -> pd.Timestamp:
    try:
        return ts.replace(year=year)
    except ValueError:
        last_day = int(pd.Timestamp(year=year, month=ts.month, day=1).days_in_month)
        return ts.replace(year=year, day=min(ts.day, last_day))


def discover_esp_mizuroute_outputs(run_dir: str | Path) -> list[Path]:
    root = Path(run_dir)
    return sorted(root.rglob("esp/*/*/*/mizuroute/mizuroute_*.nc"))


def group_esp_trace_files(paths: Iterable[Path]) -> dict[str, list[Path]]:
    """Group traces by state-date folder, pooling any state-member ensembles."""
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        parts = list(path.parts)
        if "esp" not in parts:
            continue
        idx = parts.index("esp")
        try:
            state_date_key = parts[idx + 2]
        except IndexError:
            continue
        grouped.setdefault(state_date_key, []).append(path)
    return {key: sorted(files) for key, files in sorted(grouped.items())}


def _member_id(path: Path) -> str:
    parts = list(path.parts)
    idx = parts.index("esp")
    state_member = parts[idx + 1]
    trace_year = parts[idx + 3]
    return f"{state_member}_{trace_year}"


def read_esp_ensemble(
    paths: Sequence[Path],
    *,
    variable: str = "IRFroutedRunoff",
    seg: int = 0,
    reach_id: Any | None = None,
) -> xr.DataArray:
    """Read mizuRoute traces into a ``member`` × ``time`` DataArray."""
    if not paths:
        raise ValueError("No ESP mizuRoute files were provided.")
    members: list[xr.DataArray] = []
    for path in sorted(paths):
        da = read_routed_streamflow(
            path,
            variable=variable,
            seg=seg,
            reach_id=reach_id,
        )
        members.append(da.expand_dims({ENSEMBLE_DIM: [_member_id(path)]}))
    fcst = xr.concat(members, dim=ENSEMBLE_DIM)
    return fcst.rename("fcst")


def pair_ensemble_streamflow(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    resample: str | None = DEFAULT_RESAMPLE,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Resample and inner-join an ensemble with observations.

    Times with a missing observation or with no finite ensemble members are dropped.
    Individual members may remain NaN.
    """
    if ENSEMBLE_DIM not in fcst.dims:
        raise ValueError(
            f"Forecast is missing ensemble dimension {ENSEMBLE_DIM!r} "
            f"(dims={list(fcst.dims)})."
        )
    fcst = _ensure_unique_sorted_time(fcst)
    obs = _ensure_unique_sorted_time(obs)
    rule = _normalize_resample(resample)
    if rule is not None:
        fcst = _resample_mean(fcst, rule)
        obs = _resample_mean(obs, rule)
    template = fcst.isel({ENSEMBLE_DIM: 0}, drop=True)
    _, obs = _maybe_convert_units(template, obs)
    fcst.attrs["units"] = template.attrs.get("units", STREAMFLOW_UNIT)
    fcst, obs = xr.align(fcst, obs, join="inner")
    member_ok = np.isfinite(fcst).any(dim=ENSEMBLE_DIM)
    valid = np.isfinite(obs) & member_ok
    fcst = fcst.where(valid, drop=True)
    obs = obs.where(valid, drop=True)
    fcst = fcst.rename("fcst")
    obs = obs.rename("obs")
    if obs.sizes.get("time", 0) == 0:
        raise ValueError("No overlapping pairwise-complete ESP streamflow samples.")
    return fcst, obs


def assign_lead(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    init_time: pd.Timestamp,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Replace ``time`` with integer ``lead`` (days since forecast initiation)."""
    init = pd.Timestamp(init_time).normalize()
    times = pd.DatetimeIndex(pd.to_datetime(fcst["time"].values)).normalize()
    lead = np.asarray(((times - init) / pd.Timedelta(days=1)).astype(int))
    valid_time = times.to_numpy()
    fcst = fcst.assign_coords(lead=("time", lead), valid_time=("time", valid_time))
    obs = obs.assign_coords(lead=("time", lead), valid_time=("time", valid_time))
    fcst = fcst.swap_dims({"time": "lead"})
    obs = obs.swap_dims({"time": "lead"})
    if "time" in fcst.coords:
        fcst = fcst.reset_coords("time", drop=True)
    if "time" in obs.coords:
        obs = obs.reset_coords("time", drop=True)
    if pd.Index(fcst["lead"].values).duplicated().any():
        fcst = fcst.groupby("lead").mean()
        obs = obs.groupby("lead").mean()
    return fcst, obs


def observational_climatology_ensemble(
    obs: xr.DataArray,
    times: xr.DataArray | pd.DatetimeIndex | np.ndarray,
    *,
    exclude_years: Sequence[int] | None = None,
    member_dim: str = CLIMO_MEMBER_DIM,
) -> xr.DataArray:
    """Historical observations on the same calendar day as each valid time.

    Leap-day dates that do not exist in a climatology year are NaN. ``exclude_years``
    removes verification years so the verifying observation is not in the reference.
    """
    if isinstance(times, xr.DataArray):
        time_index = pd.DatetimeIndex(pd.to_datetime(times.values))
        time_dim = times.dims[0]
        time_coord = times[time_dim]
        extra_coords = {
            name: times.coords[name]
            for name in times.coords
            if name not in {time_dim, member_dim}
        }
    else:
        time_index = pd.DatetimeIndex(pd.to_datetime(times))
        time_dim = "time"
        time_coord = time_index
        extra_coords = {}

    obs_index = pd.DatetimeIndex(pd.to_datetime(obs["time"].values)).normalize()
    values = np.asarray(obs.values, dtype=float)
    lookup: dict[tuple[int, int, int], float] = {}
    for ts, value in zip(obs_index, values):
        lookup[(int(ts.year), int(ts.month), int(ts.day))] = float(value)

    exclude = {int(year) for year in (exclude_years or [])}
    years = sorted({int(ts.year) for ts in obs_index if int(ts.year) not in exclude})
    if not years:
        raise ValueError("No climatology years remain after exclusions.")

    data = np.full((time_index.size, len(years)), np.nan, dtype=float)
    for j, year in enumerate(years):
        for i, ts in enumerate(time_index):
            data[i, j] = lookup.get((year, int(ts.month), int(ts.day)), np.nan)

    coords: dict[str, Any] = {
        time_dim: time_coord,
        member_dim: np.asarray(years, dtype=int),
    }
    coords.update(extra_coords)
    climo = xr.DataArray(
        data,
        dims=(time_dim, member_dim),
        coords=coords,
        name="climo",
        attrs={
            "units": obs.attrs.get("units", STREAMFLOW_UNIT),
            "long_name": "observational streamflow climatology ensemble",
        },
    )
    return climo


def climatology_window_peaks(
    obs: xr.DataArray,
    init_time: pd.Timestamp,
    n_lead: int,
    *,
    exclude_years: Sequence[int] | None = None,
    member_dim: str = CLIMO_MEMBER_DIM,
) -> xr.DataArray:
    """Peak observed flow in the same calendar window of each climatology year."""
    start = pd.Timestamp(init_time).normalize()
    window_days = max(int(n_lead) - 1, 0)
    obs_index = pd.DatetimeIndex(pd.to_datetime(obs["time"].values))
    exclude = {int(year) for year in (exclude_years or [])}
    years = sorted({int(year) for year in obs_index.year if int(year) not in exclude})
    peaks = np.full(len(years), np.nan, dtype=float)
    for i, year in enumerate(years):
        year_start = _replace_year(start, year)
        year_end = year_start + pd.Timedelta(days=window_days)
        window = obs.sel(time=slice(year_start, year_end))
        if window.size == 0:
            continue
        finite = window.where(np.isfinite(window), drop=True)
        if finite.size:
            peaks[i] = float(finite.max())
    return xr.DataArray(
        peaks,
        dims=(member_dim,),
        coords={member_dim: np.asarray(years, dtype=int)},
        name="climo_peak",
        attrs={
            "units": obs.attrs.get("units", STREAMFLOW_UNIT),
            "long_name": "climatology peak streamflow over forecast window",
        },
    )


def peak_over_horizon(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    dim: str = "lead",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Maximum flow over the forecast window for each ensemble member and the obs."""
    if dim not in fcst.dims:
        raise ValueError(f"Forecast is missing horizon dimension {dim!r}.")
    if dim not in obs.dims:
        raise ValueError(f"Observations are missing horizon dimension {dim!r}.")
    fcst_peak = fcst.max(dim=dim, skipna=True).rename("fcst_peak")
    obs_peak = obs.max(dim=dim, skipna=True).rename("obs_peak")
    return fcst_peak, obs_peak


def volume_over_horizon(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    dim: str = "lead",
    seconds_per_step: float = SECONDS_PER_DAY,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Total water volume over the paired horizon.

    Each lead is a daily mean discharge in m³/s, so the volume is the sum of
    those means times ``seconds_per_step``. A member or observation with any
    missing lead is missing: a partial sum would undercount the horizon.
    """
    if dim not in fcst.dims:
        raise ValueError(f"Forecast is missing horizon dimension {dim!r}.")
    if dim not in obs.dims:
        raise ValueError(f"Observations are missing horizon dimension {dim!r}.")
    scale = float(seconds_per_step)
    fcst_volume = (
        (fcst.astype(float) * scale).sum(dim=dim, skipna=False).rename("fcst_volume")
    )
    obs_volume = (
        (obs.astype(float) * scale).sum(dim=dim, skipna=False).rename("obs_volume")
    )
    fcst_volume.attrs["units"] = VOLUME_UNIT
    obs_volume.attrs["units"] = VOLUME_UNIT
    fcst_volume.attrs["long_name"] = "forecast water volume over the paired horizon"
    obs_volume.attrs["long_name"] = "observed water volume over the paired horizon"
    return fcst_volume, obs_volume


def climatology_window_volumes(
    obs: xr.DataArray,
    valid_time: xr.DataArray | pd.DatetimeIndex | np.ndarray,
    *,
    exclude_years: Sequence[int] | None = None,
    member_dim: str = CLIMO_MEMBER_DIM,
    seconds_per_step: float = SECONDS_PER_DAY,
) -> xr.DataArray:
    """Observed water volume on the forecast's calendar days in each other year.

    A climatology year is missing when any of those calendar days is missing.
    A horizon that includes 29 February therefore drops non-leap years. The
    forecast and verifying observation still include that day.
    """
    if isinstance(valid_time, xr.DataArray):
        times = pd.DatetimeIndex(pd.to_datetime(valid_time.values)).normalize()
    else:
        times = pd.DatetimeIndex(pd.to_datetime(np.asarray(valid_time))).normalize()
    obs_index = pd.DatetimeIndex(pd.to_datetime(obs["time"].values)).normalize()
    values = np.asarray(obs.values, dtype=float)
    lookup: dict[tuple[int, int, int], float] = {}
    for ts, value in zip(obs_index, values):
        lookup[(int(ts.year), int(ts.month), int(ts.day))] = float(value)

    exclude = {int(year) for year in (exclude_years or [])}
    years = sorted({int(ts.year) for ts in obs_index if int(ts.year) not in exclude})
    volumes = np.full(len(years), np.nan, dtype=float)
    scale = float(seconds_per_step)
    for i, year in enumerate(years):
        day_values: list[float] = []
        for ts in times:
            value = lookup.get((year, int(ts.month), int(ts.day)), np.nan)
            if not np.isfinite(value):
                day_values = []
                break
            day_values.append(value)
        if day_values:
            volumes[i] = float(np.sum(day_values) * scale)
    return xr.DataArray(
        volumes,
        dims=(member_dim,),
        coords={member_dim: np.asarray(years, dtype=int)},
        name="climo_volume",
        attrs={
            "units": VOLUME_UNIT,
            "long_name": "climatology water volume over the forecast calendar days",
        },
    )


def read_return_period_thresholds(path: str | Path) -> xr.DataArray:
    """Read a CSV with ``T_yrs`` and ``Yp`` columns into a ``return_period`` DataArray."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    if "T_yrs" not in df.columns or "Yp" not in df.columns:
        raise ValueError(
            f"Thresholds CSV must contain 'T_yrs' and 'Yp' columns (got {list(df.columns)})."
        )
    periods = pd.to_numeric(df["T_yrs"], errors="coerce")
    yp = pd.to_numeric(df["Yp"], errors="coerce")
    valid = periods.notna() & yp.notna()
    periods = periods[valid].to_numpy(dtype=float)
    yp_values = yp[valid].to_numpy(dtype=float)
    order = np.argsort(periods)
    return xr.DataArray(
        yp_values[order],
        dims=("return_period",),
        coords={"return_period": periods[order]},
        name="yp",
        attrs={
            "units": STREAMFLOW_UNIT,
            "long_name": "return-period streamflow threshold",
            "source": str(csv_path),
        },
    )


def select_return_period_thresholds(
    thresholds: xr.DataArray,
    return_periods: Sequence[float] = DEFAULT_RETURN_PERIODS,
) -> xr.DataArray:
    missing = [
        t for t in return_periods if t not in set(thresholds["return_period"].values)
    ]
    if missing:
        available = list(thresholds["return_period"].values)
        raise KeyError(
            f"Return periods {missing} not found in thresholds (available: {available})."
        )
    return thresholds.sel(return_period=list(return_periods))


def load_esp_forecasts(
    run_dir: str | Path,
    obs_csv: str | Path,
    *,
    variable: str = "IRFroutedRunoff",
    resample: str | None = DEFAULT_RESAMPLE,
    seg: int = 0,
    reach_id: Any | None = None,
    date_col: str | None = None,
    flow_col: str | None = None,
) -> tuple[list[EspPairedForecast], xr.DataArray]:
    """Discover ESP traces, pair each initiation with observations, assign lead."""
    obs = read_streamflow_observations(
        obs_csv,
        date_col=date_col,
        flow_col=flow_col,
    )
    obs = _ensure_unique_sorted_time(obs)
    rule = _normalize_resample(resample)
    if rule is not None:
        obs = _resample_mean(obs, rule)

    paths = discover_esp_mizuroute_outputs(run_dir)
    if not paths:
        raise FileNotFoundError(f"No ESP mizuRoute outputs found in {run_dir}")
    grouped = group_esp_trace_files(paths)
    forecasts: list[EspPairedForecast] = []
    for state_date_key, files in grouped.items():
        init_time = parse_state_date_key(state_date_key)
        fcst = read_esp_ensemble(
            files,
            variable=variable,
            seg=seg,
            reach_id=reach_id,
        )
        paired_fcst, paired_obs = pair_ensemble_streamflow(
            fcst, obs, resample=None if rule is not None else resample
        )
        paired_fcst, paired_obs = assign_lead(paired_fcst, paired_obs, init_time)
        forecasts.append(
            EspPairedForecast(
                fcst=paired_fcst,
                obs=paired_obs,
                init_time=init_time,
                init_month=init_month_key(init_time),
                n_pairs=int(paired_obs.sizes.get("lead", 0)),
                source_files=tuple(str(path) for path in files),
            )
        )
        log.info(
            "Paired ESP init %s (%s): %s members, %s lead steps",
            init_time.strftime("%Y-%m-%d"),
            init_month_key(init_time),
            forecasts[-1].n_members,
            forecasts[-1].n_pairs,
        )
    forecasts.sort(key=lambda item: (item.init_month, item.init_time))
    return forecasts, obs


def group_forecasts_by_init_month(
    forecasts: Sequence[EspPairedForecast],
) -> dict[str, list[EspPairedForecast]]:
    grouped: dict[str, list[EspPairedForecast]] = {}
    for forecast in forecasts:
        grouped.setdefault(forecast.init_month, []).append(forecast)
    return dict(sorted(grouped.items()))
