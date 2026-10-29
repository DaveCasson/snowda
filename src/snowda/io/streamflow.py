"""Load and pair routed streamflow simulations with gauge observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from snowda.utils.logging import get_logger

log = get_logger(__name__)

STREAMFLOW_UNIT = "m3 s-1"
DEFAULT_RESAMPLE = "1D"
_STREAMFLOW_UNIT_ALIASES = {
    "m3s-1": STREAMFLOW_UNIT,
    "m3/s": STREAMFLOW_UNIT,
    "m^3s-1": STREAMFLOW_UNIT,
    "m^3/s": STREAMFLOW_UNIT,
    "m**3/s": STREAMFLOW_UNIT,
    "m**3s**-1": STREAMFLOW_UNIT,
    "cms": STREAMFLOW_UNIT,
}

_DATE_CANDIDATES = (
    "datetime",
    "date",
    "time",
    "timestamp",
    "Datetime",
    "DATE",
    "Date",
    "Time",
)
_FLOW_CANDIDATES = (
    "discharge_cms",
    "discharge",
    "Discharge_cms",
    "streamflow",
    "flow",
    "Q",
    "q",
)
_NODATA_SENTINELS = (-999999.0, -9999.0, -999.0, -99.0)


def _normalize_resample(rule: str | None) -> str | None:
    if rule is None or rule is False or rule == "":
        return None
    text = str(rule).strip()
    if text.lower() in ("none", "null", "false"):
        return None
    if text.upper() in ("D", "1D"):
        return "1D"
    return text


def _naive_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    if times.tz is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    times = times.rename("time")
    return times


def _mask_nodata_values(values: np.ndarray) -> np.ndarray:
    # xarray/netCDF often expose a read-only view; always copy before writing.
    out = np.array(values, dtype=float, copy=True)
    out[~np.isfinite(out)] = np.nan
    for sentinel in _NODATA_SENTINELS:
        out[np.isclose(out, sentinel, equal_nan=False)] = np.nan
    return out


def _ensure_unique_sorted_time(da: xr.DataArray) -> xr.DataArray:
    """Drop invalid timestamps, sort, and average duplicate times."""
    da = _assign_naive_time(da)
    times = pd.DatetimeIndex(da["time"].values)
    valid = np.asarray(~times.isna())
    n_bad = int((~valid).sum())
    if n_bad:
        log.warning("Dropping %s samples with invalid timestamps from %s", n_bad, da.name)
        da = da.isel(time=valid)
        times = pd.DatetimeIndex(da["time"].values)
    if da.sizes["time"] == 0:
        raise ValueError(f"No valid timestamps remain in {da.name}.")
    order = np.argsort(times.to_numpy(dtype="datetime64[ns]"), kind="stable")
    if not np.array_equal(order, np.arange(order.size)):
        da = da.isel(time=order)
        times = pd.DatetimeIndex(da["time"].values)
    if times.duplicated().any():
        n_dup = int(times.duplicated().sum())
        log.warning("Averaging %s duplicate timestamps in %s", n_dup, da.name)
        da = da.groupby("time").mean()
    da = da.copy(data=_mask_nodata_values(da.values))
    return da


def _assign_naive_time(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.dims:
        raise ValueError(f"Expected a 'time' dimension (dims={list(da.dims)}).")
    times = _naive_datetime_index(pd.Index(da["time"].values))
    return da.assign_coords(time=("time", times.to_numpy()))


def detect_date_column(df: pd.DataFrame, name: str | None = None) -> str:
    if name and name in df.columns:
        return name
    for col in _DATE_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not auto-detect observation date column. "
        f"Columns: {list(df.columns)}"
    )


def detect_flow_column(
    df: pd.DataFrame, name: str | None, date_col: str
) -> str:
    if name and name in df.columns:
        return name
    for col in _FLOW_CANDIDATES:
        if col in df.columns:
            return col
    numeric = [
        c
        for c in df.columns
        if c != date_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(numeric) == 1:
        return numeric[0]
    if df.shape[1] >= 2:
        return str(df.columns[1] if df.columns[0] == date_col else df.columns[1])
    raise ValueError(
        "Could not auto-detect observation flow column. "
        f"Columns: {list(df.columns)}"
    )


def read_streamflow_observations(
    path: str | Path,
    *,
    date_col: str | None = None,
    flow_col: str | None = None,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
) -> xr.DataArray:
    """Read a USGS-style streamflow CSV into a 1-D ``time`` DataArray."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    n_empty = int(df.isna().all(axis=1).sum())
    if n_empty:
        log.debug("Dropping %s empty observation rows in %s", n_empty, csv_path)
        df = df.dropna(how="all")
    date_col = detect_date_column(df, date_col)
    flow_col = detect_flow_column(df, flow_col, date_col)
    series = df[[date_col, flow_col]].copy()
    raw_dates = series[date_col]
    series[date_col] = pd.to_datetime(raw_dates, errors="coerce")
    n_bad_time = int(series[date_col].isna().sum())
    if n_bad_time:
        samples = (
            raw_dates[series[date_col].isna()]
            .dropna()
            .astype(str)
            .head(3)
            .tolist()
        )
        extra = f" examples={samples}" if samples else " (empty date values)"
        log.warning(
            "Dropping %s observation rows with invalid timestamps in %s%s",
            n_bad_time,
            csv_path,
            extra,
        )
        series = series.dropna(subset=[date_col])
    if series.empty:
        raise ValueError(f"No valid timestamps remain in {csv_path}.")
    series = series.set_index(date_col).sort_index()
    times = _naive_datetime_index(series.index)
    times.name = "time"
    values = _mask_nodata_values(pd.to_numeric(series[flow_col], errors="coerce").to_numpy())
    da = xr.DataArray(
        values,
        dims=("time",),
        coords={"time": ("time", times.to_numpy())},
        name="obs",
        attrs={
            "units": STREAMFLOW_UNIT,
            "long_name": "observed streamflow",
            "source": str(csv_path),
            "standard_name": "water_volume_transport_in_river_channel",
        },
    )
    da = _ensure_unique_sorted_time(da)
    return _slice_time(da, start_time, end_time)


def _select_reach(
    da: xr.DataArray,
    *,
    seg: int = 0,
    reach_id: Any | None = None,
) -> xr.DataArray:
    if reach_id is not None:
        for coord in ("reachID", "reachId", "reach_id"):
            if coord not in da.coords:
                continue
            coord_da = da[coord]
            if coord in da.xindexes:
                return da.sel({coord: reach_id}).squeeze(drop=True)
            matching = np.asarray(coord_da.values) == reach_id
            hits = np.flatnonzero(np.atleast_1d(matching))
            if hits.size == 0:
                raise ValueError(f"reach_id={reach_id!r} not found in {coord}.")
            if hits.size > 1:
                raise ValueError(
                    f"reach_id={reach_id!r} matches {hits.size} locations in {coord}."
                )
            dim = coord_da.dims[0]
            return da.isel({dim: int(hits[0])}).squeeze(drop=True)
        raise ValueError(
            f"reach_id={reach_id!r} was given but no reachID coordinate was found "
            f"(coords={list(da.coords)})."
        )
    if "seg" in da.dims:
        return da.isel(seg=seg).squeeze(drop=True)
    if "hru" in da.dims:
        return da.isel(hru=seg).squeeze(drop=True)
    return da.squeeze(drop=True)


def read_routed_streamflow(
    path: str | Path,
    *,
    variable: str = "IRFroutedRunoff",
    seg: int = 0,
    reach_id: Any | None = None,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
) -> xr.DataArray:
    """Read mizuRoute routed runoff (default ``IRFroutedRunoff``) as a 1-D series."""
    nc_path = Path(path)
    with xr.open_dataset(nc_path) as ds:
        if variable not in ds:
            raise KeyError(
                f"Variable {variable!r} not found in {nc_path}. "
                f"Available: {list(ds.data_vars)}"
            )
        da = _select_reach(ds[variable], seg=seg, reach_id=reach_id).load()
    da = _assign_naive_time(da)
    extra = [d for d in da.dims if d != "time"]
    if extra:
        raise ValueError(
            f"Streamflow still has extra dimensions {extra}; "
            "specify seg or reach_id to select a single reach."
        )
    units = da.attrs.get("units") or STREAMFLOW_UNIT
    da = da.rename("sim")
    da.attrs["units"] = units
    da.attrs.setdefault("long_name", "simulated streamflow")
    da.attrs["source"] = str(nc_path)
    da.attrs["standard_name"] = "water_volume_transport_in_river_channel"
    return _slice_time(da, start_time, end_time)


def _slice_time(
    da: xr.DataArray,
    start_time: str | pd.Timestamp | None,
    end_time: str | pd.Timestamp | None,
) -> xr.DataArray:
    if start_time is None and end_time is None:
        return da
    start = pd.to_datetime(start_time) if start_time is not None else None
    end = pd.to_datetime(end_time) if end_time is not None else None
    return da.sel(time=slice(start, end))


def _resample_mean(da: xr.DataArray, rule: str) -> xr.DataArray:
    resampled = da.resample(time=rule).mean()
    resampled.attrs.update(da.attrs)
    resampled.name = da.name
    return resampled


def _time_span(da: xr.DataArray) -> str:
    n = int(da.sizes.get("time", 0))
    if n == 0:
        return "empty"
    start = pd.Timestamp(da["time"].values[0])
    end = pd.Timestamp(da["time"].values[-1])
    return f"{start} to {end} (n={n})"


def _canonical_streamflow_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    compact = str(unit).strip().lower().replace(" ", "")
    return _STREAMFLOW_UNIT_ALIASES.get(compact, str(unit).strip())


def _maybe_convert_units(
    sim: xr.DataArray, obs: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    sim = sim.copy()
    obs = obs.copy()
    sim_u = _canonical_streamflow_unit(sim.attrs.get("units"))
    obs_u = _canonical_streamflow_unit(obs.attrs.get("units"))
    if not sim_u:
        sim.attrs["units"] = STREAMFLOW_UNIT
        log.warning(
            "Simulation has no units attribute; assuming %s", STREAMFLOW_UNIT
        )
        sim_u = STREAMFLOW_UNIT
    if not obs_u:
        obs.attrs["units"] = STREAMFLOW_UNIT
        log.warning(
            "Observations have no units attribute; assuming %s", STREAMFLOW_UNIT
        )
        obs_u = STREAMFLOW_UNIT
    if sim_u == obs_u:
        sim.attrs["units"] = sim_u
        obs.attrs["units"] = obs_u
        return sim, obs
    try:
        from pint import UnitRegistry

        ureg = UnitRegistry()
        converted = (obs.values * ureg(str(obs_u))).to(str(sim_u)).magnitude
        obs = obs.copy(data=np.asarray(converted, dtype=float))
        obs.attrs["units"] = sim_u
        sim.attrs["units"] = sim_u
        log.info("Units: obs: %s -> %s", obs_u, sim_u)
    except Exception as exc:
        log.warning(
            "Could not convert obs units %s to sim units %s (%s); scoring as-is",
            obs_u,
            sim_u,
            exc,
        )
    return sim, obs


def pair_streamflow(
    sim: xr.DataArray,
    obs: xr.DataArray,
    *,
    resample: str | None = DEFAULT_RESAMPLE,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Resample, time-slice, inner-join, and drop pairwise NaNs."""
    sim = _ensure_unique_sorted_time(sim)
    obs = _ensure_unique_sorted_time(obs)
    sim = _slice_time(sim, start_time, end_time)
    obs = _slice_time(obs, start_time, end_time)
    rule = _normalize_resample(resample)
    if rule is not None:
        sim = _resample_mean(sim, rule)
        obs = _resample_mean(obs, rule)
    sim, obs = _maybe_convert_units(sim, obs)
    sim_span = _time_span(sim)
    obs_span = _time_span(obs)
    sim, obs = xr.align(sim, obs, join="inner")
    n_aligned = int(sim.sizes.get("time", 0))
    valid = np.isfinite(sim.values) & np.isfinite(obs.values)
    sim = sim.isel(time=valid)
    obs = obs.isel(time=valid)
    sim = sim.rename("sim")
    obs = obs.rename("obs")
    if sim.size == 0:
        raise ValueError(
            "No overlapping pairwise-complete streamflow samples. "
            f"sim time {sim_span}; obs time {obs_span}; "
            f"aligned times={n_aligned}."
        )
    log.info(
        "Paired streamflow: %s samples from %s to %s (resample=%s)",
        int(sim.size),
        sim["time"].min().values,
        sim["time"].max().values,
        rule,
    )
    return sim, obs


def load_paired_streamflow(
    sim_nc: str | Path,
    obs_csv: str | Path,
    *,
    variable: str = "IRFroutedRunoff",
    seg: int = 0,
    reach_id: Any | None = None,
    resample: str | None = DEFAULT_RESAMPLE,
    date_col: str | None = None,
    flow_col: str | None = None,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Load mizuRoute output and a streamflow CSV, then pair them."""
    sim = read_routed_streamflow(
        sim_nc,
        variable=variable,
        seg=seg,
        reach_id=reach_id,
        start_time=start_time,
        end_time=end_time,
    )
    obs = read_streamflow_observations(
        obs_csv,
        date_col=date_col,
        flow_col=flow_col,
        start_time=start_time,
        end_time=end_time,
    )
    return pair_streamflow(
        sim,
        obs,
        resample=resample,
        start_time=None,
        end_time=None,
    )
