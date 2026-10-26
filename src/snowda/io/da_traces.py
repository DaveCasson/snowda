"""Load historic DA traces, assimilation states, and ESP forecasts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from snowda.io.esp_streamflow import (
    ENSEMBLE_DIM,
    group_esp_trace_files,
    init_month_key,
    parse_state_date_key,
)
from snowda.utils.logging import get_logger

log = get_logger(__name__)

_HISTORIC_FILE_RE = re.compile(
    r"^(?P<member>.+?)_(?P<stamp>\d{4}_\d{2}_\d{2})_timestep\.nc$"
)
_MIZUROUTE_FILE_RE = re.compile(
    r"^(?P<member>.+?)_(?P<stamp>\d{4}_\d{2}_\d{2})_mizuroute\.nc$"
)
_DETERMINISTIC_MIZU_RE = re.compile(r"^mizuroute_deterministic\.nc$")
_SPECIFIC_DISCHARGE_UNITS = {"m/s", "ms-1", "ms**-1", "m/sec", "meter/second"}
_MIN_DAY_FRACTION = 0.5


@dataclass(frozen=True)
class HistoricMemberTrace:
    """One historic ensemble member, split at assimilation-segment boundaries."""

    member: str
    segments: tuple[xr.DataArray, ...]


@dataclass(frozen=True)
class AssimilationState:
    """Ensemble values at one assimilation time, plus the assimilated observation."""

    time: pd.Timestamp
    values: xr.DataArray
    observation: float | None = None


@dataclass(frozen=True)
class DaForecast:
    """ESP traces issued from one assimilation state."""

    init_time: pd.Timestamp
    init_month: str
    values: xr.DataArray


@dataclass(frozen=True)
class DaTraceSet:
    """Historic traces, assimilation states, and forecasts for one variable."""

    historic: tuple[HistoricMemberTrace, ...]
    states: tuple[AssimilationState, ...]
    forecasts: tuple[DaForecast, ...]
    units: str
    ylabel: str


def load_da_swe(run_dir: str | Path) -> DaTraceSet:
    """Daily SWE for the historic ensemble, update times, and ESP forecasts.

    Historic series and assimilation states come from the particle-filter
    SUMMA outputs. Forecasts come from the year-shifted ESP SUMMA traces.
    State markers are the instantaneous member SWE at the assimilation time.
    """
    return _load_da_traces(
        Path(run_dir),
        historic_variable="scalarSWE",
        forecast_paths=_discover_shifted_summa(Path(run_dir)),
        forecast_variable="scalarSWE",
        units="kg m-2",
        ylabel="SWE (kg m-2)",
        include_swe_observation=True,
    )


def load_da_streamflow(
    run_dir: str | Path,
    *,
    topology_nc: str | Path | None = None,
) -> DaTraceSet:
    """Daily discharge for the historic ensemble, update times, and ESP forecasts.

    When the run routed each SUMMA segment, historic traces are mizuRoute
    ``IRFroutedRunoff`` from ``<member>_<date>_mizuroute.nc`` or from
    ``mizuroute_deterministic.nc``. Otherwise historic traces are SUMMA
    ``averageRoutedRunoff``, and specific discharge (m s-1) is multiplied by
    the mizuRoute HRU area. Area comes from an ESP or interval
    ``topology.nc``, then ``<run_dir>/mizuroute/topology.nc``. ``topology_nc``
    supplies that area when none of those files exist.
    """
    root = Path(run_dir)
    routed = _historic_mizuroute_files(root)
    if routed:
        deterministic = all(path.name == "mizuroute_deterministic.nc" for path in routed)
        return _load_da_traces(
            root,
            historic_variable="IRFroutedRunoff",
            forecast_paths=_discover_mizuroute(root),
            forecast_variable="IRFroutedRunoff",
            units="m3 s-1",
            ylabel="Streamflow (m3 s-1)",
            include_swe_observation=False,
            historic_dir=root / "mizuroute",
            historic_glob=(
                "mizuroute_deterministic.nc" if deterministic else "*_mizuroute.nc"
            ),
            historic_pattern=(
                _DETERMINISTIC_MIZU_RE if deterministic else _MIZUROUTE_FILE_RE
            ),
        )
    area = _basin_area_m2(root, topology_nc)

    def _to_discharge(da: xr.DataArray) -> xr.DataArray:
        if not _is_specific_discharge(da.attrs.get("units")):
            return da
        converted = da * area
        converted.attrs = dict(da.attrs)
        converted.attrs["units"] = "m3 s-1"
        converted.name = da.name
        return converted

    return _load_da_traces(
        root,
        historic_variable="averageRoutedRunoff",
        forecast_paths=_discover_mizuroute(root),
        forecast_variable="IRFroutedRunoff",
        units="m3 s-1",
        ylabel="Streamflow (m3 s-1)",
        include_swe_observation=False,
        transform=_to_discharge,
    )


def _identity(da: xr.DataArray) -> xr.DataArray:
    return da


def _load_da_traces(
    run_dir: Path,
    *,
    historic_variable: str,
    forecast_paths: list[Path],
    forecast_variable: str,
    units: str,
    ylabel: str,
    include_swe_observation: bool,
    transform=None,
    historic_dir: Path | None = None,
    historic_glob: str = "*_timestep.nc",
    historic_pattern: re.Pattern[str] | None = None,
) -> DaTraceSet:
    if transform is None:
        transform = _identity
    historic, raw_by_member = _read_historic(
        run_dir,
        historic_variable,
        transform,
        source_dir=historic_dir,
        file_glob=historic_glob,
        file_pattern=historic_pattern,
    )
    dated_observations = _read_swe_observations(run_dir)
    observations = dated_observations if include_swe_observation else {}
    times = _assimilation_times(run_dir, dated_observations)
    states = _states_at_times(raw_by_member, times, observations)
    forecasts = _read_forecasts(forecast_paths, forecast_variable, transform)
    if not historic and not forecasts:
        raise FileNotFoundError(f"No historic traces or ESP forecasts found in {run_dir}.")
    log.info(
        "Loaded DA traces from %s: %s historic members, %s states, %s forecasts",
        run_dir,
        len(historic),
        len(states),
        len(forecasts),
    )
    return DaTraceSet(
        historic=tuple(historic),
        states=tuple(states),
        forecasts=tuple(forecasts),
        units=units,
        ylabel=ylabel,
    )


def _historic_mizuroute_files(run_dir: Path) -> list[Path]:
    folder = run_dir / "mizuroute"
    if not folder.is_dir():
        return []
    member_files = [
        path
        for path in sorted(folder.glob("*_mizuroute.nc"))
        if _MIZUROUTE_FILE_RE.match(path.name)
    ]
    if member_files:
        return member_files
    deterministic = folder / "mizuroute_deterministic.nc"
    if deterministic.is_file():
        return [deterministic]
    return []


def _read_historic(
    run_dir: Path,
    variable: str,
    transform,
    *,
    source_dir: Path | None = None,
    file_glob: str = "*_timestep.nc",
    file_pattern: re.Pattern[str] | None = None,
) -> tuple[list[HistoricMemberTrace], dict[str, list[tuple[pd.Timestamp, xr.DataArray]]]]:
    folder = source_dir if source_dir is not None else run_dir / "summa"
    pattern = file_pattern or _HISTORIC_FILE_RE
    if not folder.is_dir():
        raise FileNotFoundError(f"Historic output directory not found: {folder}")
    grouped: dict[str, list[Path]] = {}
    for path in sorted(folder.glob(file_glob)):
        match = pattern.match(path.name)
        if match is None:
            continue
        member = match.groupdict().get("member") or "001"
        grouped.setdefault(member, []).append(path)
    if not grouped:
        raise FileNotFoundError(f"No historic trace files in {folder}.")

    traces: list[HistoricMemberTrace] = []
    raw_by_member: dict[str, list[tuple[pd.Timestamp, xr.DataArray]]] = {}
    for member, paths in sorted(grouped.items()):
        segments: list[xr.DataArray] = []
        raw_segments: list[tuple[pd.Timestamp, xr.DataArray]] = []
        for path in sorted(paths, key=lambda item: item.name):
            match = pattern.match(path.name)
            if match is None:
                continue
            raw = transform(_read_series(path, variable))
            stamp = match.groupdict().get("stamp")
            if stamp:
                segment_end = pd.Timestamp(stamp.replace("_", "-"))
            else:
                segment_end = pd.Timestamp(raw["time"].values[-1])
            raw_segments.append((segment_end, raw))
            plotted = _append_end_instant(_to_daily(raw), raw)
            if int(plotted.sizes.get("time", 0)) == 0:
                continue
            plotted.name = variable
            segments.append(plotted)
        if segments:
            traces.append(HistoricMemberTrace(member=member, segments=tuple(segments)))
        raw_by_member[member] = raw_segments
    return traces, raw_by_member


def _states_at_times(
    raw_by_member: dict[str, list[tuple[pd.Timestamp, xr.DataArray]]],
    times: list[pd.Timestamp],
    observations: dict[pd.Timestamp, float],
) -> list[AssimilationState]:
    states: list[AssimilationState] = []
    for when in times:
        member_ids: list[str] = []
        values: list[float] = []
        for member, segments in raw_by_member.items():
            value = _value_at(segments, when)
            if value is None:
                continue
            member_ids.append(member)
            values.append(value)
        if not member_ids and when not in observations:
            continue
        da = xr.DataArray(
            np.asarray(values, dtype=float),
            dims=(ENSEMBLE_DIM,),
            coords={ENSEMBLE_DIM: member_ids},
            name="state",
        )
        states.append(
            AssimilationState(
                time=when,
                values=da,
                observation=observations.get(when),
            )
        )
    return states


def _value_at(
    segments: list[tuple[pd.Timestamp, xr.DataArray]],
    when: pd.Timestamp,
) -> float | None:
    target = pd.Timestamp(when)
    for segment_end, raw in segments:
        if segment_end.normalize() != target.normalize():
            continue
        times = pd.DatetimeIndex(raw["time"].values)
        if times.empty:
            continue
        nearest = int(np.argmin(np.abs(times - target)))
        if abs(times[nearest] - target) > pd.Timedelta(hours=12):
            continue
        return float(raw.isel(time=nearest).values)
    return None


def _read_forecasts(
    paths: list[Path],
    variable: str,
    transform,
) -> list[DaForecast]:
    grouped = group_esp_trace_files(paths)
    forecasts: list[DaForecast] = []
    for state_date_key, files in grouped.items():
        init_time = parse_state_date_key(state_date_key)
        members: list[xr.DataArray] = []
        for path in files:
            series = _to_daily(transform(_read_series(path, variable)))
            if int(series.sizes.get("time", 0)) == 0:
                continue
            members.append(series.expand_dims({ENSEMBLE_DIM: [_trace_member_id(path)]}))
        if not members:
            continue
        forecasts.append(
            DaForecast(
                init_time=init_time,
                init_month=init_month_key(init_time),
                values=_concat_members(members),
            )
        )
    forecasts.sort(key=lambda item: item.init_time)
    return forecasts


def _concat_members(members: list[xr.DataArray]) -> xr.DataArray:
    times = pd.DatetimeIndex([])
    for da in members:
        times = times.union(pd.DatetimeIndex(da["time"].values))
    times = times.sort_values()
    aligned = [da.reindex(time=times.to_numpy()) for da in members]
    return xr.concat(aligned, dim=ENSEMBLE_DIM)


def _discover_shifted_summa(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("esp/*/*/*/mizuroute/summa_*.nc"))


def _discover_mizuroute(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("esp/*/*/*/mizuroute/mizuroute_*.nc"))


def _trace_member_id(path: Path) -> str:
    parts = list(path.parts)
    idx = parts.index("esp")
    return f"{parts[idx + 1]}_{parts[idx + 3]}"


def _basin_area_m2(run_dir: Path, topology_nc: str | Path | None = None) -> float:
    matches = sorted(run_dir.glob("esp/*/*/*/mizuroute/topology.nc"))
    if not matches:
        matches = sorted(run_dir.glob("esp_intervals/*/*/mizuroute/topology.nc"))
    if not matches:
        local = run_dir / "mizuroute" / "topology.nc"
        if local.is_file():
            matches = [local]
    if not matches:
        matches = sorted(run_dir.glob("mizuroute/segments/*/*/topology.nc"))
    if not matches and topology_nc is not None:
        candidate = Path(topology_nc)
        if not candidate.is_file():
            raise FileNotFoundError(
                f"No mizuRoute topology.nc under {run_dir}, and {candidate} does not exist."
            )
        matches = [candidate]
    if not matches:
        raise FileNotFoundError(
            f"No mizuRoute topology.nc under {run_dir}. "
            "Streamflow plots need HRU area to convert SUMMA runoff to m3 s-1."
        )
    with xr.open_dataset(matches[0]) as ds:
        if "area" not in ds:
            raise KeyError(f"{matches[0]} has no 'area' variable.")
        area = float(np.nansum(np.asarray(ds["area"].values, dtype=float)))
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError(f"HRU area in {matches[0]} is {area}.")
    log.info("Using HRU area %s m2 from %s", area, matches[0])
    return area


def _is_specific_discharge(units: str | None) -> bool:
    if units is None:
        return False
    compact = str(units).strip().lower().replace(" ", "")
    return compact in _SPECIFIC_DISCHARGE_UNITS


def _read_swe_observations(run_dir: Path) -> dict[pd.Timestamp, float]:
    obs_dir = run_dir / "assimilation"
    if not obs_dir.is_dir():
        return {}
    observations: dict[pd.Timestamp, float] = {}
    for path in sorted(obs_dir.glob("*_swe_obs.csv")):
        frame = pd.read_csv(path)
        if frame.empty or "lumped_mean" not in frame.columns:
            continue
        if "assimilation_date" in frame.columns:
            when = pd.Timestamp(frame["assimilation_date"].iloc[0])
        else:
            stamp = path.name[:10].replace("_", "-")
            when = pd.Timestamp(stamp)
        observations[when] = float(frame["lumped_mean"].iloc[0])
    return observations


def _assimilation_times(
    run_dir: Path,
    observations: dict[pd.Timestamp, float],
) -> list[pd.Timestamp]:
    if observations:
        return sorted(observations)
    provenance = run_dir / "provenance.json"
    if not provenance.is_file():
        return []
    payload = json.loads(provenance.read_text())
    dates = payload.get("assimilation_dates") or []
    return sorted({pd.Timestamp(value) for value in dates})


def _read_series(path: Path, variable: str) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise KeyError(
                f"Variable {variable!r} not found in {path}. "
                f"Available: {list(ds.data_vars)}"
            )
        da = ds[variable].load()
    extra = [dim for dim in da.dims if dim != "time"]
    for dim in extra:
        size = int(da.sizes[dim])
        if size != 1:
            raise ValueError(
                f"{variable} dimension {dim!r} has size {size} in {path}; "
                "DA plots expect a single HRU or reach."
            )
        da = da.isel({dim: 0}, drop=True)
    if "time" not in da.dims:
        raise ValueError(f"{variable} in {path} has no time dimension.")
    times = pd.to_datetime(pd.Index(da["time"].values))
    if getattr(times, "tz", None) is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    da = da.assign_coords(time=("time", pd.DatetimeIndex(times).to_numpy()))
    da.attrs["source"] = str(path)
    return da


def _to_daily(da: xr.DataArray) -> xr.DataArray:
    """Mean within each calendar day, dropping days with too few samples."""
    times = pd.DatetimeIndex(da["time"].values)
    if times.size < 2:
        return da
    step = pd.Series(times).diff().median()
    if pd.isna(step) or step >= pd.Timedelta(hours=20):
        return da
    counted = da.resample(time="1D").count()
    averaged = da.resample(time="1D").mean()
    expected = pd.Timedelta(days=1) / step
    minimum = max(int(expected * _MIN_DAY_FRACTION), 1)
    averaged = averaged.where(counted >= minimum, drop=True)
    averaged.attrs.update(da.attrs)
    averaged.name = da.name
    return averaged


def _append_end_instant(daily: xr.DataArray, raw: xr.DataArray) -> xr.DataArray:
    """Keep the segment-end sample when daily averaging dropped that calendar day.

    SUMMA segment files end at 00:00 on the assimilation date, so that day has
    a single sample. The instantaneous value is the state the filter used.
    """
    if int(raw.sizes.get("time", 0)) == 0:
        return daily
    last_time = pd.Timestamp(raw["time"].values[-1])
    if int(daily.sizes.get("time", 0)):
        daily_days = set(pd.DatetimeIndex(daily["time"].values).normalize())
        if last_time.normalize() in daily_days:
            return daily
    instant = xr.DataArray(
        [float(raw.isel(time=-1).values)],
        dims=("time",),
        coords={"time": [np.datetime64(last_time.to_datetime64())]},
        name=raw.name,
        attrs=dict(raw.attrs),
    )
    if int(daily.sizes.get("time", 0)) == 0:
        return instant
    return xr.concat([daily, instant], dim="time")
