"""Read model-equivalent SWE observations and lumped SUMMA SWE."""

from __future__ import annotations

import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from snowda.utils.logging import get_logger

log = get_logger(__name__)

_SWE_UNIT = "kg / m**2"


def units_are_swe(units: str | None) -> bool:
    """Return True when ``units`` is mass per area, or when units are absent."""
    if units is None or str(units).strip() == "":
        return True
    try:
        import pint
    except ImportError as exc:  # pragma: no cover - pint is a project dependency
        raise ImportError("pint is required to check SWE units.") from exc
    registry = pint.UnitRegistry()

    def parse(text: str):
        normalized = text.strip().replace("m-2", "m**-2").replace("m^-2", "m**-2")
        return registry(normalized).units

    try:
        return parse(str(units)) == parse(_SWE_UNIT)
    except (pint.errors.UndefinedUnitError, pint.errors.DimensionalityError):
        return False


def read_model_equivalent_swe(
    path: str | Path,
    *,
    variable: str = "swe_model_equivalent",
    flag_variable: str = "mapping_flag",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str | None]:
    """Load ``(time, station)`` model-equivalent SWE and mapping flags.

    Returns times, station ids, values, flags, and the SWE units string.
    """
    nc_path = Path(path)
    with xr.open_dataset(nc_path) as ds:
        if variable not in ds:
            raise KeyError(
                f"Variable {variable!r} not found in {nc_path}. "
                f"Available: {list(ds.data_vars)}"
            )
        if flag_variable not in ds:
            raise KeyError(
                f"Variable {flag_variable!r} not found in {nc_path}. "
                f"Available: {list(ds.data_vars)}"
            )
        swe = ds[variable]
        flags = ds[flag_variable]
        if "time" not in swe.dims or "station_id" not in swe.dims:
            raise ValueError(
                f"{variable} must have dims (time, station_id); got {swe.dims}."
            )
        swe = swe.transpose("time", "station_id")
        flags = flags.transpose("time", "station_id")
        units = swe.attrs.get("units")
        times = swe["time"].values
        station_ids = np.asarray(swe["station_id"].values)
        values = np.asarray(swe.values, dtype=float)
        flag_values = np.asarray(flags.values)
    if units is not None and not units_are_swe(str(units)):
        raise ValueError(
            f"{variable} units {units!r} are not {_SWE_UNIT}. "
            "SWE assimilation compares this field directly to SUMMA scalarSWE."
        )
    return times, station_ids, values, flag_values, None if units is None else str(units)


def _select_time_index(times: pd.DatetimeIndex, when: datetime) -> int:
    target = pd.Timestamp(when)
    exact = np.flatnonzero(times == target)
    if exact.size:
        return int(exact[0])
    same_day = np.flatnonzero(times.normalize() == target.normalize())
    if same_day.size == 0:
        raise ValueError(f"No SUMMA SWE output on {target.date()}.")
    offsets = [abs(times[index] - target) for index in same_day]
    return int(same_day[int(np.argmin(offsets))])


def read_lumped_member_swe(
    output_dir: str | Path,
    member: str,
    *,
    variable: str,
    when: datetime,
    file_suffix: str,
) -> tuple[int, float, str | None]:
    """Read one HRU's SWE for ``member`` at ``when``.

    The assimilation observation is a single basin value, so the member
    output must contain one HRU.
    """
    dated = Path(output_dir) / f"{member}_{when.strftime('%Y_%m_%d')}_{file_suffix}"
    if dated.is_file():
        files = [str(dated)]
    else:
        pattern = str(Path(output_dir) / f"{member}*{file_suffix}")
        files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No SUMMA output for member {member} matching {pattern}")
    with xr.open_mfdataset(files, combine="by_coords") as ds:
        if variable not in ds:
            raise KeyError(
                f"Variable {variable!r} not found for member {member}. "
                f"Available: {list(ds.data_vars)}"
            )
        da = ds[variable]
        if "hru" in da.dims and int(da.sizes["hru"]) != 1:
            raise ValueError(
                f"SWE assimilation expects one HRU; member {member} has "
                f"{int(da.sizes['hru'])}."
            )
        if "time" not in da.dims:
            raise ValueError(f"{variable} for member {member} has no time dimension.")
        time_index = _select_time_index(pd.DatetimeIndex(da["time"].values), when)
        selected = da.isel(time=time_index)
        if "hru" in selected.dims:
            selected = selected.isel(hru=0)
        value = float(selected.values)
        units = da.attrs.get("units")
        if "hruId" in ds:
            hru_id = int(np.asarray(ds["hruId"].values).reshape(-1)[0])
        elif "hruId" in da.coords:
            hru_id = int(np.asarray(da["hruId"].values).reshape(-1)[0])
        else:
            raise KeyError(f"Member {member} output has no hruId.")
    if units is not None and not units_are_swe(str(units)):
        raise ValueError(
            f"Member {member} {variable} units {units!r} are not {_SWE_UNIT}."
        )
    log.info("Member %s %s at %s is %s", member, variable, when, value)
    return hru_id, value, None if units is None else str(units)
