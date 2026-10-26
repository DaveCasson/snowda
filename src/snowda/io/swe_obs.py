"""Read model-equivalent SWE observations and lumped SUMMA SWE."""

from __future__ import annotations

import glob
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from snowda.utils.logging import get_logger

log = get_logger(__name__)


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    try:
        import json
        import time

        payload = {
            "sessionId": "83d907",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open("/Users/dcasson/GitHub/snowda/.cursor/debug-83d907.log", "a") as handle:
            handle.write(json.dumps(payload) + "\n")
    except Exception:
        pass
    # #endregion

_SWE_UNIT = "kg / m**2"
# Yukon snow-survey SWE is stored in mm. One millimeter of water equivalent
# has the same magnitude as 1 kg m-2, so those values compare directly to
# SUMMA scalarSWE.
_MM_SWE_UNITS = frozenset(
    {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}
)


def units_are_swe(units: str | None) -> bool:
    """Return True when ``units`` match SUMMA SWE, or when units are absent.

    Accepted units are mass per area (kg m-2) and millimeters of water
    equivalent. Other length units are rejected.
    """
    if units is None or str(units).strip() == "":
        return True
    if str(units).strip().lower() in _MM_SWE_UNITS:
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
            f"{variable} units {units!r} are not {_SWE_UNIT} or mm. "
            "SWE assimilation compares this field directly to SUMMA scalarSWE."
        )
    return times, station_ids, values, flag_values, None if units is None else str(units)


@contextmanager
def _open_member_outputs(files: list[str]) -> Iterator[xr.Dataset]:
    """Open member output on this thread.

    ``open_mfdataset`` hands the read to dask. A notebook kernel that already
    has a distributed client then blocks forever on the first ``.values`` call.
    """
    opened = [xr.open_dataset(path) for path in files]
    combined = opened[0]
    if len(opened) > 1:
        combined = xr.concat(
            opened,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )
    try:
        yield combined
    finally:
        if combined is not opened[0]:
            combined.close()
        for dataset in opened:
            dataset.close()


def _member_output_files(
    output_dir: str | Path,
    member: str,
    when: datetime,
    file_suffix: str,
) -> list[str]:
    dated = Path(output_dir) / f"{member}_{when.strftime('%Y_%m_%d')}_{file_suffix}"
    if dated.is_file():
        return [str(dated)]
    pattern = str(Path(output_dir) / f"{member}*{file_suffix}")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No SUMMA output for member {member} matching {pattern}"
        )
    return files


def _hru_id_array(ds: xr.Dataset, da: xr.DataArray, member: str) -> np.ndarray:
    if "hruId" in ds:
        return np.asarray(ds["hruId"].values).reshape(-1)
    if "hruId" in da.coords:
        return np.asarray(da["hruId"].values).reshape(-1)
    raise KeyError(f"Member {member} output has no hruId.")


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

    The assimilation observation is compared with the HRU named in the
    config. This helper still requires the member file to contain one HRU.
    """
    files = _member_output_files(output_dir, member, when, file_suffix)
    with _open_member_outputs(files) as ds:
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
        hru_ids = _hru_id_array(ds, da, member)
        hru_id = int(hru_ids[0])
    if units is not None and not units_are_swe(str(units)):
        raise ValueError(
            f"Member {member} {variable} units {units!r} are not {_SWE_UNIT} or mm."
        )
    log.info("Member %s %s at %s is %s", member, variable, when, value)
    return hru_id, value, None if units is None else str(units)


def read_member_hru_swe(
    output_dir: str | Path,
    member: str,
    *,
    variable: str,
    when: datetime,
    file_suffix: str,
    hru_ids: Sequence[int],
) -> dict[int, float]:
    """Read SWE for the requested SUMMA ``hruId`` values at ``when``."""
    requested = [int(hru_id) for hru_id in hru_ids]
    if not requested:
        raise ValueError("No HRU ids were requested.")
    files = _member_output_files(output_dir, member, when, file_suffix)
    # #region agent log
    dask_client = None
    try:
        from distributed import get_client

        dask_client = str(get_client(timeout=0.2))
    except Exception as exc:
        dask_client = f"{type(exc).__name__}: {exc}"
    _agent_log(
        "B",
        "swe_obs.py:read_member_hru_swe",
        "opening member swe",
        {
            "member": member,
            "when": str(when),
            "files": files,
            "dask_client": dask_client,
        },
    )
    # #endregion
    with _open_member_outputs(files) as ds:
        if variable not in ds:
            raise KeyError(
                f"Variable {variable!r} not found for member {member}. "
                f"Available: {list(ds.data_vars)}"
            )
        da = ds[variable]
        if "time" not in da.dims:
            raise ValueError(f"{variable} for member {member} has no time dimension.")
        ids = _hru_id_array(ds, da, member)
        id_to_index = {int(hru_id): index for index, hru_id in enumerate(ids)}
        missing = [hru_id for hru_id in requested if hru_id not in id_to_index]
        if missing:
            available = [int(hru_id) for hru_id in ids]
            raise ValueError(
                f"Member {member} has no hruId {missing}. Available: {available}."
            )
        time_index = _select_time_index(pd.DatetimeIndex(da["time"].values), when)
        selected = da.isel(time=time_index)
        values: dict[int, float] = {}
        for hru_id in requested:
            hru_index = id_to_index[hru_id]
            if "hru" in selected.dims:
                value = float(selected.isel(hru=hru_index).values)
            else:
                value = float(np.asarray(selected.values).reshape(-1)[hru_index])
            values[hru_id] = value
        units = da.attrs.get("units")
    if units is not None and not units_are_swe(str(units)):
        raise ValueError(
            f"Member {member} {variable} units {units!r} are not {_SWE_UNIT} or mm."
        )
    log.info("Member %s %s at %s is %s", member, variable, when, values)
    # #region agent log
    _agent_log(
        "B",
        "swe_obs.py:read_member_hru_swe",
        "read member swe",
        {"member": member, "when": str(when), "values": values, "runId": "post-fix"},
    )
    # #endregion
    return values
