"""Read NorSWE / SUMMA SWE and write mapped observation datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import xarray as xr

from snowda.obs_qm.pairing import extract_midnight_swe, squeeze_single_location

DEFAULT_OBS_VARIABLE = "snw"
DEFAULT_MODEL_VARIABLE = "scalarSWE"
SWE_UNITS = "kg m-2"

_MAPPED_VARIABLES = (
    "swe_obs",
    "swe_model",
    "swe_model_equivalent",
    "quantile_obs",
    "n_samples_mapping",
    "mapping_flag",
    "paired",
    "n_paired",
    "obs_zero_fraction",
    "model_zero_fraction",
    "both_positive_fraction",
    "n_calendar_years",
    "n_water_years",
    "n_water_years_positive",
    "first_year",
    "last_year",
    "passes_year_screen",
)


def read_swe_observations(
    path: str | Path,
    *,
    variable: str = DEFAULT_OBS_VARIABLE,
) -> xr.Dataset:
    """Open a point SWE observation dataset (e.g. NorSWE).

    The dataset is returned intact. ``variable`` must exist and include
    ``time`` and a station/location dimension.
    """
    nc_path = Path(path)
    ds = xr.open_dataset(nc_path)
    if variable not in ds:
        ds.close()
        raise KeyError(
            f"Variable {variable!r} not found in {nc_path}. "
            f"Available: {list(ds.data_vars)}"
        )
    da = ds[variable]
    if "time" not in da.dims:
        ds.close()
        raise ValueError(
            f"Observation variable {variable!r} has no time dimension "
            f"(dims={list(da.dims)})."
        )
    station_dims = [d for d in da.dims if d != "time"]
    if not station_dims:
        ds.close()
        raise ValueError(
            f"Observation variable {variable!r} has no station/location "
            f"dimension (dims={list(da.dims)})."
        )
    ds.attrs["obs_variable"] = variable
    ds.attrs["source"] = str(nc_path)
    return ds


def read_midnight_model_swe(
    path: str | Path,
    *,
    variable: str = DEFAULT_MODEL_VARIABLE,
    hru: Optional[int] = None,
    hru_id: Any = None,
    midnight_tolerance: str = "1min",
) -> xr.DataArray:
    """Read model SWE and keep only midnight (00:00) samples.

    Lumped one-HRU files are squeezed automatically. No daily mean and no
    interpolation.
    """
    nc_path = Path(path)
    with xr.open_dataset(nc_path) as ds:
        if variable not in ds:
            raise KeyError(
                f"Variable {variable!r} not found in {nc_path}. "
                f"Available: {list(ds.data_vars)}"
            )
        da = squeeze_single_location(ds[variable], hru=hru, hru_id=hru_id).load()
    da = extract_midnight_swe(da, tolerance=midnight_tolerance)
    da.name = "swe_model"
    units = da.attrs.get("units") or SWE_UNITS
    da.attrs["units"] = units
    da.attrs["source"] = str(nc_path)
    da.attrs["model_variable"] = variable
    return da


def attach_mapped_variables(
    obs_ds: xr.Dataset,
    mapped: xr.Dataset,
    *,
    obs_variable: str = DEFAULT_OBS_VARIABLE,
) -> xr.Dataset:
    """Copy mapped fields onto the original observation dataset.

    Original variables are preserved. ``swe_obs`` is added as an explicit
    copy of the observation SWE used for mapping.
    """
    out = obs_ds.copy()
    for name in _MAPPED_VARIABLES:
        if name not in mapped:
            continue
        da = mapped[name]
        # Match observation dimension order when possible.
        if obs_variable in out and set(da.dims) == set(out[obs_variable].dims):
            da = da.transpose(*out[obs_variable].dims)
        out[name] = da
        out[name].attrs.update(mapped[name].attrs)

    skip_attrs = {
        "n_paired",
        "obs_zero_fraction",
        "model_zero_fraction",
        "both_positive_fraction",
        "station_id",
        "calibration_start",
        "calibration_end",
    }
    for key, value in mapped.attrs.items():
        if key in skip_attrs:
            continue
        out.attrs[key] = value
    if "swe_obs" in out and obs_variable in out:
        out["swe_obs"].attrs["source_variable"] = obs_variable
    return out


def write_mapped_observations(ds: xr.Dataset, path: str | Path) -> Path:
    """Write the augmented observation dataset to netCDF."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    return out_path
