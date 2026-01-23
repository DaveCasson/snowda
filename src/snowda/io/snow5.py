from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import xarray as xr

from snowda.utils.logging import get_logger

log = get_logger(__name__)

def read_snow5_forcing(forcing_dir: Path | str, *, pattern: str = "*.nc") -> xr.Dataset:
    forcing_dir = Path(forcing_dir)
    if not forcing_dir.exists():
        raise FileNotFoundError(f"Forcing directory does not exist: {forcing_dir}")

    # Support both a directory (containing matching files) and a file path directly
    if forcing_dir.is_file():
        forcing_files = [forcing_dir]
    else:
        forcing_files = sorted(forcing_dir.glob(pattern))
    if not forcing_files:
        raise FileNotFoundError(f"No forcing files found matching {pattern} in {forcing_dir}")
    if len(forcing_files) != 1:
        raise ValueError(f"Expected 1 forcing file, found {len(forcing_files)}: {forcing_files[:5]}")

    f = forcing_files[0]
    log.info("Reading snow5 forcing from %s", f)

    ds = xr.open_dataset(f, decode_times=True)  # much less overhead than open_mfdataset

    # Unit conversions
    hru_vals = ds["hru"].values
    airtemp = (ds["airtemp"] - 273.15).astype("float32")
    pptrate = (ds["pptrate"] * 3600.0).astype("float32")
    if "hruId" in ds.coords:
        hru_ids = ds["hruId"].values
    elif "hruId" in ds:
        hru_ids = ds["hruId"].values
    else:
        hru_ids = hru_vals

    # Daily aggregates (still lazy until computed)
    airtemp_daily = airtemp.resample(time="1D").mean().rename("airtemp")
    pptrate_daily = pptrate.resample(time="1D").sum().rename("pptrate")

    # DOY as a coordinate over time only (broadcast happens lazily when needed)
    doy_time = pptrate_daily["time"].dt.dayofyear.astype("int16").rename("doy_time")

    out = xr.Dataset(
        data_vars={
            "airtemp": airtemp_daily,
            "pptrate": pptrate_daily,
            # doy will broadcast across hru without constructing a giant numpy array
            "doy": doy_time.broadcast_like(pptrate_daily),
        }
    ).assign_coords({"hru": hru_vals, "hruId": ("hru", hru_ids)})

    # Close file ASAP; data stays lazy unless you compute
    ds.close()

    # If you truly need everything loaded, do out.load()
    return out


def convert_arrays_to_dataset(
    output_arrays: Mapping[str, Iterable[Any]],
    basis_ds: xr.Dataset,
    *,
    include_forcing: bool = True,
) -> xr.Dataset:
    """Turn the simulation arrays into a CF-like dataset."""
    if "time" not in basis_ds.dims or "hru" not in basis_ds.dims:
        raise ValueError("Basis dataset must contain 'time' and 'hru' dimensions.")
    coords = {
        "time": basis_ds["time"].values,
        "hru": basis_ds["hru"].values,
    }
    if "hruId" in basis_ds.coords:
        coords["hruId"] = ("hru", basis_ds["hruId"].values)
    data_vars = {}
    for label, array in output_arrays.items():
        data = np.asarray(array, dtype="float64")
        if data.ndim != 2:
            raise ValueError(f"Output array {label} must be 2D (time, hru).")
        data_vars[label] = (("time", "hru"), data)

    model_ds = xr.Dataset(data_vars=data_vars, coords=coords)
    if include_forcing:
        forcing_vars = {
            name: basis_ds[name]
            for name in ("airtemp", "pptrate", "doy")
            if name in basis_ds
        }
        if forcing_vars:
            model_ds = xr.merge([model_ds, xr.Dataset(forcing_vars)])
    model_ds["time"].encoding.setdefault("units", "seconds since 1970-01-01 00:00:00")
    return model_ds


def write_results(dataset: xr.Dataset, output_path: Path | str) -> Path:
    """Persist the snow5 output to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)
    log.info("Wrote snow_5 results: %s", output_path)
    return output_path
