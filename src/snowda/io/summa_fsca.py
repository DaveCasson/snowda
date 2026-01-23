from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import xarray as xr
import re

from snowda.core.fsca_models import FscaState, compute_fsca_hru, initialize_output_arrays
from snowda.io.units import UnitsError, convert_vars, dequantify
from snowda.utils.logging import get_logger

log = get_logger(__name__)
_TZ_REGEX = re.compile(r"\s[+-]\d{1,2}:\d{2}$")


def _strip_timezone_units(ds: xr.Dataset) -> None:
    for da in list(ds.variables.values()):
        units = da.attrs.get("units")
        if isinstance(units, str) and _TZ_REGEX.search(units):
            da.attrs["units"] = _TZ_REGEX.sub("", units).strip()


def _open_summa_output(summa_output_files: Sequence[str | Path]) -> xr.Dataset:
    if len(summa_output_files) == 1:
        return xr.open_dataset(summa_output_files[0])
    return xr.open_mfdataset(summa_output_files, combine="by_coords")


def prepare_summa_fsca_input(
    summa_output_files: Sequence[str | Path], fsca_input: str | Path
) -> tuple[Path, list[dict[str, str]]]:
    log.info("Preparing SUMMA fSCA input from %s", summa_output_files)
    conversions: list[dict[str, str]] = []

    with _open_summa_output(summa_output_files) as summa_ds:
        fsca_ds = xr.Dataset(coords=summa_ds.coords)

        for var in ["scalarCanopyLiqDrainage_total", "scalarThroughfallRain_total"]:
            fsca_ds[var] = xr.where(summa_ds["nSnow"] != 0, summa_ds[var], 0)

        for var in ["scalarSnowDrainage_total", "scalarInfiltration_total"]:
            fsca_ds[var] = summa_ds[var]

        fsca_ds["snow_input"] = (
            summa_ds["scalarCanopySnowUnloading_total"]
            + summa_ds["scalarThroughfallSnow_total"]
            + fsca_ds["scalarCanopyLiqDrainage_total"]
            + fsca_ds["scalarThroughfallRain_total"]
        )

        fsca_ds["snow_output"] = (
            fsca_ds["scalarSnowDrainage_total"]
            - summa_ds["scalarSnowSublimation_total"]
            + summa_ds["scalarSfcMeltPond"]
        )

        fsca_ds["snow_water_balance"] = (
            fsca_ds["snow_input"].cumsum(dim="time")
            - fsca_ds["snow_output"].cumsum(dim="time")
            - summa_ds["scalarSWE"]
        )

        for var in summa_ds.data_vars:
            if var not in fsca_ds.data_vars:
                fsca_ds[var] = summa_ds[var]

        try:
            targets = {
                "scalarSnowDrainage_total": "millimeter",
                "scalarInfiltration_total": "millimeter",
            }
            for name, target in targets.items():
                if name in fsca_ds:
                    from_unit = fsca_ds[name].attrs.get("units")
                    conversions.append(
                        {"variable": name, "from": str(from_unit), "to": target}
                    )
            _strip_timezone_units(fsca_ds)
            try:
                fsca_ds = convert_vars(
                    fsca_ds, targets, quantify_first=True, strict=True
                )
            except (AssertionError, TypeError, ValueError) as exc:
                log.warning(
                    "Unit conversion failed (%s). Proceeding without conversions.",
                    exc,
                )
                fsca_ds = fsca_ds.copy()
            fsca_ds = dequantify(fsca_ds)
        except UnitsError as exc:
            raise UnitsError(
                f"Failed unit conversion in fSCA prep: {exc}"
            ) from exc

        fsca_ds_daily = fsca_ds.resample(time="D").sum()
        fsca_ds_daily["scalarSWE"] = summa_ds["scalarSWE"].resample(time="D").mean()

        out_path = Path(fsca_input)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fsca_ds_daily.to_netcdf(out_path)
        log.info("Prepared fSCA input written to %s", out_path)
        return out_path, conversions


def convert_arrays_to_dataset(output_arrays: dict, basis_ds: xr.Dataset) -> xr.Dataset:
    new_ds = xr.Dataset()
    timesteps = basis_ds["time"].values
    for label, array in output_arrays.items():
        da = xr.DataArray(
            np.vstack(array),
            coords=[timesteps, basis_ds["hru"]],
            dims=["time", "hru"],
            name=label,
        )
        new_ds[label] = da
    new_ds.time.encoding["units"] = "seconds since 1970-01-01 00:00:00"
    return new_ds


def calc_summa_fsca(
    fsca_input: str | Path,
    lumped_cv_value: float,
    output_file: str | Path,
    *,
    fsca_model: str = "ssnowd",
    helbig_settings: dict | None = None,
    snow_depth_var: str = "scalarSnowDepth",
) -> Path:
    log.info("Calculating fSCA for %s using %s", fsca_input, fsca_model)

    with xr.open_dataset(fsca_input) as summa_ds:
        num_hrus = summa_ds["hru"].sizes["hru"]
        cv_array = np.full(num_hrus, lumped_cv_value)

        timesteps = summa_ds["time"].values
        swe_array = summa_ds["scalarSWE"].values
        snowfall_array = summa_ds["snow_input"].values
        snowfall_total_array = summa_ds["snow_input"].cumsum(dim="time").values
        melt_array = summa_ds["snow_output"].values
        melt_total_array = summa_ds["snow_output"].cumsum(dim="time").values

        zeros_array = np.zeros_like(summa_ds["hru"])
        output_arrays = initialize_output_arrays(timesteps, zeros_array)

        states = [FscaState() for _ in range(num_hrus)]
        for j, cv in enumerate(cv_array):
            states[j].snowfall_total = snowfall_total_array[0, j]
            states[j].melt_total = melt_total_array[0, j]
            states[j].fsca = 0
            states[j].residual_new_snow = 0

            for i, _timestep in enumerate(timesteps):
                fsca_model_input = {
                    "swe": swe_array[i, j],
                    "snowfall": snowfall_array[i, j],
                    "melt": melt_array[i, j],
                    "snowfall_total": states[j].snowfall_total,
                    "melt_total": states[j].melt_total,
                    "fsca": states[j].fsca,
                    "residual_new_snow": states[j].residual_new_snow,
                }
                if fsca_model == "helbig":
                    if snow_depth_var not in summa_ds:
                        raise ValueError(
                            f"Missing snow depth variable '{snow_depth_var}' for helbig model."
                        )
                    fsca_model_input["snow_depth"] = summa_ds[snow_depth_var].values[i, j]

                hru_values = compute_fsca_hru(
                    fsca_model_input,
                    cv,
                    fsca_model=fsca_model,
                    state=states[j],
                    helbig_settings=helbig_settings,
                )
                states[j].snowfall_total = hru_values.get(
                    "snowfall_total", states[j].snowfall_total
                )
                states[j].melt_total = hru_values.get("melt_total", states[j].melt_total)
                states[j].fsca = hru_values.get("fsca", states[j].fsca)
                states[j].residual_new_snow = hru_values.get(
                    "residual_new_snow", states[j].residual_new_snow
                )

                for label, value in hru_values.items():
                    if label in output_arrays:
                        output_arrays[label][i, j] = value

        result_ds = convert_arrays_to_dataset(output_arrays, summa_ds)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_ds.to_netcdf(out_path)
    log.info("fSCA output written to %s", out_path)
    return out_path
