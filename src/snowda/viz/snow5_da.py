from __future__ import annotations

from numbers import Integral
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

Figure = plt.Figure


def _resolve_hru_index(ds: xr.Dataset, hru: str | Integral | None) -> int:
    if hru is None:
        return 0
    if isinstance(hru, Integral):
        return int(hru)
    if "hru" in ds.coords and hru in ds["hru"].values:
        return int(np.where(ds["hru"].values == hru)[0][0])
    if "hruId" in ds.coords and hru in ds["hruId"].values:
        return int(np.where(ds["hruId"].values == hru)[0][0])
    raise ValueError(f"HRU '{hru}' not found in dataset")


def plot_snow5_da(
    member_datasets: Iterable[xr.Dataset],
    *,
    obs_da: xr.DataArray | None = None,
    obs_var_name: str | None = None,
    open_loop_ds: xr.Dataset | None = None,
    hru: str | Integral | None = None,
    forcing_var: str = "pptrate",
    temp_var: str = "airtemp",
    swe_var: str = "swe",
    figsize: tuple[int, int] = (12, 8),
) -> Figure:
    """Plot ensemble forcings and SWE outputs for a single HRU."""
    members = list(member_datasets)
    if not members:
        raise ValueError("No member datasets provided.")

    idx = _resolve_hru_index(members[0], hru)
    times = members[0]["time"].values

    forcing_vals = np.vstack(
        [
            ds[forcing_var].isel(hru=idx).values
            for ds in members
        ]
    ).T
    temp_vals = np.vstack(
        [
            ds[temp_var].isel(hru=idx).values
            for ds in members
        ]
    ).T
    swe_vals = np.vstack(
        [
            ds[swe_var].isel(hru=idx).values
            for ds in members
        ]
    ).T

    quantiles = np.quantile(forcing_vals, [0.1, 0.5, 0.9], axis=1)
    temp_quantiles = np.quantile(temp_vals, [0.1, 0.5, 0.9], axis=1)
    swe_quantiles = np.quantile(swe_vals, [0.1, 0.5, 0.9], axis=1)

    # Ratio subplot heights to 1, 1, 4
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True, constrained_layout=True,
        gridspec_kw={'height_ratios': [1, 1, 4]}
    )
    ax_force, ax_temp, ax_swe = axes

    ax_force.fill_between(
        times,
        quantiles[0],
        quantiles[2],
        color="tab:cyan",
        alpha=0.3,
        label="10-90% precip",
    )
    ax_force.plot(times, quantiles[1], color="tab:blue", label="Median precip")
    ax_force.set_ylabel("Precip (mm/day)")
    ax_force.legend(loc="upper left")
    ax_force.grid(True)

    ax_temp.fill_between(
        times,
        temp_quantiles[0],
        temp_quantiles[2],
        color="tab:orange",
        alpha=0.3,
        label="10-90% temp",
    )
    ax_temp.plot(times, temp_quantiles[1], color="tab:red", label="Median temp")
    ax_temp.set_ylabel("Air temp (°C)")
    ax_temp.legend(loc="upper left")
    ax_temp.grid(True)

    for member_idx in range(swe_vals.shape[1]):
        ax_swe.plot(
            times,
            swe_vals[:, member_idx],
            color="tab:gray",
            alpha=0.4,
            linewidth=1,
        )
    ax_swe.plot(
        times,
        swe_quantiles[1],
        color="tab:green",
        linewidth=2,
        label="Median SWE",
    )
    ax_swe.fill_between(
        times,
        swe_quantiles[0],
        swe_quantiles[2],
        color="tab:green",
        alpha=0.3,
        label="10-90% SWE",
    )
    if obs_da is not None and obs_var_name:
        obs_vals = obs_da.sel(hru=hru if hru is not None else obs_da["hru"].values[0])
        ax_swe.plot(
            obs_vals["time"].values,
            obs_vals.values,
            linestyle="--",
            color="black",
            alpha=0.8,
            label=f"{obs_var_name} obs",
        )
    ax_swe.set_ylabel("SWE (kg/m²)")
    ax_swe.set_xlabel("Time")
    ax_swe.set_xlim(times[0], times[-1])

    if open_loop_ds is not None:
        ol_val = open_loop_ds[swe_var].sel(hru=idx)
        ax_swe.plot(
            ol_val["time"].values,
            ol_val.values,
            linestyle="--",
            color="tab:purple",
            label="open-loop SWE",
        )
    ax_swe.legend(loc="upper left")
    ax_swe.grid(True)

    fig.suptitle(f"Snow5 DA ensemble (HRU #{idx})", fontsize=14)
    return fig
