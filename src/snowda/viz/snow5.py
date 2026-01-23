from __future__ import annotations

from numbers import Integral
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

Figure = plt.Figure


def _resolve_hru_selection(ds: xr.Dataset, hru: str | Integral | None) -> xr.Dataset:
    """Select a single HRU slice from a dataset."""
    if "hru" not in ds.dims:
        raise ValueError("Dataset must contain an 'hru' dimension.")

    if hru is None:
        return ds.isel(hru=0)

    if isinstance(hru, Integral):
        return ds.isel(hru=int(hru))

    if isinstance(hru, str):
        if hru not in ds["hru"].values:
            raise ValueError(f"HRU '{hru}' not present in dataset.")
        return ds.sel(hru=hru)

    raise TypeError("hru must be an int index, a str coordinate label, or None.")


def plot_snow5_timeseries(
    forcing_ds: xr.Dataset,
    result_ds: xr.Dataset,
    *,
    hru: str | Integral | None = None,
    figsize: tuple[int, int] = (10, 6),
    cmap: Literal["tab10", "tab20", "tab20b", "tab20c"] | None = "tab10",
) -> Figure:
    """Plot forcing inputs (temperature + precipitation) and SWE output for a single HRU."""

    forcing = _resolve_hru_selection(forcing_ds, hru)
    result = _resolve_hru_selection(result_ds, hru)

    if "swe" not in result:
        raise KeyError("`swe` is missing from the result dataset.")

    times = forcing["time"].values
    color_cycle = None
    if cmap:
        color_cycle = plt.cm.get_cmap(cmap).colors

    # Set subplot ratios to 1, 1, 3
    height_ratios = [1, 1, 3]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={'height_ratios': height_ratios},
    )
    ax_weather, ax_swe, ax_fsca = axes

    color_temp = color_cycle[0] if color_cycle is not None else "tab:blue"
    ax_weather.plot(times, forcing["airtemp"].values, label="Air temp (°C)", color=color_temp)
    ax_weather.set_ylabel("Air temp (°C)")
    ax_weather.set_title(f"Snow5 HRU {forcing['hru'].values}")
    ax_weather.grid(True, linestyle=":", alpha=0.6)

    ax_precip = ax_weather.twinx()
    color_precip = color_cycle[1] if color_cycle is not None else "tab:purple"
    ax_precip.bar(
        times,
        forcing["pptrate"].values,
        color=color_precip,
        alpha=0.4,
        label="Precipitation (mm/day)",
    )
    ax_precip.set_ylabel("Precip (mm/day)")
    ax_precip.set_ylim(bottom=0)

    ax_swe.plot(
        times,
        result["swe"].values,
        color=color_cycle[2] if color_cycle is not None else "tab:orange",
        label="SWE output",
    )
    ax_swe.set_ylabel("SWE (kg m⁻²)")
    ax_swe.grid(True, linestyle=":", alpha=0.6)

    if "fsca" in result:
        ax_fsca.plot(
            times,
            result["fsca"].values,
            color=color_cycle[3] if color_cycle is not None else "tab:green",
            label="fSCA",
        )
        ax_fsca.set_ylabel("fSCA")
        ax_fsca.set_ylim(0, 1)
        ax_fsca.grid(True, linestyle=":", alpha=0.6)
    else:
        ax_fsca.set_visible(False)

    ax_fsca.set_xlabel("Time")

    handles_weather, labels_weather = ax_weather.get_legend_handles_labels()
    handles_precip, labels_precip = ax_precip.get_legend_handles_labels()
    ax_weather.legend(
        handles_weather + handles_precip,
        labels_weather + labels_precip,
        loc="upper left",
    )

    return fig
