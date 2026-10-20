"""Plot historic DA traces, assimilation states, and ESP forecast envelopes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from snowda.io.da_traces import DaForecast, DaTraceSet
from snowda.io.esp_streamflow import ENSEMBLE_DIM

QUANTILE_OUTER = (0.10, 0.90)
QUANTILE_INNER = (0.25, 0.75)
_HISTORIC_COLOR = "0.35"


def _ensure_axes(
    ax: Axes | None, *, figsize: tuple[float, float]
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        return fig, ax
    return ax.figure, ax


def _time_index(da: xr.DataArray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(da["time"].values))


def _forecast_colors(forecasts: tuple[DaForecast, ...]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab10")
    months = list(dict.fromkeys(forecast.init_month for forecast in forecasts))
    return {month: cmap(i % cmap.N) for i, month in enumerate(months)}


def _forecast_label(forecast: DaForecast, month_counts: dict[str, int]) -> str:
    if month_counts[forecast.init_month] == 1:
        return forecast.init_time.strftime("%Y-%m-%d")
    return forecast.init_month


def _data_bounds(traces: DaTraceSet) -> tuple[pd.Timestamp, pd.Timestamp]:
    times: list[pd.Timestamp] = []
    for member in traces.historic:
        for segment in member.segments:
            index = _time_index(segment)
            if index.empty:
                continue
            times.extend((index[0], index[-1]))
    for forecast in traces.forecasts:
        index = _time_index(forecast.values)
        if index.empty:
            continue
        times.extend((index[0], index[-1]))
    for state in traces.states:
        times.append(pd.Timestamp(state.time))
    if not times:
        raise ValueError("No historic traces, states, or forecasts to plot.")
    return min(times), max(times)


def _ensemble_quantiles(values: xr.DataArray) -> pd.DataFrame:
    quantiles = values.quantile(
        [
            QUANTILE_OUTER[0],
            QUANTILE_INNER[0],
            0.5,
            QUANTILE_INNER[1],
            QUANTILE_OUTER[1],
        ],
        dim=ENSEMBLE_DIM,
        skipna=True,
    )
    return pd.DataFrame(
        {
            "low": quantiles.sel(quantile=QUANTILE_OUTER[0]).values,
            "inner_low": quantiles.sel(quantile=QUANTILE_INNER[0]).values,
            "median": quantiles.sel(quantile=0.5).values,
            "inner_high": quantiles.sel(quantile=QUANTILE_INNER[1]).values,
            "high": quantiles.sel(quantile=QUANTILE_OUTER[1]).values,
        },
        index=_time_index(values),
    )


def _draw_forecasts(ax: Axes, traces: DaTraceSet) -> list[Patch]:
    colors = _forecast_colors(traces.forecasts)
    month_counts: dict[str, int] = {}
    for forecast in traces.forecasts:
        month_counts[forecast.init_month] = month_counts.get(forecast.init_month, 0) + 1
    handles: list[Patch] = []
    seen: set[str] = set()
    for forecast in traces.forecasts:
        color = colors[forecast.init_month]
        quantiles = _ensemble_quantiles(forecast.values)
        ax.fill_between(
            quantiles.index,
            quantiles["low"],
            quantiles["high"],
            color=color,
            alpha=0.28,
            zorder=3,
        )
        ax.fill_between(
            quantiles.index,
            quantiles["inner_low"],
            quantiles["inner_high"],
            color=color,
            alpha=0.5,
            zorder=3,
        )
        ax.plot(
            quantiles.index,
            quantiles["median"],
            color=color,
            linewidth=2.0,
            zorder=4,
        )
        if forecast.init_month in seen:
            continue
        seen.add(forecast.init_month)
        handles.append(
            Patch(
                facecolor=color,
                edgecolor="k",
                alpha=0.5,
                label=f"Forecast {_forecast_label(forecast, month_counts)}",
            )
        )
    return handles


def _draw_historic(ax: Axes, traces: DaTraceSet) -> list:
    labeled = False
    handles = []
    for member in traces.historic:
        for segment in member.segments:
            index = _time_index(segment)
            line = ax.plot(
                index,
                np.asarray(segment.values, dtype=float),
                color=_HISTORIC_COLOR,
                linewidth=1.0,
                alpha=0.85,
                zorder=2,
                label="Historic ensemble" if not labeled else "_nolegend_",
            )
            if not labeled:
                handles.extend(line)
                labeled = True
    return handles


def _draw_states(ax: Axes, traces: DaTraceSet) -> list:
    handles = []
    marked_state = False
    marked_obs = False
    marked_time = False
    for state in traces.states:
        line = ax.axvline(
            pd.Timestamp(state.time),
            color="0.15",
            linestyle=":",
            linewidth=1.0,
            zorder=4,
            label="Assimilation" if not marked_time else "_nolegend_",
        )
        if not marked_time:
            handles.append(line)
            marked_time = True
        values = np.asarray(state.values.values, dtype=float)
        finite = np.isfinite(values)
        if np.any(finite):
            scatter = ax.scatter(
                np.full(int(np.sum(finite)), pd.Timestamp(state.time)),
                values[finite],
                s=36,
                color="black",
                edgecolors="white",
                linewidths=0.6,
                zorder=6,
                label="State at assimilation" if not marked_state else "_nolegend_",
            )
            if not marked_state:
                handles.append(scatter)
                marked_state = True
        if state.observation is None or not np.isfinite(state.observation):
            continue
        star = ax.scatter(
            [pd.Timestamp(state.time)],
            [state.observation],
            marker="*",
            s=160,
            color="crimson",
            edgecolors="k",
            linewidths=0.4,
            zorder=7,
            label="SWE observation" if not marked_obs else "_nolegend_",
        )
        if not marked_obs:
            handles.append(star)
            marked_obs = True
    return handles


def _draw_observed(
    ax: Axes,
    obs: xr.DataArray,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list:
    times = pd.DatetimeIndex(pd.to_datetime(obs["time"].values))
    values = np.asarray(obs.values, dtype=float)
    keep = (times >= start) & (times <= end) & np.isfinite(values)
    if not np.any(keep):
        return []
    line = ax.plot(
        times[keep],
        values[keep],
        color="darkorange",
        linewidth=2.0,
        zorder=5,
        label="Observed",
    )
    return list(line)


def plot_da_traces(
    traces: DaTraceSet,
    obs: xr.DataArray | None = None,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Historic ensemble lines, assimilation states, and ESP quantile envelopes.

    Forecasts that share an initialization month-day use one color, matching
    the ESP streamflow envelopes (p10–p90, p25–p75, and median). ``obs`` is a
    continuous verification series drawn over the historic and forecast window.
    """
    fig, ax = _ensure_axes(ax, figsize=(14, 7))
    start, end = _data_bounds(traces)
    forecast_handles = _draw_forecasts(ax, traces)
    historic_handles = _draw_historic(ax, traces)
    observed_handles = []
    if obs is not None:
        observed_handles = _draw_observed(ax, obs, start, end)
    state_handles = _draw_states(ax, traces)
    ax.set_ylabel(traces.ylabel)
    ax.set_xlabel("Time")
    ax.set_title(title or "Data assimilation and ESP")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(start, end)
    handles = historic_handles + forecast_handles + state_handles + observed_handles
    labels = [handle.get_label() for handle in handles]
    if handles:
        ax.legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=8,
        )
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    return fig, ax
