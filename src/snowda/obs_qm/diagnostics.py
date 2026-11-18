"""Diagnostic plots for observation-to-model SWE quantile mapping.

These figures check that the transformation behaves as intended. They are
not skill scores for forecasts or data assimilation.
"""

from __future__ import annotations

from typing import Any, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from snowda.obs_qm.mapping import empirical_quantiles
from snowda.obs_qm.preprocess import FLAG_MAPPED
from snowda.obs_qm.seasonal import (
    DEFAULT_WATER_YEAR_START_MONTH,
    DEFAULT_WINDOW_DAYS,
    get_seasonal_samples,
    positive_paired_samples,
    water_year,
)

_PUB_STYLE = {
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "axes.unicode_minus": False,
}
_OBS_COLOR = "#2166ac"
_MODEL_COLOR = "#4d4d4d"
_MAPPED_COLOR = "#d6604d"
_SWE_UNIT = "kg m$^{-2}$"


def _station_slice(ds: xr.Dataset, station_id: Any | None, station_dim: str) -> xr.Dataset:
    if station_id is None:
        extra = [d for d in ds.dims if d != "time"]
        if extra:
            raise ValueError("station_id is required when more than one station is present.")
        return ds
    if station_dim in ds.dims:
        return ds.sel({station_dim: station_id})
    return ds


def _mapped_mask(ds: xr.Dataset) -> np.ndarray:
    return np.asarray(ds["mapping_flag"].values) == FLAG_MAPPED


def _station_title_extra(station: xr.Dataset) -> str:
    extra = ""
    if "n_water_years" in station:
        n_wy = int(np.asarray(station["n_water_years"]).reshape(-1)[0])
        extra = f"  ({n_wy} water years)"
    if "passes_year_screen" in station:
        passed = bool(np.asarray(station["passes_year_screen"]).reshape(-1)[0])
        if not passed:
            extra += "  SCREENED"
    return extra


def _station_series(station: xr.Dataset) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    times = pd.DatetimeIndex(station["time"].values)
    obs = np.asarray(station["swe_obs"].values, dtype=np.float64)
    model = np.asarray(station["swe_model"].values, dtype=np.float64)
    y_star = np.asarray(station["swe_model_equivalent"].values, dtype=np.float64)
    return times, obs, model, y_star


def _one_to_one(ax, x: np.ndarray, y: np.ndarray) -> None:
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    ax.plot([lo, hi], [lo, hi], color="k", linewidth=0.8, linestyle="--", zorder=1)


def _scatter_by_water_year(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    wy: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
):
    """Scatter colored by held-out water year (leave-one-year-out)."""
    sc = ax.scatter(
        x,
        y,
        c=wy,
        s=22,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.85,
        edgecolors="none",
        zorder=2,
    )
    _one_to_one(ax, x, y)
    return sc


def _timeseries_model(
    model: xr.DataArray | None,
    fallback_times: pd.DatetimeIndex,
    fallback_values: np.ndarray,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Midnight model line for the timeseries panel.

    ``model`` is the full midnight series. Without it, the line falls back
    to ``swe_model`` on the observation dates.
    """
    if model is None:
        return fallback_times, fallback_values
    series = model
    extra = [d for d in series.dims if d != "time"]
    if extra:
        series = series.squeeze(drop=True)
    extra = [d for d in series.dims if d != "time"]
    if extra:
        raise ValueError(
            f"model series for the timeseries must be 1-D in time (extra dims={extra})."
        )
    times = pd.DatetimeIndex(pd.to_datetime(series["time"].values, errors="coerce"))
    if times.tz is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    values = np.asarray(series.values, dtype=np.float64).reshape(-1)
    return times, values


def _panel_letter(ax, letter: str) -> None:
    ax.text(
        0.0,
        1.02,
        letter,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=11,
        va="bottom",
        ha="left",
        clip_on=False,
    )


def _water_year_rows(
    first: int,
    last: int,
    years_per_row: int,
    start_month: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Half-open water-year windows of up to ``years_per_row`` years.

    The last window stops at ``last`` so the axis does not run past the record.
    """
    rows = []
    start_wy = int(first)
    last_wy = int(last)
    while start_wy <= last_wy:
        n_years = min(int(years_per_row), last_wy - start_wy + 1)
        start = pd.Timestamp(year=start_wy - 1, month=int(start_month), day=1)
        end = start + pd.DateOffset(years=n_years)
        rows.append((start, end, n_years))
        start_wy += n_years
    return rows


def plot_station_diagnostics(
    ds: xr.Dataset,
    station_id: Any | None = None,
    *,
    station_dim: str = "station_id",
    water_year_start_month: int = DEFAULT_WATER_YEAR_START_MONTH,
    bins: int = 30,
    observation_relative_uncertainty: float = 0.10,
    observation_sigma_floor: float = 36.0,
) -> Figure:
    """Four-panel comparison of one station's observation-to-model mapping.

    Panels are (a) positive SWE distributions, (b) observed versus
    model-equivalent SWE, (c) same-day midnight model versus model-equivalent
    SWE, and (d) quantiles of mapped SWE versus the positive model
    climatology. Scatter colour is the water year. Panel (c) shades
    ``±max(observation_relative_uncertainty * SWE, observation_sigma_floor)``
    about the 1:1 line. The time series is drawn by
    :func:`plot_station_hydrograph`.
    """
    if observation_relative_uncertainty < 0.0:
        raise ValueError("observation_relative_uncertainty must be >= 0.")
    if observation_sigma_floor < 0.0:
        raise ValueError("observation_sigma_floor must be >= 0.")
    station = _station_slice(ds, station_id, station_dim)
    times, obs, model, y_star = _station_series(station)
    mapped = _mapped_mask(station)
    wy = water_year(times, start_month=water_year_start_month)

    paired = np.isfinite(obs) & np.isfinite(model)
    positive = paired & (obs > 0) & (model > 0)
    obs_pos = obs[positive]
    model_pos = model[positive]
    y_valid = mapped & np.isfinite(y_star)
    y_star_pos = y_star[y_valid]
    finite_both = y_valid & np.isfinite(obs)
    both_model = y_valid & np.isfinite(model)

    with plt.rc_context(_PUB_STYLE):
        fig = plt.figure(figsize=(7.4, 6.4))
        gs = GridSpec(
            2,
            3,
            figure=fig,
            width_ratios=[1.0, 1.0, 0.028],
            wspace=0.45,
            hspace=0.48,
            left=0.10,
            right=0.90,
            bottom=0.09,
            top=0.84,
        )
        ax_hist = fig.add_subplot(gs[0, 0])
        ax_obs = fig.add_subplot(gs[0, 1])
        ax_same = fig.add_subplot(gs[1, 0])
        ax_qq = fig.add_subplot(gs[1, 1])
        cax = fig.add_subplot(gs[:, 2])

        samples = [values for values in (obs_pos, model_pos, y_star_pos) if values.size]
        edges = None
        if samples:
            low = min(np.min(values) for values in samples)
            high = max(np.max(values) for values in samples)
            edges = np.linspace(low, high, bins + 1)
        if obs_pos.size:
            ax_hist.hist(
                obs_pos,
                bins=edges,
                density=True,
                alpha=0.55,
                color=_OBS_COLOR,
                label="Observed",
            )
        if model_pos.size:
            ax_hist.hist(
                model_pos,
                bins=edges,
                density=True,
                alpha=0.45,
                color=_MODEL_COLOR,
                label="Model",
            )
        if y_star_pos.size:
            ax_hist.hist(
                y_star_pos,
                bins=edges,
                density=True,
                alpha=0.45,
                color=_MAPPED_COLOR,
                label="Model-equivalent",
            )
        ax_hist.set_xlabel(f"SWE ({_SWE_UNIT})")
        ax_hist.set_ylabel("Density")
        if ax_hist.get_legend_handles_labels()[0]:
            ax_hist.legend(
                loc="lower left",
                bbox_to_anchor=(0.14, 1.12),
                frameon=False,
                borderaxespad=0.0,
                handlelength=1.1,
                labelspacing=0.15,
            )

        wy_color = wy[y_valid] if y_valid.any() else wy
        vmin = float(np.min(wy_color)) if wy_color.size else 0.0
        vmax = float(np.max(wy_color)) if wy_color.size else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0
        color_artist = None
        if finite_both.any():
            color_artist = _scatter_by_water_year(
                ax_obs,
                obs[finite_both],
                y_star[finite_both],
                wy[finite_both],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
        ax_obs.set_xlabel(f"Observed SWE ({_SWE_UNIT})")
        ax_obs.set_ylabel(f"Model-equivalent SWE ({_SWE_UNIT})")

        if both_model.any():
            color_artist = _scatter_by_water_year(
                ax_same,
                model[both_model],
                y_star[both_model],
                wy[both_model],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            x_lim = ax_same.get_xlim()
            y_lim = ax_same.get_ylim()
            upper = max(x_lim[1], y_lim[1])
            lower = max(min(x_lim[0], y_lim[0]), 0.0)
            grid = np.linspace(lower, upper, 400)
            sigma = np.maximum(
                observation_relative_uncertainty * grid,
                observation_sigma_floor,
            )
            ax_same.fill_between(
                grid,
                grid - sigma,
                grid + sigma,
                color="0.75",
                alpha=0.35,
                linewidth=0,
                zorder=0,
            )
            ax_same.set_xlim(x_lim)
            ax_same.set_ylim(y_lim)
        ax_same.set_xlabel(f"Model SWE ({_SWE_UNIT})")
        ax_same.set_ylabel(f"Model-equivalent SWE ({_SWE_UNIT})")

        if y_star_pos.size and model_pos.size:
            n = min(y_star_pos.size, model_pos.size)
            quantiles = np.linspace(0.0, 1.0, n, endpoint=True)
            qq_y = np.quantile(y_star_pos, quantiles)
            qq_m = np.quantile(model_pos, quantiles)
            ax_qq.scatter(qq_m, qq_y, s=16, color="0.15", edgecolors="none", zorder=2)
            _one_to_one(ax_qq, qq_m, qq_y)
        ax_qq.set_xlabel(f"Model SWE quantile ({_SWE_UNIT})")
        ax_qq.set_ylabel(f"Model-equivalent SWE quantile ({_SWE_UNIT})")

        for ax, letter in (
            (ax_hist, "a"),
            (ax_obs, "b"),
            (ax_same, "c"),
            (ax_qq, "d"),
        ):
            _panel_letter(ax, letter)

        if color_artist is not None:
            position = cax.get_position()
            height = position.height * 0.48
            cax.set_position(
                [
                    position.x0 + position.width * 0.15,
                    position.y0 + (position.height - height) / 2.0,
                    position.width * 0.45,
                    height,
                ]
            )
            cbar = fig.colorbar(color_artist, cax=cax)
            cbar.set_label("Water year", fontsize=7, color="0.35", labelpad=3)
            cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
            cbar.ax.tick_params(labelsize=6.5, width=0.4, length=2, colors="0.35")
            cbar.outline.set_linewidth(0.3)
            cbar.outline.set_edgecolor("0.45")
        else:
            cax.set_axis_off()
    return fig


def plot_station_hydrograph(
    ds: xr.Dataset,
    station_id: Any | None = None,
    *,
    model: xr.DataArray | None = None,
    station_dim: str = "station_id",
    water_year_start_month: int = DEFAULT_WATER_YEAR_START_MONTH,
    years_per_row: int = 5,
) -> Figure:
    """Midnight model SWE, observations, and model-equivalent observations.

    Each row spans ``years_per_row`` water years. Pass ``model`` to draw every
    midnight value; otherwise the line is ``swe_model`` on the observation dates.
    """
    if years_per_row < 1:
        raise ValueError("years_per_row must be >= 1.")
    station = _station_slice(ds, station_id, station_dim)
    times, obs, paired_model, y_star = _station_series(station)
    model_times, model_line = _timeseries_model(model, times, paired_model)

    span = model_times.append(times)
    span = span[span.notna()]
    if span.empty:
        raise ValueError("No timestamps to draw.")
    wy = water_year(span, start_month=water_year_start_month)
    rows = _water_year_rows(
        int(wy.min()),
        int(wy.max()),
        years_per_row,
        water_year_start_month,
    )
    finite_values = np.concatenate(
        [
            model_line[np.isfinite(model_line)],
            obs[np.isfinite(obs)],
            y_star[np.isfinite(y_star)],
        ]
    )
    ymax = float(np.max(finite_values)) if finite_values.size else 1.0

    with plt.rc_context(_PUB_STYLE):
        # A short final row spans fewer columns so each year keeps the same width.
        fig = plt.figure(figsize=(7.4, 0.45 + 1.15 * len(rows)))
        gs = GridSpec(
            len(rows),
            years_per_row,
            figure=fig,
            wspace=0.0,
            hspace=0.55,
            left=0.12,
            right=0.98,
            bottom=0.03,
            top=0.94,
        )
        first_ax = None
        for row_index, (start, end, n_years) in enumerate(rows):
            ax = fig.add_subplot(gs[row_index, :n_years], sharey=first_ax)
            if first_ax is None:
                first_ax = ax
            model_mask = (model_times >= start) & (model_times < end) & np.isfinite(model_line)
            obs_mask = (times >= start) & (times < end) & np.isfinite(obs)
            mapped_mask = (times >= start) & (times < end) & np.isfinite(y_star)
            first = row_index == 0
            if model_mask.any():
                ax.plot(
                    model_times[model_mask],
                    model_line[model_mask],
                    color=_MODEL_COLOR,
                    linewidth=0.7,
                    label="Model" if first else "_nolegend_",
                    zorder=1,
                )
            if obs_mask.any():
                ax.scatter(
                    times[obs_mask],
                    obs[obs_mask],
                    s=16,
                    color=_OBS_COLOR,
                    linewidths=0,
                    label="Observed" if first else "_nolegend_",
                    zorder=3,
                )
            if mapped_mask.any():
                ax.scatter(
                    times[mapped_mask],
                    y_star[mapped_mask],
                    s=18,
                    marker="o",
                    facecolors="none",
                    edgecolors=_MAPPED_COLOR,
                    linewidths=0.9,
                    label="Model-equivalent" if first else "_nolegend_",
                    zorder=4,
                )
            ax.margins(x=0)
            ax.set_xlim(start, end)
            ax.xaxis.set_major_locator(mdates.YearLocator(base=1, month=1, day=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.set_ylabel(f"SWE ({_SWE_UNIT})")
        if first_ax is not None:
            first_ax.set_ylim(0.0, ymax * 1.08 if ymax > 0.0 else 1.0)
        fig.legend(
            handles=[
                Line2D([0], [0], color=_MODEL_COLOR, linewidth=0.8, label="Model"),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=_OBS_COLOR,
                    markeredgecolor=_OBS_COLOR,
                    markersize=4.5,
                    linestyle="none",
                    label="Observed",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="none",
                    markeredgecolor=_MAPPED_COLOR,
                    markeredgewidth=0.9,
                    markersize=4.5,
                    linestyle="none",
                    label="Model-equivalent",
                ),
            ],
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.55, 0.995),
        )
    return fig


def plot_station_comparison_timeseries(
    ds: xr.Dataset,
    station_id: Any | None = None,
    *,
    station_dim: str = "station_id",
) -> Figure:
    """Compare midnight model SWE, point observations, and model-equivalent obs.

    The model series is drawn as a line (daily midnight values). Observations
    and the quantile-mapped model-equivalent values are points because the
    in-situ record is gappy.
    """
    station = _station_slice(ds, station_id, station_dim)
    times, obs, model, y_star = _station_series(station)
    extra = _station_title_extra(station)
    title_id = station_id if station_id is not None else station.attrs.get("station_id", "")

    fig, ax = plt.subplots(figsize=(14, 4.5))
    model_valid = np.isfinite(model)
    obs_valid = np.isfinite(obs)
    y_valid = np.isfinite(y_star)
    if model_valid.any():
        ax.plot(
            times,
            model,
            color="0.55",
            linewidth=0.8,
            label="model SWE",
            zorder=1,
        )
    if obs_valid.any():
        ax.scatter(
            times[obs_valid],
            obs[obs_valid],
            s=22,
            color="C0",
            label="obs",
            zorder=3,
        )
    if y_valid.any():
        ax.scatter(
            times[y_valid],
            y_star[y_valid],
            s=22,
            marker="o",
            facecolors="none",
            edgecolors="C1",
            linewidths=1.1,
            label="obs in model space",
            zorder=4,
        )
    ax.set_ylabel("SWE (kg m-2)")
    ax.set_xlabel("time")
    ax.set_title(
        f"Model, observations, and model-equivalent observations  {title_id}{extra}"
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_station_cross_validation(
    ds: xr.Dataset,
    station_id: Any | None = None,
    *,
    station_dim: str = "station_id",
    water_year_start_month: int = DEFAULT_WATER_YEAR_START_MONTH,
    bins: int = 10,
) -> Figure:
    """Leave-one-water-year-out checks for one station.

    These panels test the mapping when the target water year is held out.
    They are not forecast or assimilation skill scores.

    1. PIT of ``quantile_obs``: if other years describe this year's ranks,
       percentiles should be roughly uniform on (0, 1).
    2. Q-Q: held-out model-equivalent SWE vs positive midnight model SWE.
    3. Same-day midnight model vs model-equivalent, by water year.
    4. Number of mapped observations in each held-out water year.
    """
    station = _station_slice(ds, station_id, station_dim)
    times, obs, model, y_star = _station_series(station)
    mapped = _mapped_mask(station)
    extra = _station_title_extra(station)
    title_id = station_id if station_id is not None else station.attrs.get("station_id", "")
    cv_label = station.attrs.get("cross_validation") or ds.attrs.get(
        "cross_validation", "leave_one_water_year_out"
    )

    q_obs = np.asarray(station["quantile_obs"].values, dtype=np.float64)
    wy = water_year(times, start_month=water_year_start_month)
    paired = np.isfinite(obs) & np.isfinite(model)
    model_pos = model[paired & (obs > 0) & (model > 0)]
    y_valid = mapped & np.isfinite(y_star)
    q_valid = mapped & np.isfinite(q_obs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Leave-one-water-year-out cross-validation  {title_id}{extra}\n"
        f"({cv_label}; not forecast skill)"
    )

    ax = axes[0, 0]
    if q_valid.any():
        ax.hist(
            q_obs[q_valid],
            bins=np.linspace(0.0, 1.0, bins + 1),
            density=True,
            color="C0",
            alpha=0.75,
            edgecolor="white",
        )
        ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--", label="Uniform(0, 1)")
        ax.legend(fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("quantile_obs  $F_{obs}(y)$")
    ax.set_ylabel("density")
    ax.set_title("PIT of held-out observation percentiles")

    ax = axes[0, 1]
    y_star_pos = y_star[y_valid]
    if y_star_pos.size and model_pos.size:
        n = min(y_star_pos.size, model_pos.size)
        q = np.linspace(0.0, 1.0, n, endpoint=True)
        qq_y = np.quantile(y_star_pos, q)
        qq_m = np.quantile(model_pos, q)
        ax.scatter(qq_m, qq_y, s=14, alpha=0.7, color="C1")
        lo = float(min(qq_m.min(), qq_y.min()))
        hi = float(max(qq_m.max(), qq_y.max()))
        ax.plot([lo, hi], [lo, hi], color="k", linewidth=0.8, linestyle="--")
        ax.set_xlabel("model positive SWE quantiles")
        ax.set_ylabel("LOYO model-equivalent quantiles")
    ax.set_title("Q-Q: mapped obs vs model climatology")

    ax = axes[1, 0]
    both = y_valid & np.isfinite(model)
    if both.any():
        sc = ax.scatter(
            model[both],
            y_star[both],
            c=wy[both],
            s=16,
            cmap="viridis",
            alpha=0.8,
        )
        lo = float(np.nanmin([model[both].min(), y_star[both].min()]))
        hi = float(np.nanmax([model[both].max(), y_star[both].max()]))
        ax.plot([lo, hi], [lo, hi], color="k", linewidth=0.8, linestyle="--")
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("held-out water year")
    ax.set_xlabel("same-day midnight model SWE")
    ax.set_ylabel("LOYO model-equivalent SWE")
    ax.set_title("Same-day model vs mapped obs (not pair-matched)")

    ax = axes[1, 1]
    if y_valid.any():
        years = wy[y_valid]
        uniq, counts = np.unique(years, return_counts=True)
        ax.bar(uniq.astype(int), counts, color="C0", width=0.8, align="center")
        ax.set_xlabel("held-out water year")
        ax.set_ylabel("n mapped")
        if uniq.size > 12:
            ax.tick_params(axis="x", labelrotation=45)
    ax.set_title("Mapped observations per held-out year")

    fig.tight_layout()
    return fig


def mapping_curve_arrays(
    ds: xr.Dataset,
    target_date,
    *,
    station_id: Any | None = None,
    station_dim: str = "station_id",
    window_days: int = DEFAULT_WINDOW_DAYS,
    exclude_water_year: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Return the empirical mapping curve for one date (for inspection)."""
    station = _station_slice(ds, station_id, station_dim)
    times = pd.DatetimeIndex(station["time"].values)
    obs = np.asarray(station["swe_obs"].values, dtype=np.float64)
    model = np.asarray(station["swe_model"].values, dtype=np.float64)
    obs_s, mod_s = get_seasonal_samples(
        times,
        obs,
        model,
        target_date,
        window_days=window_days,
        exclude_water_year=exclude_water_year,
    )
    obs_pos, mod_pos = positive_paired_samples(obs_s, mod_s)
    if obs_pos.size == 0:
        return {
            "obs_quantiles": np.array([]),
            "model_quantiles": np.array([]),
            "percentiles": np.array([]),
            "n_samples": np.array([0]),
        }
    x_obs, p_obs, n = empirical_quantiles(obs_pos)
    x_mod, p_mod, _ = empirical_quantiles(mod_pos)
    p_grid = np.linspace(max(p_obs[0], p_mod[0]), min(p_obs[-1], p_mod[-1]), 200)
    return {
        "obs_quantiles": np.interp(p_grid, p_obs, x_obs),
        "model_quantiles": np.interp(p_grid, p_mod, x_mod),
        "percentiles": p_grid,
        "n_samples": np.array([n]),
    }
