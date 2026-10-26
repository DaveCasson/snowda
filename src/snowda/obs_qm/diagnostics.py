"""Diagnostic plots for observation-to-model SWE quantile mapping.

These figures check that the transformation behaves as intended. They are
not skill scores for forecasts or data assimilation.
"""

from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from snowda.obs_qm.mapping import empirical_quantiles
from snowda.obs_qm.preprocess import FLAG_MAPPED
from snowda.obs_qm.seasonal import (
    DEFAULT_WATER_YEAR_START_MONTH,
    DEFAULT_WINDOW_DAYS,
    get_seasonal_samples,
    positive_paired_samples,
    water_year,
)


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


def plot_station_diagnostics(
    ds: xr.Dataset,
    station_id: Any | None = None,
    *,
    model: xr.DataArray | None = None,
    station_dim: str = "station_id",
    water_year_start_month: int = DEFAULT_WATER_YEAR_START_MONTH,
    bins: int = 30,
) -> Figure:
    """One diagnostic figure: four comparison panels plus a timeseries row.

    Top row:

    1. Positive SWE distributions (obs, midnight model, transformed obs)
    2. Observed vs model-equivalent, colored by held-out water year
    3. Same-day midnight model vs mapped obs, colored by held-out water year
    4. Q-Q of mapped obs vs model climatology

    Bottom row: midnight model SWE (line), observations (points), and
    observations in model space (open points). Pass ``model`` to draw every
    midnight value. Otherwise the line uses ``swe_model``, which is only
    defined on observation dates.
    """
    full_model = model
    station = _station_slice(ds, station_id, station_dim)
    times, obs, model, y_star = _station_series(station)
    mapped = _mapped_mask(station)
    extra = _station_title_extra(station)
    title_id = station_id if station_id is not None else station.attrs.get("station_id", "")
    wy = water_year(times, start_month=water_year_start_month)
    cv_label = station.attrs.get("cross_validation") or ds.attrs.get(
        "cross_validation", "leave_one_water_year_out"
    )

    paired = np.isfinite(obs) & np.isfinite(model)
    obs_pos = obs[paired & (obs > 0) & (model > 0)]
    model_pos = model[paired & (obs > 0) & (model > 0)]
    y_valid = mapped & np.isfinite(y_star)
    y_star_pos = y_star[y_valid]
    finite_both = y_valid & np.isfinite(obs)
    both_model = y_valid & np.isfinite(model)

    fig = plt.figure(figsize=(17.5, 9.5))
    gs = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.75, 1.0],
        hspace=0.18,
        top=0.90,
        bottom=0.07,
        left=0.05,
        right=0.98,
    )
    gs_top = gs[0].subgridspec(1, 5, width_ratios=[1.05, 1.0, 1.0, 1.0, 0.045], wspace=0.28)
    fig.suptitle(
        f"Observation → model SWE quantile mapping  {title_id}{extra}\n"
        f"({cv_label}; scatter colour = held-out water year)",
        fontsize=12,
    )

    ax_hist = fig.add_subplot(gs_top[0, 0])
    ax_obs = fig.add_subplot(gs_top[0, 1])
    ax_same = fig.add_subplot(gs_top[0, 2])
    ax_qq = fig.add_subplot(gs_top[0, 3])
    cax = fig.add_subplot(gs_top[0, 4])
    for ax in (ax_hist, ax_obs, ax_same, ax_qq):
        ax.tick_params(labelsize=8)
        ax.title.set_fontsize(10)

    if obs_pos.size:
        ax_hist.hist(obs_pos, bins=bins, density=True, alpha=0.5, label="obs > 0")
    if model_pos.size:
        ax_hist.hist(model_pos, bins=bins, density=True, alpha=0.5, label="model > 0")
    if y_star_pos.size:
        ax_hist.hist(y_star_pos, bins=bins, density=True, alpha=0.5, label="model-equivalent")
    ax_hist.set_xlabel("SWE (kg m-2)")
    ax_hist.set_ylabel("density")
    ax_hist.set_title("Positive SWE distributions")
    if ax_hist.get_legend_handles_labels()[0]:
        ax_hist.legend(fontsize=8, loc="upper right", frameon=False)

    cmap = "viridis"
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
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    ax_obs.set_xlabel("observed SWE")
    ax_obs.set_ylabel("model-equivalent SWE")
    ax_obs.set_title("Observed vs model-equivalent")

    if both_model.any():
        color_artist = _scatter_by_water_year(
            ax_same,
            model[both_model],
            y_star[both_model],
            wy[both_model],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    ax_same.set_xlabel("same-day midnight model SWE")
    ax_same.set_ylabel("model-equivalent SWE")
    ax_same.set_title("Same-day model vs mapped obs")

    if y_star_pos.size and model_pos.size:
        n = min(y_star_pos.size, model_pos.size)
        q = np.linspace(0.0, 1.0, n, endpoint=True)
        qq_y = np.quantile(y_star_pos, q)
        qq_m = np.quantile(model_pos, q)
        ax_qq.scatter(qq_m, qq_y, s=18, alpha=0.75, color="0.35", edgecolors="none")
        _one_to_one(ax_qq, qq_m, qq_y)
        ax_qq.set_xlabel("model positive SWE quantiles")
        ax_qq.set_ylabel("mapped obs quantiles")
    ax_qq.set_title("Q-Q: mapped obs vs climatology")

    if color_artist is not None:
        cbar = fig.colorbar(color_artist, cax=cax)
        cbar.set_label("held-out water year", fontsize=8)
        cbar.ax.tick_params(labelsize=8)
    else:
        cax.set_axis_off()

    ax = fig.add_subplot(gs[1, 0])
    ax.tick_params(labelsize=8)
    model_times, model_line = _timeseries_model(full_model, times, model)
    obs_valid = np.isfinite(obs)
    if np.isfinite(model_line).any():
        ax.plot(
            model_times,
            model_line,
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
    ax.set_title("Model, observations, and model-equivalent observations", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
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
