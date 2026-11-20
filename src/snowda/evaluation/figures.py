"""Figures written by the automatic evaluation.

Figures are built on ``matplotlib.figure.Figure`` directly, so writing them
never changes the pyplot backend or state of a notebook that calls this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from snowda.evaluation.historic import (
    ESTIMATE_DETERMINISTIC,
    ESTIMATE_ENSEMBLE,
    ESTIMATE_WEIGHTED,
    WINDOW_SEASON,
)
from snowda.io.da_traces import DaTraceSet
from snowda.io.esp_streamflow import ENSEMBLE_DIM, EspPairedForecast
from snowda.utils.logging import get_logger
from snowda.viz.summa_da import _forecast_colors, plot_da_traces

log = get_logger(__name__)

_OBS_COLOR = "darkorange"
_ENSEMBLE_COLOR = "0.45"
_WEIGHTED_COLOR = "tab:purple"
_ISSUE_COLORS = ("tab:green", "tab:purple", "tab:red", "tab:brown", "tab:olive", "tab:cyan")
_ESTIMATE_COLORS = {
    ESTIMATE_DETERMINISTIC: "0.25",
    ESTIMATE_ENSEMBLE: _ENSEMBLE_COLOR,
    ESTIMATE_WEIGHTED: _WEIGHTED_COLOR,
}


def _new_figure(nrows: int, figsize: tuple[float, float], **kwargs) -> tuple[Figure, np.ndarray]:
    fig = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(nrows, 1, **kwargs))
    return fig, axes


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    return path


def _draw_estimate(ax, series: xr.Dataset, name: str, keep: np.ndarray, label: str) -> None:
    times = pd.DatetimeIndex(series["time"].values)[keep]
    color = _ESTIMATE_COLORS[name]
    mean = series[f"{name}_mean"].values[keep]
    if name != ESTIMATE_DETERMINISTIC:
        bands = series[f"{name}_quantiles"].values[keep]
        ax.fill_between(times, bands[:, 0], bands[:, -1], color=color, alpha=0.18, linewidth=0,
                        label=f"{label} 10–90 %")
        ax.fill_between(times, bands[:, 1], bands[:, -2], color=color, alpha=0.30, linewidth=0,
                        label=f"{label} 25–75 %")
    ax.plot(times, mean, color=color, linewidth=1.4, label=f"{label} mean" if name != ESTIMATE_DETERMINISTIC else label)


def _draw_forecast_fans(
    ax, forecasts: Sequence[EspPairedForecast], start: pd.Timestamp, end: pd.Timestamp
) -> None:
    # Same colour per issue date as the SWE panel drawn by ``plot_da_traces``.
    colors = _forecast_colors(tuple(forecasts))
    for forecast in forecasts:
        init = pd.Timestamp(forecast.init_time)
        if init < start or init > end:
            continue
        times = pd.DatetimeIndex(forecast.valid_time.values)
        values = forecast.fcst.transpose("lead", ENSEMBLE_DIM).values
        low, median, high = np.nanquantile(values, [0.10, 0.50, 0.90], axis=1)
        color = colors[forecast.init_month]
        ax.fill_between(times, low, high, color=color, alpha=0.22, linewidth=0,
                        label=f"Forecast {forecast.init_month} 10–90 %")
        ax.plot(times, median, color=color, linewidth=1.2, linestyle="--",
                label=f"Forecast {forecast.init_month} median")
        ax.axvline(init, color=color, linewidth=0.8, linestyle=":")


def plot_water_year(
    series: xr.Dataset,
    water_year: int,
    *,
    scores: Optional[pd.DataFrame] = None,
    forecasts: Sequence[EspPairedForecast] = (),
    update_dates: Sequence[pd.Timestamp] = (),
    swe: Optional[DaTraceSet] = None,
    title: str = "",
) -> Figure:
    """Streamflow of one water year: observation, ensemble bands and forecasts.

    The equally weighted ensemble is grey. The weighted (assimilated) estimate
    is blue. Forecast fans are coloured by issue date. With ``swe``, a second
    panel shows member SWE, the SWE observations and the SWE forecasts.
    """
    keep = series["water_year"].values == water_year
    times = pd.DatetimeIndex(series["time"].values)[keep]
    start, end = times.min(), times.max()
    fig, axes = _new_figure(2 if swe is not None else 1, (13, 8.5 if swe is not None else 5), sharex=True)
    ax = axes[0]
    units = series.attrs.get("units", "m3 s-1")
    if f"{ESTIMATE_DETERMINISTIC}_mean" in series:
        _draw_estimate(ax, series, ESTIMATE_DETERMINISTIC, keep, "Simulated")
    if f"{ESTIMATE_ENSEMBLE}_mean" in series:
        _draw_estimate(ax, series, ESTIMATE_ENSEMBLE, keep, "Ensemble")
    if f"{ESTIMATE_WEIGHTED}_mean" in series:
        _draw_estimate(ax, series, ESTIMATE_WEIGHTED, keep, "Weighted")
    _draw_forecast_fans(ax, forecasts, start, end)
    ax.plot(times, series["obs"].values[keep], color=_OBS_COLOR, linewidth=1.8, label="Observed")
    for when in update_dates:
        if start <= pd.Timestamp(when) <= end:
            ax.axvline(pd.Timestamp(when), color="0.15", linewidth=0.8, linestyle=":")
    ax.set_ylabel(f"Streamflow ({units})")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8)

    heading = title or f"Water year {water_year}"
    if scores is not None and not scores.empty:
        rows = scores[(scores["water_year"] == water_year) & (scores["window"] == WINDOW_SEASON)]
        parts = [
            f"{row.estimate}: KGE {row.kge:.2f}, CRPS {row.crps:.1f}, "
            f"volume {round(row.volume_error_pct) + 0.0:+.0f} %"
            for row in rows.itertuples()
        ]
        if parts:
            heading += "\nseason scores — " + "; ".join(parts)
    ax.set_title(heading, fontsize=10, loc="left")

    if swe is not None:
        try:
            plot_da_traces(swe, ax=axes[1], start=start, end=end, title="SWE")
        except ValueError as exc:
            log.warning("SWE panel skipped for water year %s: %s", water_year, exc)
    axes[-1].set_xlim(start, end)
    return fig


def plot_scores_by_water_year(scores: pd.DataFrame, units: str = "m3 s-1") -> Figure:
    """Season KGE, CRPSS and volume error for each water year and estimate."""
    panels = (
        ("kge", "KGE"),
        ("crpss", "CRPSS vs climatology"),
        ("volume_error_pct", "Volume error (%)"),
        ("peak_timing_error_days", "Peak timing error (days)"),
    )
    fig, axes = _new_figure(len(panels), (12, 10), sharex=True)
    season = scores[scores["window"] == WINDOW_SEASON]
    estimates = list(dict.fromkeys(season["estimate"]))
    width = 0.8 / max(len(estimates), 1)
    for ax, (column, label) in zip(axes, panels):
        for i, estimate in enumerate(estimates):
            rows = season[season["estimate"] == estimate]
            ax.bar(
                rows["water_year"] + (i - (len(estimates) - 1) / 2) * width,
                rows[column],
                width=width,
                color=_ESTIMATE_COLORS.get(estimate, "0.5"),
                label=estimate,
            )
        ax.axhline(0, color="0.2", linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(
        fontsize=8,
        ncols=len(estimates) or 1,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0.2,
    )
    axes[0].set_title("Melt-season scores by water year", loc="left", fontsize=10)
    axes[-1].set_xlabel("Water year")
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig


def plot_pit_histograms(summary: dict, window: str = WINDOW_SEASON) -> Optional[Figure]:
    """PIT histogram of each ensemble estimate. Flat is calibrated."""
    histograms = {
        name: np.asarray(windows[window]["pit_histogram"], dtype=float)
        for name, windows in summary.get("estimates", {}).items()
        if window in windows and "pit_histogram" in windows[window]
    }
    if not histograms:
        return None
    fig = Figure(figsize=(4.2 * len(histograms), 3.6), layout="constrained")
    FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(1, len(histograms), sharey=True))
    for ax, (name, histogram) in zip(axes, histograms.items()):
        edges = np.linspace(0, 1, histogram.size + 1)
        ax.bar(edges[:-1], histogram, width=np.diff(edges), align="edge",
               color=_ESTIMATE_COLORS.get(name, "0.5"), edgecolor="white")
        ax.axhline(1.0 / histogram.size, color="0.2", linestyle="--", linewidth=0.9)
        ax.set_title(f"{name} ({window})", fontsize=10)
        ax.set_xlabel("PIT of the observation")
    axes[0].set_ylabel("Relative frequency")
    return fig


def plot_forecast_scores(by_issue: pd.DataFrame) -> Figure:
    """Volume forecast against observation, and volume CRPSS by issue year."""
    fig = Figure(figsize=(12, 4.6), layout="constrained")
    FigureCanvasAgg(fig)
    scatter, skill = fig.subplots(1, 2)
    months = sorted(by_issue["init_month"].unique())
    scale = 1e-9
    for i, month in enumerate(months):
        rows = by_issue[by_issue["init_month"] == month]
        color = _ISSUE_COLORS[i % len(_ISSUE_COLORS)]
        scatter.scatter(rows["volume_obs"] * scale, rows["volume_median"] * scale,
                        s=22, color=color, label=f"Issued {month}")
        skill.plot(rows["init_time"].dt.year, rows["volume_crpss"], marker="o",
                   markersize=3.5, linewidth=1.0, color=color, label=f"Issued {month}")
    finite = by_issue[["volume_obs", "volume_median"]].to_numpy(dtype=float) * scale
    if np.isfinite(finite).any():
        low, high = np.nanmin(finite), np.nanmax(finite)
        scatter.plot([low, high], [low, high], color="0.2", linewidth=0.9, linestyle="--")
    scatter.set_xlabel("Observed volume (km³)")
    scatter.set_ylabel("Forecast median volume (km³)")
    scatter.set_title("Forecast-period volume", loc="left", fontsize=10)
    scatter.legend(fontsize=8)
    scatter.grid(True, alpha=0.25)
    skill.axhline(0, color="0.2", linewidth=0.8)
    skill.set_xlabel("Issue year")
    skill.set_ylabel("Volume CRPSS vs climatology")
    skill.set_title("Volume skill by issue date", loc="left", fontsize=10)
    skill.grid(True, alpha=0.25)
    return fig


def write_water_year_figures(
    series: xr.Dataset,
    scores: pd.DataFrame,
    out_dir: Path,
    *,
    forecasts: Sequence[EspPairedForecast] = (),
    update_dates: Sequence[pd.Timestamp] = (),
    swe: Optional[DaTraceSet] = None,
    title: str = "",
) -> list[Path]:
    """One ``wy_<year>.png`` for each scored water year."""
    if scores.empty:
        return []
    paths = []
    for water_year in sorted(scores["water_year"].unique()):
        fig = plot_water_year(
            series,
            int(water_year),
            scores=scores,
            forecasts=forecasts,
            update_dates=update_dates,
            swe=swe,
            title=f"{title} — water year {water_year}" if title else "",
        )
        paths.append(_save(fig, out_dir / f"wy_{int(water_year)}.png"))
    return paths


def save_figure(fig: Optional[Figure], path: Path) -> Optional[Path]:
    if fig is None:
        return None
    return _save(fig, path)
