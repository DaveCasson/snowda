"""Plot ESP streamflow ensembles and evaluation metrics from in-memory arrays."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from snowda.io.esp_streamflow import ENSEMBLE_DIM, EspPairedForecast

QUANTILE_OUTER = (0.10, 0.90)
QUANTILE_INNER = (0.25, 0.75)
THRESHOLD_LINESTYLES = (":", "--", "-.")
THRESHOLD_COLOR = "0.2"


def _ensure_axes(
    ax: Axes | None, *, figsize: tuple[float, float]
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.figure, ax


def plot_crps_by_lead(
    metrics: xr.Dataset,
    *,
    ax: Axes | None = None,
    skill: bool = False,
) -> tuple[Figure, Axes]:
    """CRPS or CRPSS versus lead time, one line per initiation month."""
    created = ax is None
    fig, ax = _ensure_axes(ax, figsize=(10, 4.5))
    var = "crpss" if skill else "crps"
    if var not in metrics:
        raise KeyError(f"Metrics dataset is missing {var!r}.")
    months = [str(m) for m in np.atleast_1d(metrics["init_month"].values)]
    for month in months:
        row = metrics[var].sel(init_month=month)
        lead = np.asarray(row["lead"].values)
        values = np.asarray(row.values, dtype=float)
        ax.plot(lead, values, marker="o", markersize=3, linewidth=1.6, label=month)
    ax.set_xlabel("Lead (days)")
    ax.set_ylabel("CRPSS" if skill else "CRPS")
    ax.set_title("CRPSS vs lead" if skill else "CRPS vs lead")
    ax.grid(True, alpha=0.25)
    if skill:
        ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
    if len(months) > 1:
        ax.legend(title="Init month", loc="best")
    if ax is None or created:
        fig.tight_layout()
    return fig, ax


def _peak_series(da: xr.DataArray) -> list[tuple[str, pd.DatetimeIndex, np.ndarray]]:
    """Finite peak scores grouped by initiation month, in initialization-time order."""
    if "init_time" not in da.dims:
        raise ValueError(
            f"{da.name or 'peak score'} has no init_time dimension. "
            "Enable score_peak to store one peak score per forecast."
        )
    if "init_month" in da.dims:
        groups = [
            (str(month), da.sel(init_month=month))
            for month in np.atleast_1d(da["init_month"].values)
        ]
    else:
        groups = [("forecast", da)]

    series: list[tuple[str, pd.DatetimeIndex, np.ndarray]] = []
    for label, row in groups:
        times = pd.to_datetime(np.atleast_1d(row["init_time"].values))
        values = np.atleast_1d(np.asarray(row.values, dtype=float))
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        order = np.argsort(times[finite])
        series.append(
            (label, pd.DatetimeIndex(times[finite][order]), values[finite][order])
        )
    if not series:
        raise ValueError(f"{da.name or 'peak score'} has no finite values to plot.")
    return series


def plot_peak_scores(
    metrics: xr.Dataset,
    *,
    fig: Figure | None = None,
) -> Figure:
    """Peak-flow CRPS and CRPSS versus forecast initialization time.

    One point per initialization. Forecasts that share an initiation month
    use the same color. Non-finite scores from other months are omitted.
    """
    if "peak_crps" not in metrics or "peak_crpss" not in metrics:
        raise KeyError(
            "Metrics dataset is missing peak_crps and peak_crpss. "
            "Recompute with score_peak=True."
        )
    crps_series = _peak_series(metrics["peak_crps"])
    crpss_series = _peak_series(metrics["peak_crpss"])
    if fig is None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    else:
        axes = np.asarray(fig.axes).reshape(2)
    panels = (
        (axes[0], crps_series, "Peak CRPS", "Peak CRPS vs initialization"),
        (axes[1], crpss_series, "Peak CRPSS", "Peak CRPSS vs initialization"),
    )
    for ax, series, ylabel, title in panels:
        for label, times, values in series:
            ax.plot(
                times,
                values,
                marker="o",
                markersize=5,
                linewidth=1.6,
                label=label,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[1].axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Initialization time")
    if len(crps_series) > 1:
        axes[0].legend(title="Init month", loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_threshold_metrics(
    metrics: xr.Dataset,
    *,
    fig: Figure | None = None,
) -> Figure:
    """Grouped bars of Brier score and Brier skill score versus return period."""
    names = ("bs", "bss")
    titles = ("Brier score", "Brier skill score")
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    else:
        axes = np.asarray(fig.axes).reshape(2)
    months = [str(m) for m in np.atleast_1d(metrics["init_month"].values)]
    periods = np.atleast_1d(metrics["return_period"].values)
    x = np.arange(len(periods))
    width = 0.8 / max(len(months), 1)
    for ax, name, title in zip(np.atleast_1d(axes), names, titles):
        for i, month in enumerate(months):
            values = np.asarray(metrics[name].sel(init_month=month).values, dtype=float)
            ax.bar(x + i * width, values, width=width, label=month)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        if name == "bss":
            ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
        ax.set_xticks(x + width * (len(months) - 1) / 2)
        ax.set_xticklabels(_return_period_tick_labels(metrics, periods))
        ax.set_xlabel("Return period (years)")
    np.atleast_1d(axes)[0].legend(title="Init month", loc="best")
    fig.tight_layout()
    return fig


def _panel_axes(
    n_panels: int, fig: Figure | None, figsize: tuple[float, float]
) -> tuple[Figure, np.ndarray]:
    ncols = 1 if n_panels <= 1 else 2
    nrows = int(np.ceil(n_panels / ncols))
    if fig is None:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
        )
    else:
        axes = np.asarray(fig.axes).reshape(nrows, ncols)
    return fig, axes


def _diagram_periods(diagram: xr.Dataset) -> list[float | None]:
    if "return_period" not in diagram.dims:
        return [None]
    return [float(period) for period in np.atleast_1d(diagram["return_period"].values)]


def _diagram_months(diagram: xr.Dataset) -> list[str]:
    if "init_month" not in diagram.dims:
        return ["forecast"]
    return [str(month) for month in np.atleast_1d(diagram["init_month"].values)]


def _select_diagram_series(
    diagram: xr.Dataset,
    period: float | None,
    month: str,
) -> xr.Dataset:
    selected = diagram if period is None else diagram.sel(return_period=period)
    if "init_month" in selected.dims:
        return selected.sel(init_month=month)
    return selected


def plot_reliability_diagram(
    diagram: xr.Dataset,
    *,
    fig: Figure | None = None,
) -> Figure:
    """Reliability diagram, one panel per return period and one line per month.

    Empty bins are omitted. Marker area is proportional to the number of
    forecasts in the bin. The dotted line is that series' observed event
    frequency.
    """
    required = ("mean_forecast", "observed_frequency", "count")
    missing = [name for name in required if name not in diagram]
    if missing:
        raise KeyError(f"Reliability diagram is missing {missing}.")
    periods = _diagram_periods(diagram)
    months = _diagram_months(diagram)
    ncols = 1 if len(periods) <= 1 else 2
    nrows = int(np.ceil(len(periods) / ncols))
    fig, axes = _panel_axes(len(periods), fig, (5.0 * ncols, 4.4 * nrows))
    cmap = plt.get_cmap("tab10")
    colors = {month: cmap(i % cmap.N) for i, month in enumerate(months)}
    for panel, (ax, period) in enumerate(zip(axes.ravel(), periods)):
        ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            color="0.45",
            linestyle="--",
            linewidth=0.8,
            label="1:1",
        )
        drawn = []
        for month in months:
            series = _select_diagram_series(diagram, period, month)
            forecast = np.asarray(series["mean_forecast"].values, dtype=float).ravel()
            frequency = np.asarray(
                series["observed_frequency"].values, dtype=float
            ).ravel()
            counts = np.asarray(series["count"].values, dtype=float).ravel()
            keep = np.isfinite(forecast) & np.isfinite(frequency) & (counts > 0)
            if not np.any(keep):
                continue
            forecast = forecast[keep]
            frequency = frequency[keep]
            counts = counts[keep]
            order = np.argsort(forecast)
            drawn.append(
                (month, forecast[order], frequency[order], counts[order], series)
            )
        max_count = max((float(np.max(item[3])) for item in drawn), default=1.0)
        for month, forecast, frequency, counts, series in drawn:
            color = colors[month]
            ax.plot(forecast, frequency, color=color, linewidth=1.4, label=month)
            ax.scatter(
                forecast,
                frequency,
                s=28.0 + 140.0 * (counts / max_count),
                color=color,
                zorder=3,
            )
            if "event_frequency" in series:
                rate = float(np.asarray(series["event_frequency"].values, dtype=float))
                if np.isfinite(rate):
                    ax.axhline(
                        rate,
                        color=color,
                        linestyle=":",
                        linewidth=0.9,
                        label="_nolegend_",
                    )
        title = "Reliability" if period is None else f"{_period_text(period)}-year"
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Forecast probability")
        ax.set_ylabel("Observed frequency")
        ax.grid(True, alpha=0.25)
        if drawn:
            ax.legend(title="Init month", loc="lower right", fontsize=8)
        elif panel == 0:
            ax.legend(loc="lower right", fontsize=8)
    for ax in axes.ravel()[len(periods) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_roc_diagram(
    diagram: xr.Dataset,
    *,
    fig: Figure | None = None,
) -> Figure:
    """ROC diagram of hit rate versus false-alarm rate.

    One panel per return period and one curve per initiation month. The legend
    shows the area under the curve.
    """
    required = ("POD", "POFD")
    missing = [name for name in required if name not in diagram]
    if missing:
        raise KeyError(f"ROC diagram is missing {missing}.")
    periods = _diagram_periods(diagram)
    months = _diagram_months(diagram)
    ncols = 1 if len(periods) <= 1 else 2
    nrows = int(np.ceil(len(periods) / ncols))
    fig, axes = _panel_axes(len(periods), fig, (5.0 * ncols, 4.4 * nrows))
    cmap = plt.get_cmap("tab10")
    colors = {month: cmap(i % cmap.N) for i, month in enumerate(months)}
    for ax, period in zip(axes.ravel(), periods):
        ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            color="0.45",
            linestyle="--",
            linewidth=0.8,
            label="No skill",
        )
        drawn = False
        for month in months:
            series = _select_diagram_series(diagram, period, month)
            pofd = np.asarray(series["POFD"].values, dtype=float).ravel()
            pod = np.asarray(series["POD"].values, dtype=float).ravel()
            keep = np.isfinite(pofd) & np.isfinite(pod)
            if not np.any(keep):
                continue
            label = month
            if "AUC" in series:
                auc = float(np.asarray(series["AUC"].values, dtype=float))
                if np.isfinite(auc):
                    label = f"{month} (AUC {auc:.2f})"
            color = colors[month]
            ax.plot(
                pofd[keep],
                pod[keep],
                color=color,
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                label=label,
            )
            drawn = True
        title = "ROC" if period is None else f"{_period_text(period)}-year"
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("False alarm rate")
        ax.set_ylabel("Probability of detection")
        ax.grid(True, alpha=0.25)
        if drawn:
            ax.legend(loc="lower right", fontsize=8)
    for ax in axes.ravel()[len(periods) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_rank_histogram(
    metrics: xr.Dataset,
    *,
    fig: Figure | None = None,
) -> Figure:
    """Rank histogram of the observation within the ESP ensemble.

    One panel per initiation month. The dashed line is the frequency of a
    flat, calibrated ensemble.
    """
    if "rank_histogram" not in metrics:
        raise KeyError("Metrics dataset is missing rank_histogram.")
    histogram = metrics["rank_histogram"]
    if "init_month" in histogram.dims:
        months = [str(month) for month in np.atleast_1d(histogram["init_month"].values)]
    else:
        months = ["forecast"]
    n_months = len(months)
    if fig is None:
        fig, axes = plt.subplots(
            1,
            n_months,
            figsize=(4.2 * n_months, 3.8),
            squeeze=False,
            sharey=True,
        )
    else:
        axes = np.asarray(fig.axes).reshape(1, n_months)
    for ax, month in zip(axes.ravel(), months):
        row = (
            histogram.sel(init_month=month)
            if "init_month" in histogram.dims
            else histogram
        )
        ranks = np.asarray(row["rank"].values)
        frequency = np.asarray(row.values, dtype=float)
        keep = np.isfinite(frequency)
        ax.bar(ranks[keep], frequency[keep], width=0.8, color="C0")
        n_bins = int(np.sum(keep))
        if n_bins:
            ax.axhline(1.0 / n_bins, color="0.3", linestyle="--", linewidth=0.9)
        title = month
        if "rank_histogram_n" in metrics:
            sample_size = metrics["rank_histogram_n"]
            if "init_month" in sample_size.dims:
                sample_size = sample_size.sel(init_month=month)
            n_cases = int(np.asarray(sample_size.values))
            title = f"{month} (n={n_cases})"
        ax.set_title(title)
        ax.set_xlabel("Observation rank")
        ax.set_ylabel("Frequency")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _ensemble_quantiles(fcst: xr.DataArray) -> pd.DataFrame:
    q = fcst.quantile(
        [
            QUANTILE_OUTER[0],
            QUANTILE_INNER[0],
            0.5,
            QUANTILE_INNER[1],
            QUANTILE_OUTER[1],
        ],
        dim=ENSEMBLE_DIM,
    )
    times = pd.DatetimeIndex(pd.to_datetime(fcst["valid_time"].values))
    return pd.DataFrame(
        {
            "low": q.sel(quantile=QUANTILE_OUTER[0]).values,
            "inner_low": q.sel(quantile=QUANTILE_INNER[0]).values,
            "median": q.sel(quantile=0.5).values,
            "inner_high": q.sel(quantile=QUANTILE_INNER[1]).values,
            "high": q.sel(quantile=QUANTILE_OUTER[1]).values,
        },
        index=times,
    )


def _obs_climatology_bands(obs: xr.DataArray, times: pd.DatetimeIndex) -> pd.DataFrame:
    series = obs.to_series()
    index = pd.DatetimeIndex(series.index)
    frame = series.to_frame("flow")
    frame["month"] = index.month
    frame["day"] = index.day
    grouped = frame.groupby(["month", "day"])["flow"]
    stats = grouped.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
    stats.columns = ["p10", "p25", "p50", "p75", "p90"]
    rows = []
    for ts in times:
        key = (int(ts.month), int(ts.day))
        if key in stats.index:
            rows.append(stats.loc[key])
        else:
            rows.append(pd.Series({c: np.nan for c in stats.columns}))
    return pd.DataFrame(rows).set_index(times)


def _period_text(period: float) -> str:
    if float(period).is_integer():
        return str(int(period))
    return str(period)


def _iter_thresholds(
    thresholds: xr.DataArray | dict[float, float],
) -> list[tuple[float, float]]:
    if isinstance(thresholds, xr.DataArray):
        if (
            "return_period" not in thresholds.dims
            and "return_period" not in thresholds.coords
        ):
            raise ValueError(
                "Thresholds DataArray must have a 'return_period' coordinate."
            )
        items = []
        for period in np.atleast_1d(thresholds["return_period"].values):
            items.append((float(period), float(thresholds.sel(return_period=period))))
        return items
    return [(float(period), float(yp)) for period, yp in thresholds.items()]


def threshold_line_label(period: float, yp: float) -> str:
    return f"{_period_text(period)}-year ({yp:.0f} m3 s-1)"


def _return_period_tick_labels(metrics: xr.Dataset, periods: np.ndarray) -> list[str]:
    labels = []
    for period in periods:
        period_txt = _period_text(float(period))
        if "yp" in metrics:
            yp = metrics["yp"]
            if "init_month" in yp.dims:
                yp = yp.isel(init_month=0)
            value = float(yp.sel(return_period=period))
            labels.append(f"{period_txt} yr\n({value:.0f})")
        else:
            labels.append(period_txt)
    return labels


def draw_return_period_thresholds(
    ax: Axes,
    thresholds: xr.DataArray | dict[float, float],
) -> list:
    """Draw labeled horizontal lines for return-period flow thresholds."""
    handles = []
    for i, (period, yp) in enumerate(_iter_thresholds(thresholds)):
        style = THRESHOLD_LINESTYLES[i % len(THRESHOLD_LINESTYLES)]
        label = threshold_line_label(period, yp)
        line = ax.axhline(
            yp,
            color=THRESHOLD_COLOR,
            linestyle=style,
            linewidth=1.2,
            zorder=4,
            label=label,
        )
        ax.text(
            1.0,
            yp,
            f" {_period_text(period)}-year",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=8,
            color=THRESHOLD_COLOR,
            clip_on=False,
        )
        handles.append(line)
    return handles


def _optional_timestamp(value: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value)


def _init_month_colors(forecasts: Sequence[EspPairedForecast]) -> dict[str, tuple]:
    """One color per initialization month-day, in first-seen order."""
    cmap = plt.get_cmap("tab10")
    months = list(dict.fromkeys(forecast.init_month for forecast in forecasts))
    return {month: cmap(i % cmap.N) for i, month in enumerate(months)}


def plot_ensemble_envelopes(
    forecasts: Sequence[EspPairedForecast],
    obs: xr.DataArray | None = None,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    thresholds: xr.DataArray | dict[float, float] | None = None,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> tuple[Figure, Axes]:
    """Overlay ESP quantile envelopes, observations, climatology, and flow thresholds.

    Forecasts that share an initialization month-day use the same color and a
    single legend entry. ``start`` and ``end`` set the x-axis limits; omit
    either side to leave that limit at the data range.
    """
    fig, ax = _ensure_axes(ax, figsize=(14, 7))
    x_start = _optional_timestamp(start)
    x_end = _optional_timestamp(end)
    if x_start is not None and x_end is not None and x_start > x_end:
        raise ValueError(f"x-axis start {x_start} is after end {x_end}.")
    all_times = pd.DatetimeIndex([])
    for forecast in forecasts:
        all_times = all_times.union(
            pd.DatetimeIndex(pd.to_datetime(forecast.valid_time.values))
        )
    if all_times.empty:
        raise ValueError("No ESP valid times to plot.")

    if obs is not None:
        obs_times = pd.DatetimeIndex(pd.to_datetime(obs["time"].values))
        obs_start = x_start if x_start is not None else all_times.min()
        obs_end = x_end if x_end is not None else all_times.max()
        in_range = (obs_times >= obs_start) & (obs_times <= obs_end)
        obs_plot = obs.isel(time=in_range)
        if int(obs_plot.sizes.get("time", 0)) > 0:
            climo = _obs_climatology_bands(
                obs, pd.DatetimeIndex(pd.to_datetime(obs_plot["time"].values))
            )
            ax.fill_between(
                climo.index,
                climo["p10"],
                climo["p90"],
                color="lightgray",
                alpha=0.55,
                label="Climatology p10-p90",
            )
            ax.fill_between(
                climo.index,
                climo["p25"],
                climo["p75"],
                color="gray",
                alpha=0.35,
                label="Climatology p25-p75",
            )
            ax.plot(
                climo.index,
                climo["p50"],
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="Climatology median",
            )
            ax.plot(
                pd.DatetimeIndex(pd.to_datetime(obs_plot["time"].values)),
                np.asarray(obs_plot.values),
                color="darkorange",
                linewidth=2.0,
                label="Observed",
            )

    colors = _init_month_colors(forecasts)
    legend_handles: list[Patch] = []
    seen_months: set[str] = set()
    for forecast in forecasts:
        color = colors[forecast.init_month]
        quantiles = _ensemble_quantiles(forecast.fcst)
        ax.fill_between(
            quantiles.index,
            quantiles["low"],
            quantiles["high"],
            color=color,
            alpha=0.28,
        )
        ax.fill_between(
            quantiles.index,
            quantiles["inner_low"],
            quantiles["inner_high"],
            color=color,
            alpha=0.45,
        )
        ax.plot(quantiles.index, quantiles["median"], color=color, linewidth=1.5)
        if forecast.init_month not in seen_months:
            seen_months.add(forecast.init_month)
            legend_handles.append(
                Patch(
                    facecolor=color,
                    edgecolor="k",
                    alpha=0.5,
                    label=forecast.init_month,
                )
            )
    threshold_handles = []
    if thresholds is not None:
        threshold_handles = draw_return_period_thresholds(ax, thresholds)
    ax.set_ylabel("Streamflow (m3 s-1)")
    ax.set_title(title or "ESP streamflow ensembles")
    ax.grid(True, alpha=0.25)
    if x_start is not None or x_end is not None:
        ax.set_xlim(x_start, x_end)
    existing = ax.get_legend_handles_labels()
    threshold_labels = {handle.get_label() for handle in threshold_handles}
    base_handles = []
    base_labels = []
    for handle, label in zip(existing[0], existing[1]):
        if label in threshold_labels:
            continue
        base_handles.append(handle)
        base_labels.append(label)
    handles = base_handles + legend_handles + threshold_handles
    labels = (
        base_labels
        + [handle.get_label() for handle in legend_handles]
        + [handle.get_label() for handle in threshold_handles]
    )
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig, ax


def plot_ensemble_forecast_separate(
    forecast: EspPairedForecast,
    obs: xr.DataArray | None = None,
    *,
    ax: Axes | None = None,
    thresholds: xr.DataArray | dict[float, float] | None = None,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> tuple[Figure, Axes]:
    """Single-initiation envelope plot."""
    title = (
        f"ESP init {forecast.init_time.strftime('%Y-%m-%d')} "
        f"(month {forecast.init_month})"
    )
    return plot_ensemble_envelopes(
        [forecast],
        obs,
        ax=ax,
        title=title,
        thresholds=thresholds,
        start=start,
        end=end,
    )
