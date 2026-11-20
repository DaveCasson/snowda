"""Score historic streamflow per water year and pooled over the period.

One code path covers a deterministic run (one member), an ensemble (equal
weights) and an SIS-weighted ensemble (filter weights). Scores are written
one row per water year, scoring window and estimate, so runs can be compared
afterwards on the water years they share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr

from snowda.adapters.scores import compute_deterministic_metrics
from snowda.config.pysummada import StreamflowEvaluationConfig
from snowda.evaluation.ensemble_scores import (
    PIT_BINS,
    interval_coverage,
    pit_histogram,
    spread_skill_ratio,
    weighted_crps,
    weighted_pit,
)
from snowda.evaluation.weights import (
    QUANTILES,
    effective_weights,
    n_eff,
    weighted_mean,
    weighted_quantiles,
)

SECONDS_PER_DAY = 86400.0
WINDOW_FULL = "water_year"
WINDOW_SEASON = "season"
# Equal weights on every member. For a deterministic run, the single trace.
ESTIMATE_ENSEMBLE = "ensemble"
# Filter weights from the assimilation applied to the same members.
ESTIMATE_WEIGHTED = "weighted"
ESTIMATE_DETERMINISTIC = "deterministic"

_CENTRAL_METRICS = ("kge", "rho", "alpha", "beta", "nse", "pbias", "pearsonr", "rmse", "mae")


@dataclass(frozen=True)
class HistoricEvaluation:
    """Tables and series from one historic evaluation."""

    by_water_year: pd.DataFrame
    summary: dict[str, Any]
    series: xr.Dataset


def water_year_of(times: pd.DatetimeIndex, start_month: int) -> np.ndarray:
    """Water year of each time, named by the calendar year it ends in."""
    years = np.asarray(times.year)
    if int(start_month) == 1:
        return years
    return np.where(np.asarray(times.month) >= int(start_month), years + 1, years)


def in_season(times: pd.DatetimeIndex, season_months: tuple[int, int]) -> np.ndarray:
    """True for times whose month is in the inclusive season window."""
    first, last = (int(month) for month in season_months)
    months = np.asarray(times.month)
    if first <= last:
        return (months >= first) & (months <= last)
    return (months >= first) | (months <= last)


def climatology_reference(
    obs: pd.Series, times: pd.DatetimeIndex, water_years: np.ndarray
) -> np.ndarray:
    """Observed flow on the same calendar day in every other water year.

    Shape ``(time, water_year)``. The column of a time's own water year is
    NaN, so the verifying observation is never in its own reference.
    """
    frame = pd.DataFrame(
        {
            "value": obs.to_numpy(dtype=float),
            "wy": water_years,
            "key": times.month * 100 + times.day,
        }
    ).dropna(subset=["value"])
    table = frame.pivot_table(index="key", columns="wy", values="value", aggfunc="mean")
    keys = np.asarray(times.month * 100 + times.day)
    reference = table.reindex(keys).to_numpy(dtype=float)
    own = np.asarray(table.columns)[None, :] == water_years[:, None]
    reference[own] = np.nan
    return reference


def _central_metrics(sim: np.ndarray, obs: np.ndarray, times: pd.DatetimeIndex) -> dict[str, float]:
    empty = {name: float("nan") for name in _CENTRAL_METRICS}
    valid = np.isfinite(sim) & np.isfinite(obs)
    if valid.sum() < 2 or np.nanstd(obs[valid]) == 0:
        return empty
    coords = {"time": times[valid]}
    try:
        metrics = compute_deterministic_metrics(
            xr.DataArray(sim[valid], dims=("time",), coords=coords),
            xr.DataArray(obs[valid], dims=("time",), coords=coords),
        )
    except ValueError:
        return empty
    return {
        name: float(metrics[name]) if name in metrics else float("nan")
        for name in _CENTRAL_METRICS
    }


def _scalar_crps(values: np.ndarray, weights: np.ndarray, obs: float) -> float:
    w = effective_weights(values[None, :], weights[None, :])
    return float(weighted_crps(values[None, :], w, np.asarray([obs]))[0])


def _window_signatures(
    values: np.ndarray,
    weights: np.ndarray,
    obs: np.ndarray,
    times: pd.DatetimeIndex,
) -> dict[str, float]:
    """Volume, peak and timing of one window, for the members and the observation.

    Only days with an observation are used. Member-level quantities (peak,
    peak date, centre of volume) are combined with the weights in force on
    the window's last day. The peak follows the last snow observation of the
    season, so these are the filter weights that apply at the peak.
    """
    paired = np.isfinite(obs) & np.isfinite(values).any(axis=1)
    out = {
        name: float("nan")
        for name in (
            "volume_obs",
            "volume_sim",
            "volume_error_pct",
            "volume_crps",
            "peak_obs",
            "peak_sim",
            "peak_error_pct",
            "peak_crps",
            "peak_timing_error_days",
            "centre_of_volume_error_days",
        )
    }
    if paired.sum() < 2:
        return out
    y = obs[paired]
    x = values[paired]
    days = np.asarray((times[paired] - times[paired][0]) / pd.Timedelta(days=1), dtype=float)
    last_weights = weights[paired][-1]
    member_ok = np.isfinite(x).all(axis=0) & np.isfinite(last_weights)
    if not member_ok.any():
        return out
    x = x[:, member_ok]
    w = last_weights[member_ok]
    w = w / w.sum() if w.sum() > 0 else np.full(w.size, 1.0 / w.size)

    volume_obs = float(y.sum() * SECONDS_PER_DAY)
    member_volume = x.sum(axis=0) * SECONDS_PER_DAY
    out["volume_obs"] = volume_obs
    out["volume_sim"] = float(member_volume @ w)
    if volume_obs > 0:
        out["volume_error_pct"] = 100.0 * (out["volume_sim"] - volume_obs) / volume_obs
    out["volume_crps"] = _scalar_crps(member_volume, w, volume_obs)

    peak_obs = float(y.max())
    member_peak = x.max(axis=0)
    out["peak_obs"] = peak_obs
    out["peak_sim"] = float(member_peak @ w)
    if peak_obs > 0:
        out["peak_error_pct"] = 100.0 * (out["peak_sim"] - peak_obs) / peak_obs
    out["peak_crps"] = _scalar_crps(member_peak, w, peak_obs)
    out["peak_timing_error_days"] = float(
        days[np.argmax(x, axis=0)] @ w - days[int(np.argmax(y))]
    )

    def centre(series: np.ndarray) -> float:
        total = series.sum()
        if not total > 0:
            return float("nan")
        return float(days[int(np.searchsorted(np.cumsum(series), 0.5 * total))])

    member_centre = np.asarray([centre(x[:, i]) for i in range(x.shape[1])])
    obs_centre = centre(y)
    if np.isfinite(obs_centre) and np.isfinite(member_centre).all():
        out["centre_of_volume_error_days"] = float(member_centre @ w - obs_centre)
    return out


def _probabilistic(
    values: np.ndarray,
    weights: np.ndarray,
    obs: np.ndarray,
    crps: np.ndarray,
    crps_reference: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    valid = np.isfinite(crps) & np.isfinite(crps_reference)
    out = {
        "crps": float("nan"),
        "crps_climatology": float("nan"),
        "crpss": float("nan"),
        "spread_skill": float("nan"),
        "coverage_80": float("nan"),
    }
    if not valid.any():
        return out
    out["crps"] = float(np.mean(crps[valid]))
    out["crps_climatology"] = float(np.mean(crps_reference[valid]))
    if out["crps_climatology"] > 0:
        out["crpss"] = 1.0 - out["crps"] / out["crps_climatology"]
    if values.shape[1] > 1:
        out["spread_skill"] = spread_skill_ratio(values, weights, obs)
        out["coverage_80"] = interval_coverage(lower, upper, obs)
    return out


def _across_years(rows: pd.DataFrame) -> dict[str, float]:
    """Summaries of the per-water-year signatures."""

    def mean_abs(name: str) -> float:
        column = rows[name].dropna()
        return float(column.abs().mean()) if len(column) else float("nan")

    def mean(name: str) -> float:
        column = rows[name].dropna()
        return float(column.mean()) if len(column) else float("nan")

    volumes = rows[["volume_obs", "volume_sim"]].dropna()
    correlation = float("nan")
    if len(volumes) >= 3 and volumes.std().min() > 0:
        correlation = float(np.corrcoef(volumes["volume_obs"], volumes["volume_sim"])[0, 1])
    return {
        "volume_bias_pct": mean("volume_error_pct"),
        "volume_mean_abs_error_pct": mean_abs("volume_error_pct"),
        "volume_correlation": correlation,
        "volume_crps": mean("volume_crps"),
        "peak_bias_pct": mean("peak_error_pct"),
        "peak_mean_abs_error_pct": mean_abs("peak_error_pct"),
        "peak_crps": mean("peak_crps"),
        "peak_timing_mean_abs_error_days": mean_abs("peak_timing_error_days"),
        "centre_of_volume_mean_abs_error_days": mean_abs("centre_of_volume_error_days"),
    }


def evaluate_historic(
    flow: pd.DataFrame,
    obs: pd.Series,
    evaluation: StreamflowEvaluationConfig,
    *,
    weights: Optional[pd.DataFrame] = None,
    units: str = "m3 s-1",
) -> HistoricEvaluation:
    """Score a ``(time, member)`` simulation against observations.

    ``flow`` and ``obs`` share one daily index. ``weights`` is a daily
    ``(time, member)`` frame of filter weights; when given, the run is scored
    twice, as the equally weighted ensemble and as the weighted one, on the
    same members. A water year or season enters the table only when at least
    ``evaluation.min_coverage`` of its days are paired.
    """
    times = pd.DatetimeIndex(flow.index)
    values = flow.to_numpy(dtype=float)
    y = obs.reindex(times).to_numpy(dtype=float)
    members = [str(member) for member in flow.columns]
    deterministic = values.shape[1] == 1
    water_years = water_year_of(times, evaluation.water_year_start_month)
    season = in_season(times, evaluation.season_months)

    estimates: dict[str, np.ndarray] = {
        ESTIMATE_DETERMINISTIC if deterministic else ESTIMATE_ENSEMBLE: effective_weights(
            values, None
        )
    }
    if weights is not None and not deterministic:
        aligned = weights.reindex(index=times, columns=members).to_numpy(dtype=float)
        if np.isnan(aligned).any():
            raise ValueError(
                "Weights do not cover every member and day of the simulation."
            )
        estimates[ESTIMATE_WEIGHTED] = effective_weights(values, aligned)
    primary = ESTIMATE_WEIGHTED if ESTIMATE_WEIGHTED in estimates else next(iter(estimates))

    reference = climatology_reference(pd.Series(y, index=times), times, water_years)
    crps_reference = weighted_crps(reference, effective_weights(reference, None), y)

    series: dict[str, Any] = {"obs": ("time", y), "crps_climatology": ("time", crps_reference)}
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for name, w in estimates.items():
        central = weighted_mean(values, w)
        central = np.where(np.isfinite(values).any(axis=1), central, np.nan)
        bands = weighted_quantiles(values, w, QUANTILES)
        crps = weighted_crps(values, w, y)
        pit_low, pit_high = weighted_pit(values, w, y)
        lower, upper = bands[:, 0], bands[:, -1]
        series[f"{name}_mean"] = ("time", central)
        series[f"{name}_quantiles"] = (("time", "quantile"), bands)
        series[f"{name}_crps"] = ("time", crps)
        if name == ESTIMATE_WEIGHTED:
            series["weights"] = (("time", "member"), w)
            series["n_eff"] = ("time", n_eff(w))

        scored = np.zeros(times.size, dtype=bool)
        scored_season = np.zeros(times.size, dtype=bool)
        estimate_rows: list[dict[str, Any]] = []
        for water_year in np.unique(water_years):
            in_year = water_years == water_year
            for window, mask in (
                (WINDOW_FULL, in_year),
                (WINDOW_SEASON, in_year & season),
            ):
                n_days = int(mask.sum())
                paired = mask & np.isfinite(y) & np.isfinite(central)
                expected = _expected_days(
                    int(water_year), window, evaluation
                )
                if n_days == 0 or paired.sum() < evaluation.min_coverage * expected:
                    continue
                if window == WINDOW_FULL:
                    scored |= paired
                else:
                    scored_season |= paired
                row: dict[str, Any] = {
                    "water_year": int(water_year),
                    "window": window,
                    "estimate": name,
                    "n_days": int(paired.sum()),
                }
                row.update(_central_metrics(central[paired], y[paired], times[paired]))
                row.update(
                    _probabilistic(
                        values[paired],
                        w[paired],
                        y[paired],
                        crps[paired],
                        crps_reference[paired],
                        lower[paired],
                        upper[paired],
                    )
                )
                row.update(_window_signatures(values[mask], w[mask], y[mask], times[mask]))
                estimate_rows.append(row)
        rows.extend(estimate_rows)

        table = pd.DataFrame(estimate_rows)
        summary[name] = {}
        for window, mask in ((WINDOW_FULL, scored), (WINDOW_SEASON, scored_season)):
            if not mask.any():
                continue
            pooled: dict[str, Any] = {
                "n_days": int(mask.sum()),
                "n_water_years": int((table["window"] == window).sum()),
            }
            pooled.update(_central_metrics(central[mask], y[mask], times[mask]))
            pooled.update(
                _probabilistic(
                    values[mask],
                    w[mask],
                    y[mask],
                    crps[mask],
                    crps_reference[mask],
                    lower[mask],
                    upper[mask],
                )
            )
            pooled.update(_across_years(table[table["window"] == window]))
            if not deterministic:
                pooled["pit_histogram"] = pit_histogram(
                    pit_low[mask], pit_high[mask], PIT_BINS
                ).tolist()
            summary[name][window] = pooled

    dataset = xr.Dataset(
        series,
        coords={
            "time": times.to_numpy(),
            "member": members,
            "quantile": list(QUANTILES),
        },
    )
    dataset["water_year"] = ("time", water_years)
    dataset.attrs.update(
        {
            "units": units,
            "primary_estimate": primary,
            "water_year_start_month": int(evaluation.water_year_start_month),
            "season_months": ",".join(str(m) for m in evaluation.season_months),
            "crps": "weighted empirical CRPS; equals absolute error for one member",
            "crps_reference": (
                "observed flow on the same calendar day in the other water years"
            ),
        }
    )
    by_water_year = pd.DataFrame(rows)
    if not by_water_year.empty:
        by_water_year = by_water_year.sort_values(
            ["estimate", "window", "water_year"]
        ).reset_index(drop=True)
    return HistoricEvaluation(
        by_water_year=by_water_year,
        summary={
            "primary_estimate": primary,
            "n_members": len(members),
            "units": units,
            "water_year_start_month": int(evaluation.water_year_start_month),
            "season_months": list(evaluation.season_months),
            "min_coverage": float(evaluation.min_coverage),
            "period": [str(times.min()), str(times.max())] if times.size else [],
            "estimates": summary,
        },
        series=dataset,
    )


def _expected_days(
    water_year: int, window: str, evaluation: StreamflowEvaluationConfig
) -> int:
    """Number of days in a complete water year or season window."""
    start_month = int(evaluation.water_year_start_month)
    first_year = water_year if start_month == 1 else water_year - 1
    start = pd.Timestamp(year=first_year, month=start_month, day=1)
    days = pd.date_range(start, start + pd.DateOffset(years=1), inclusive="left")
    if window == WINDOW_FULL:
        return int(days.size)
    return int(in_season(days, evaluation.season_months).sum())
