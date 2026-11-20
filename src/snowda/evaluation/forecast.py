"""Score the ESP forecasts of a run and tabulate them per issue date.

The monthly scores come from ``compute_esp_metrics``, unchanged. This module
adds one row per issue date, which is the unit a paired comparison between
two runs needs: with the same ``esp.seed``, issue dates and ``n_members``,
two runs use the same weather years for the same forecast member, so score
differences at one issue date come from the initial states and their weights.

An unresampled (SIS) issue keeps every particle and carries its likelihood
weight on each forecast member (``esp_sampling.WEIGHTING_LIKELIHOOD``). The
CRPS, median, mean and spread here use those weights. Resampled (SIR) and
open-loop issues have equally weighted members.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from snowda.config.pysummada import StreamflowEvaluationConfig
from snowda.core.particle_filter import weighted_quantile_rows
from snowda.evaluation.historic import water_year_of
from snowda.evaluation.loaders import empirical_return_period_thresholds
from snowda.io.esp_streamflow import (
    DEFAULT_RETURN_PERIODS,
    ENSEMBLE_DIM,
    EspPairedForecast,
    group_forecasts_by_init_month,
    load_esp_forecasts,
    member_weights,
    peak_over_horizon,
    read_return_period_thresholds,
    volume_over_horizon,
)
from snowda.utils.logging import get_logger
from snowda.workflows.evaluate_esp_streamflow import (
    _score_crps_group,
    _score_peak_group,
    _score_volume_group,
    compute_esp_metrics,
    write_esp_evaluation_products,
)

log = get_logger(__name__)

# Scored for every run so forecast runs are comparable without extra settings.
FORECAST_DEFAULTS: dict[str, Any] = {
    "return_periods": DEFAULT_RETURN_PERIODS,
    "event_probability_threshold": 0.5,
    "exclude_verification_year": True,
    "crps_method": "fair",
    "fair_brier": True,
    "score_peak": True,
    "score_volume": True,
    "bootstrap_iterations": 100,
    "bootstrap_block_size": 2,
    "bootstrap_circular": True,
    "bootstrap_ci_level": 0.9,
    "bootstrap_seed": 0,
}


@dataclass(frozen=True)
class ForecastEvaluation:
    metrics: xr.Dataset
    by_issue: pd.DataFrame
    summary: dict[str, Any]
    forecasts: tuple[EspPairedForecast, ...]
    obs: xr.DataArray
    thresholds: xr.DataArray


def forecast_options(evaluation: StreamflowEvaluationConfig) -> dict[str, Any]:
    options = dict(FORECAST_DEFAULTS)
    options.update(dict(evaluation.forecast))
    options["return_periods"] = [float(t) for t in options["return_periods"]]
    return options


def _member_statistics(
    values: xr.DataArray, weights: np.ndarray | None
) -> dict[str, float]:
    """Median, mean and spread of one value per forecast member.

    ``weights`` are the likelihood weights of an unresampled (SIS) issue, or
    ``None`` for equally weighted members. The weighted median follows
    ``weighted_quantile_rows``; the spread is the weighted standard deviation.
    """
    data = np.asarray(values.transpose(ENSEMBLE_DIM).values, dtype=float)
    finite = np.isfinite(data)
    if not finite.any():
        return {"median": np.nan, "mean": np.nan, "spread": np.nan}
    if weights is None:
        sample = data[finite]
        return {
            "median": float(np.median(sample)),
            "mean": float(np.mean(sample)),
            "spread": float(np.std(sample, ddof=1)) if sample.size > 1 else np.nan,
        }
    w = np.where(finite, weights, 0.0)
    w = w / w.sum()
    mean = float(np.sum(w * np.where(finite, data, 0.0)))
    variance = float(np.sum(w * np.where(finite, data - mean, 0.0) ** 2))
    median = float(weighted_quantile_rows(data[None, :], weights, [0.5])[0, 0])
    return {"median": median, "mean": mean, "spread": float(np.sqrt(variance))}


def scores_by_issue(
    forecasts: Sequence[EspPairedForecast],
    obs: xr.DataArray,
    *,
    exclude_verification_year: bool,
    crps_method: str,
    water_year_start_month: int,
) -> pd.DataFrame:
    """One row per forecast issue date."""
    rows: list[dict[str, Any]] = []
    for init_month, group in group_forecasts_by_init_month(forecasts).items():
        kwargs = {
            "exclude_verification_year": exclude_verification_year,
            "crps_method": crps_method,
        }
        crps, crpss = _score_crps_group(group, obs, **kwargs)
        peak_crps, peak_crpss = _score_peak_group(group, obs, **kwargs)
        volume_crps, volume_crpss = _score_volume_group(group, obs, **kwargs)
        for forecast in group:
            init = pd.Timestamp(forecast.init_time)
            fcst_volume, obs_volume = volume_over_horizon(forecast.fcst, forecast.obs)
            fcst_peak, obs_peak = peak_over_horizon(forecast.fcst, forecast.obs)
            weights = member_weights(forecast.fcst)
            volume = _member_statistics(fcst_volume, weights)
            peak = _member_statistics(fcst_peak, weights)
            observed_volume = float(obs_volume)
            rows.append(
                {
                    "init_time": init,
                    "init_month": init_month,
                    "water_year": int(
                        water_year_of(pd.DatetimeIndex([init]), water_year_start_month)[0]
                    ),
                    "n_members": forecast.n_members,
                    "weighting": "likelihood" if weights is not None else "equal",
                    "n_eff": (
                        float(1.0 / np.sum(np.square(weights)))
                        if weights is not None
                        else float(forecast.n_members)
                    ),
                    "n_lead": int(forecast.fcst.sizes["lead"]),
                    "crps_mean": float(crps.sel(init_time=init).mean("lead", skipna=True)),
                    "crpss_mean": float(
                        crpss.sel(init_time=init).mean("lead", skipna=True)
                    ),
                    "volume_obs": observed_volume,
                    "volume_median": volume["median"],
                    "volume_mean": volume["mean"],
                    "volume_spread": volume["spread"],
                    "volume_error_pct": (
                        100.0 * (volume["median"] - observed_volume) / observed_volume
                        if np.isfinite(observed_volume) and observed_volume > 0
                        else np.nan
                    ),
                    "volume_crps": float(volume_crps.sel(init_time=init)),
                    "volume_crpss": float(volume_crpss.sel(init_time=init)),
                    "peak_obs": float(obs_peak),
                    "peak_median": peak["median"],
                    "peak_crps": float(peak_crps.sel(init_time=init)),
                    "peak_crpss": float(peak_crpss.sel(init_time=init)),
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["init_month", "init_time"]).reset_index(drop=True)
    return table


def summarise_by_issue_month(by_issue: pd.DataFrame) -> dict[str, Any]:
    """Mean scores and volume-forecast statistics for each issue month."""
    out: dict[str, Any] = {}
    for init_month, rows in by_issue.groupby("init_month"):
        volumes = rows[["volume_obs", "volume_median", "volume_spread"]].dropna()
        correlation = float("nan")
        spread_skill = float("nan")
        if len(volumes) >= 3 and volumes[["volume_obs", "volume_median"]].std().min() > 0:
            correlation = float(
                np.corrcoef(volumes["volume_obs"], volumes["volume_median"])[0, 1]
            )
            error = volumes["volume_median"] - volumes["volume_obs"]
            rmse = float(np.sqrt(np.mean(error**2)))
            if rmse > 0:
                spread_skill = float(
                    np.sqrt(np.mean(volumes["volume_spread"] ** 2)) / rmse
                )
        out[str(init_month)] = {
            "n_forecasts": int(len(rows)),
            "crps_mean": float(rows["crps_mean"].mean()),
            "crpss_mean": float(rows["crpss_mean"].mean()),
            "volume_crps_mean": float(rows["volume_crps"].mean()),
            "volume_crpss_mean": float(rows["volume_crpss"].mean()),
            "volume_bias_pct": float(rows["volume_error_pct"].mean()),
            "volume_mean_abs_error_pct": float(rows["volume_error_pct"].abs().mean()),
            "volume_correlation": correlation,
            "volume_spread_skill": spread_skill,
            "peak_crps_mean": float(rows["peak_crps"].mean()),
            "peak_crpss_mean": float(rows["peak_crpss"].mean()),
        }
    return out


def evaluate_forecasts(
    run_dir: str | Path,
    evaluation: StreamflowEvaluationConfig,
    output_dir: str | Path,
) -> ForecastEvaluation:
    """Score every ESP issue of ``run_dir`` and write the products to ``output_dir``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    options = forecast_options(evaluation)
    forecasts, obs = load_esp_forecasts(
        run_dir,
        evaluation.obs_csv,
        variable=evaluation.variable,
        resample=evaluation.resample,
        seg=evaluation.seg,
        reach_id=evaluation.reach_id,
        date_col=evaluation.date_col,
        flow_col=evaluation.flow_col,
    )
    if evaluation.thresholds_csv is not None:
        thresholds = read_return_period_thresholds(evaluation.thresholds_csv)
    else:
        thresholds = empirical_return_period_thresholds(obs, options["return_periods"])
        log.info(
            "No `evaluation.thresholds_csv`; using empirical return-period "
            "thresholds from the observed annual maxima."
        )
    metrics = compute_esp_metrics(forecasts, obs, thresholds, **options)
    metrics.attrs["thresholds_source"] = str(thresholds.attrs.get("source", ""))
    write_esp_evaluation_products(metrics, out_dir)
    by_issue = scores_by_issue(
        forecasts,
        obs,
        exclude_verification_year=bool(options["exclude_verification_year"]),
        crps_method=str(options["crps_method"]),
        water_year_start_month=evaluation.water_year_start_month,
    )
    summary = {
        "n_forecasts": int(len(by_issue)),
        "n_members": int(max((f.n_members for f in forecasts), default=0)),
        "ensemble_dim": ENSEMBLE_DIM,
        "crps_method": str(options["crps_method"]),
        "skill_reference": "observational streamflow climatology",
        "thresholds_source": metrics.attrs["thresholds_source"],
        "by_issue_month": summarise_by_issue_month(by_issue),
    }
    return ForecastEvaluation(
        metrics=metrics,
        by_issue=by_issue,
        summary=summary,
        forecasts=tuple(forecasts),
        obs=obs,
        thresholds=thresholds,
    )
