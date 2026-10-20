"""Evaluate ESP streamflow ensembles against gauge observations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scores
import xarray as xr

from snowda.adapters.scores import (
    DEFAULT_CRPS_METHOD,
    brier_score,
    brier_score_for_ensemble,
    crps_for_ensemble,
    false_alarm_ratio,
    probability_of_detection,
    rank_histogram,
    skill_score,
)
from snowda.config.loader import load_config
from snowda.io.esp_streamflow import (
    CLIMO_MEMBER_DIM,
    DEFAULT_RETURN_PERIODS,
    ENSEMBLE_DIM,
    EspPairedForecast,
    climatology_window_peaks,
    group_forecasts_by_init_month,
    load_esp_forecasts,
    observational_climatology_ensemble,
    peak_over_horizon,
    read_return_period_thresholds,
    select_return_period_thresholds,
)
from snowda.io.streamflow import DEFAULT_RESAMPLE
from snowda.utils.logging import configure_logging, get_logger
from snowda.workflows.evaluate_streamflow import (
    _git_commit_hash,
    _json_safe,
    _normalize_output_dir,
    _now_utc_tag,
    _require_absolute_input,
    _write_eval_netcdf,
)

log = get_logger(__name__)


def _exclude_years_for_forecast(
    forecast: EspPairedForecast,
    exclude_verification_year: bool,
) -> list[int]:
    if exclude_verification_year:
        return [int(pd.Timestamp(forecast.init_time).year)]
    return []


def _event_probability(
    values: xr.DataArray, threshold: float, dim: str
) -> xr.DataArray:
    events = xr.where(np.isfinite(values), values >= threshold, np.nan)
    return events.mean(dim=dim, skipna=True)


def _binary_from_probability(
    probability: xr.DataArray,
    event_probability_threshold: float,
) -> xr.DataArray:
    return xr.where(
        np.isfinite(probability),
        (probability >= event_probability_threshold).astype(np.int64),
        np.nan,
    )


def _binary_obs_event(obs_peak: xr.DataArray, threshold: float) -> xr.DataArray:
    return xr.where(
        np.isfinite(obs_peak), (obs_peak >= threshold).astype(np.int64), np.nan
    )


def _finite_ensemble(values: xr.DataArray, dim: str) -> xr.DataArray:
    """Drop ensemble members whose peak is missing."""
    if dim not in values.dims:
        return values
    mask = np.isfinite(np.asarray(values.values, dtype=float))
    return values.isel({dim: mask})


def _scalar_peak_crps(
    ensemble: xr.DataArray,
    obs_peak: xr.DataArray,
    *,
    member_dim: str,
    crps_method: str,
) -> xr.DataArray:
    """One CRPS for a peak-flow ensemble. NaN when the observation or ensemble is empty."""
    obs_values = np.ravel(np.asarray(obs_peak.values, dtype=float))
    obs_value = float(obs_values[0]) if obs_values.size == 1 else np.nan
    members = _finite_ensemble(ensemble, member_dim)
    if int(members.sizes.get(member_dim, 0)) == 0 or not np.isfinite(obs_value):
        return xr.DataArray(np.nan)
    return crps_for_ensemble(
        members,
        xr.DataArray(obs_value),
        ensemble_member_dim=member_dim,
        method=crps_method,
    ).astype(float)


def _score_crps_group(
    forecasts: Sequence[EspPairedForecast],
    obs: xr.DataArray,
    *,
    exclude_verification_year: bool,
    crps_method: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    crps_items: list[xr.DataArray] = []
    crpss_items: list[xr.DataArray] = []
    for forecast in forecasts:
        exclude = _exclude_years_for_forecast(forecast, exclude_verification_year)
        climo = observational_climatology_ensemble(
            obs,
            forecast.valid_time,
            exclude_years=exclude,
        )
        crps = crps_for_ensemble(
            forecast.fcst,
            forecast.obs,
            ensemble_member_dim=ENSEMBLE_DIM,
            method=crps_method,
            preserve_dims="lead",
        )
        crps_ref = crps_for_ensemble(
            climo,
            forecast.obs,
            ensemble_member_dim=CLIMO_MEMBER_DIM,
            method=crps_method,
            preserve_dims="lead",
        )
        crpss = skill_score(crps, crps_ref)
        init = pd.Timestamp(forecast.init_time)
        crps_items.append(crps.expand_dims(init_time=[init]))
        crpss_items.append(crpss.expand_dims(init_time=[init]))
    crps_by_lead = xr.concat(crps_items, dim="init_time", join="outer").mean(
        "init_time", skipna=True
    )
    crpss_by_lead = xr.concat(crpss_items, dim="init_time", join="outer").mean(
        "init_time", skipna=True
    )
    return crps_by_lead.rename("crps"), crpss_by_lead.rename("crpss")


def _stack_peaks(
    forecasts: Sequence[EspPairedForecast],
    obs: xr.DataArray,
    *,
    exclude_verification_year: bool,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    fcst_items: list[xr.DataArray] = []
    obs_items: list[xr.DataArray] = []
    climo_items: list[xr.DataArray] = []
    for forecast in forecasts:
        fcst_peak, obs_peak = peak_over_horizon(forecast.fcst, forecast.obs)
        climo_peak = climatology_window_peaks(
            obs,
            forecast.init_time,
            int(forecast.fcst.sizes["lead"]),
            exclude_years=_exclude_years_for_forecast(
                forecast, exclude_verification_year
            ),
        )
        init = pd.Timestamp(forecast.init_time)
        fcst_items.append(fcst_peak.expand_dims(init_time=[init]))
        obs_items.append(obs_peak.expand_dims(init_time=[init]))
        climo_items.append(climo_peak.expand_dims(init_time=[init]))
    fcst_peak = xr.concat(fcst_items, dim="init_time", join="outer")
    obs_peak = xr.concat(obs_items, dim="init_time", join="outer")
    climo_peak = xr.concat(climo_items, dim="init_time", join="outer")
    return fcst_peak, obs_peak, climo_peak


def _score_peak_group(
    forecasts: Sequence[EspPairedForecast],
    obs: xr.DataArray,
    *,
    exclude_verification_year: bool,
    crps_method: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """One peak-flow CRPS and CRPSS per forecast.

    The predictand is the maximum streamflow over the forecast horizon. The
    reference ensemble is the maximum observed streamflow in that same calendar
    window of each other year. CRPSS is ``1 - CRPS / CRPS_ref`` for that forecast.
    """
    crps_items: list[xr.DataArray] = []
    crpss_items: list[xr.DataArray] = []
    for forecast in forecasts:
        fcst_peak, obs_peak = peak_over_horizon(forecast.fcst, forecast.obs)
        climo_peak = climatology_window_peaks(
            obs,
            forecast.init_time,
            int(forecast.fcst.sizes["lead"]),
            exclude_years=_exclude_years_for_forecast(
                forecast, exclude_verification_year
            ),
        )
        crps = _scalar_peak_crps(
            fcst_peak,
            obs_peak,
            member_dim=ENSEMBLE_DIM,
            crps_method=crps_method,
        )
        crps_ref = _scalar_peak_crps(
            climo_peak,
            obs_peak,
            member_dim=CLIMO_MEMBER_DIM,
            crps_method=crps_method,
        )
        crpss = skill_score(crps, crps_ref)
        init = pd.Timestamp(forecast.init_time)
        crps_items.append(crps.expand_dims(init_time=[init]))
        crpss_items.append(crpss.expand_dims(init_time=[init]))
    peak_crps = xr.concat(crps_items, dim="init_time", join="outer").rename("peak_crps")
    peak_crpss = xr.concat(crpss_items, dim="init_time", join="outer").rename(
        "peak_crpss"
    )
    peak_crps.attrs["long_name"] = "CRPS of peak streamflow over the forecast horizon"
    peak_crps.attrs["units"] = str(obs.attrs.get("units", "m3 s-1"))
    peak_crpss.attrs["long_name"] = (
        "CRPSS of peak streamflow versus observational peak climatology"
    )
    peak_crpss.attrs["units"] = "1"
    return peak_crps, peak_crpss


def _score_threshold_group(
    fcst_peak: xr.DataArray,
    obs_peak: xr.DataArray,
    climo_peak: xr.DataArray,
    thresholds: xr.DataArray,
    *,
    event_probability_threshold: float,
    fair_brier: bool,
) -> xr.Dataset:
    yp_values = [
        float(thresholds.sel(return_period=t))
        for t in thresholds["return_period"].values
    ]
    return_periods = [
        int(t) if float(t).is_integer() else float(t)
        for t in thresholds["return_period"].values
    ]
    bs = brier_score_for_ensemble(
        fcst_peak,
        obs_peak,
        yp_values,
        ensemble_member_dim=ENSEMBLE_DIM,
        fair_correction=fair_brier,
        reduce_dims="init_time",
    )
    bs = (
        bs.assign_coords(threshold=("threshold", np.asarray(return_periods)))
        .rename({"threshold": "return_period"})
        .rename("bs")
    )

    p_clim_vars = []
    p_fcst_vars = []
    obs_event_vars = []
    pod_vars = []
    far_vars = []
    for period, yp in zip(return_periods, yp_values):
        p_clim = _event_probability(climo_peak, yp, CLIMO_MEMBER_DIM)
        obs_event = _binary_obs_event(obs_peak, yp).astype(float)
        p_fcst = _event_probability(fcst_peak, yp, ENSEMBLE_DIM)
        fcst_bin = _binary_from_probability(p_fcst, event_probability_threshold)
        obs_bin = _binary_obs_event(obs_peak, yp)
        pod = probability_of_detection(fcst_bin, obs_bin, reduce_dims="init_time")
        far = false_alarm_ratio(fcst_bin, obs_bin, reduce_dims="init_time")
        p_clim_vars.append(p_clim.expand_dims(return_period=[period]))
        p_fcst_vars.append(p_fcst.expand_dims(return_period=[period]))
        obs_event_vars.append(obs_event.expand_dims(return_period=[period]))
        pod_vars.append(pod.expand_dims(return_period=[period]))
        far_vars.append(far.expand_dims(return_period=[period]))

    p_clim = xr.concat(p_clim_vars, dim="return_period")
    event_probability = xr.concat(p_fcst_vars, dim="return_period").rename(
        "event_probability"
    )
    event_observed = xr.concat(obs_event_vars, dim="return_period").rename(
        "event_observed"
    )
    event_probability.attrs["long_name"] = (
        "fraction of ensemble members whose peak flow exceeds the return-period threshold"
    )
    event_observed.attrs["long_name"] = (
        "observed peak flow exceeds the return-period threshold"
    )
    bs_ref = brier_score(p_clim, event_observed, reduce_dims="init_time").rename(
        "bs_ref"
    )
    bss = skill_score(bs, bs_ref).rename("bss")
    pod = xr.concat(pod_vars, dim="return_period").rename("pod")
    far = xr.concat(far_vars, dim="return_period").rename("far")
    yp = thresholds.rename("yp")
    return xr.Dataset(
        {
            "bs": bs,
            "bs_ref": bs_ref,
            "bss": bss,
            "pod": pod,
            "far": far,
            "yp": yp,
            "event_probability": event_probability,
            "event_observed": event_observed,
        }
    )


def _forecast_rank_histogram(forecast: EspPairedForecast) -> tuple[xr.DataArray, int]:
    """Rank histogram for one ESP forecast. Leads with a missing value are omitted."""
    valid = np.isfinite(forecast.obs) & np.isfinite(forecast.fcst).all(dim=ENSEMBLE_DIM)
    lead_index = np.flatnonzero(np.asarray(valid.values, dtype=bool))
    n_valid = int(lead_index.size)
    if n_valid == 0:
        return xr.DataArray(np.nan, name="rank_histogram"), 0
    histogram = rank_histogram(
        forecast.fcst.isel(lead=lead_index),
        forecast.obs.isel(lead=lead_index),
        ensemble_member_dim=ENSEMBLE_DIM,
        reduce_dims=["lead"],
    ).rename("rank_histogram")
    return histogram, n_valid


def _spread_rank_histogram(histogram: xr.DataArray, n_bins: int) -> np.ndarray:
    """Place a rank histogram onto ``n_bins`` equal probability bins.

    A histogram whose length already equals ``n_bins`` is copied unchanged.
    A shorter histogram, from a smaller ensemble, is spread by the probability
    interval each rank occupies.
    """
    frequency = np.asarray(histogram.values, dtype=float)
    n_ranks = int(frequency.size)
    if n_ranks == n_bins:
        return frequency
    spread = np.zeros(n_bins, dtype=float)
    for rank_index, weight in enumerate(frequency):
        if not np.isfinite(weight) or weight == 0.0:
            continue
        left = rank_index / n_ranks
        right = (rank_index + 1) / n_ranks
        for bin_index in range(n_bins):
            bin_left = bin_index / n_bins
            bin_right = (bin_index + 1) / n_bins
            overlap = min(right, bin_right) - max(left, bin_left)
            if overlap > 0.0:
                spread[bin_index] += weight * overlap / (right - left)
    return spread


def _score_rank_histogram_group(
    forecasts: Sequence[EspPairedForecast],
) -> tuple[xr.DataArray, int]:
    """Pool lead-time rank histograms for forecasts that share an initiation month.

    Forecasts with fewer ensemble members are spread onto the rank axis of the
    largest ensemble in the month.
    """
    pieces: list[xr.DataArray] = []
    weights: list[int] = []
    for forecast in forecasts:
        histogram, n_valid = _forecast_rank_histogram(forecast)
        if n_valid == 0:
            continue
        pieces.append(histogram)
        weights.append(n_valid)
    if not pieces:
        n_members = int(forecasts[0].n_members)
        ranks = np.arange(1, n_members + 2)
        empty = xr.DataArray(
            np.full(ranks.size, np.nan),
            dims=("rank",),
            coords={"rank": ranks},
            name="rank_histogram",
        )
        return empty, 0
    n_bins = max(int(piece.sizes["rank"]) for piece in pieces)
    total = float(sum(weights))
    combined_values = np.zeros(n_bins, dtype=float)
    for histogram, n_valid in zip(pieces, weights):
        combined_values += _spread_rank_histogram(histogram, n_bins) * n_valid
    combined_values /= total
    combined = xr.DataArray(
        combined_values,
        dims=("rank",),
        coords={"rank": np.arange(1, n_bins + 1)},
        name="rank_histogram",
    )
    combined.attrs["long_name"] = (
        "relative frequency of the observation rank in the ensemble"
    )
    return combined, int(total)


def compute_esp_metrics(
    forecasts: Sequence[EspPairedForecast],
    obs: xr.DataArray,
    thresholds: xr.DataArray,
    *,
    return_periods: Sequence[float] = DEFAULT_RETURN_PERIODS,
    event_probability_threshold: float = 0.5,
    exclude_verification_year: bool = True,
    crps_method: str = DEFAULT_CRPS_METHOD,
    fair_brier: bool = True,
    score_peak: bool = False,
) -> xr.Dataset:
    """Score ESP forecasts and aggregate by forecast-initiation month.

    When ``score_peak`` is true, also score the maximum streamflow over each
    forecast horizon: one CRPS and one CRPSS per forecast, referenced to the
    observed peak in the same calendar window of other years. Month means of
    those scores are the mean across forecasts, matching the lead-time CRPSS.

    ``event_probability`` is the fraction of finite ensemble members whose peak
    exceeds each return-period threshold, and ``event_observed`` is the matching
    binary observation. Those two fields are the inputs to the reliability
    diagram and ROC curve. The fair-Brier correction is applied only to ``bs``.

    ``rank_histogram`` is the relative frequency of the observation's rank in
    the ensemble at each lead, pooled over forecasts that share an initiation
    month. A lead is omitted when the observation or any ensemble member is
    missing. ``rank_histogram_n`` is the number of leads that entered the histogram.
    """
    if not forecasts:
        raise ValueError("No ESP forecasts to score.")
    selected = select_return_period_thresholds(thresholds, return_periods)
    grouped = group_forecasts_by_init_month(forecasts)

    month_datasets: list[xr.Dataset] = []
    for init_month, group in grouped.items():
        crps, crpss = _score_crps_group(
            group,
            obs,
            exclude_verification_year=exclude_verification_year,
            crps_method=crps_method,
        )
        fcst_peak, obs_peak, climo_peak = _stack_peaks(
            group,
            obs,
            exclude_verification_year=exclude_verification_year,
        )
        threshold_ds = _score_threshold_group(
            fcst_peak,
            obs_peak,
            climo_peak,
            selected,
            event_probability_threshold=event_probability_threshold,
            fair_brier=fair_brier,
        )
        rank_hist, rank_n = _score_rank_histogram_group(group)
        n_pairs = int(sum(item.n_pairs for item in group))
        n_members = int(max(item.n_members for item in group))
        month_vars: dict[str, xr.DataArray] = {
            "crps": crps,
            "crpss": crpss,
            "crps_mean": crps.mean("lead", skipna=True),
            "crpss_mean": crpss.mean("lead", skipna=True),
            "rank_histogram": rank_hist,
            "rank_histogram_n": xr.DataArray(rank_n, name="rank_histogram_n"),
        }
        if score_peak:
            peak_crps, peak_crpss = _score_peak_group(
                group,
                obs,
                exclude_verification_year=exclude_verification_year,
                crps_method=crps_method,
            )
            month_vars["peak_crps"] = peak_crps
            month_vars["peak_crpss"] = peak_crpss
            month_vars["peak_crps_mean"] = peak_crps.mean(
                "init_time", skipna=True
            ).rename("peak_crps_mean")
            month_vars["peak_crpss_mean"] = peak_crpss.mean(
                "init_time", skipna=True
            ).rename("peak_crpss_mean")
        month_ds = xr.Dataset(month_vars)
        month_ds = xr.merge([month_ds, threshold_ds])
        month_ds["n_forecasts"] = n_forecasts = len(group)
        month_ds["n_pairs"] = n_pairs
        month_ds["n_members"] = n_members
        month_ds = month_ds.expand_dims(init_month=[init_month])
        month_ds.attrs["n_forecasts"] = n_forecasts
        month_datasets.append(month_ds)

    metrics = xr.concat(month_datasets, dim="init_month", join="outer")
    metrics.attrs["n_forecasts"] = int(
        sum(int(ds.attrs["n_forecasts"]) for ds in month_datasets)
    )
    metrics.attrs["n_pairs"] = int(
        sum(int(ds["n_pairs"].item()) for ds in month_datasets)
    )
    metrics.attrs["scores_version"] = getattr(scores, "__version__", "unknown")
    metrics.attrs["crps_method"] = crps_method
    metrics.attrs["fair_brier"] = str(fair_brier)
    metrics.attrs["event_probability_threshold"] = float(event_probability_threshold)
    metrics.attrs["exclude_verification_year"] = str(exclude_verification_year)
    metrics.attrs["return_periods"] = ",".join(str(t) for t in return_periods)
    metrics.attrs["skill_reference"] = "observational streamflow climatology"
    metrics.attrs["binary_event"] = "peak-over-horizon"
    metrics.attrs["score_peak"] = str(bool(score_peak))
    if score_peak:
        metrics.attrs["peak_definition"] = (
            "maximum streamflow over the forecast horizon"
        )
        metrics.attrs["peak_skill_reference"] = (
            "observational peak streamflow in the same calendar window"
        )
    return metrics


def _peak_scores_by_init(row: xr.Dataset, name: str) -> dict[str, Any]:
    """Finite per-forecast peak scores, dropping inits that belong to other months."""
    if name not in row:
        return {}
    da = row[name]
    if "init_time" not in da.dims:
        return {}
    scores_by_init: dict[str, Any] = {}
    for init in np.atleast_1d(da["init_time"].values):
        value = float(np.asarray(da.sel(init_time=init).values, dtype=float))
        if np.isfinite(value):
            scores_by_init[str(pd.Timestamp(init))] = _json_safe(value)
    return scores_by_init


def esp_metrics_to_dict(metrics: xr.Dataset) -> dict[str, Any]:
    """Nested JSON payload keyed by initiation month."""
    payload: dict[str, Any] = {str(k): _json_safe(v) for k, v in metrics.attrs.items()}
    months = [str(m) for m in np.atleast_1d(metrics["init_month"].values)]
    periods = [
        int(t) if float(t).is_integer() else float(t)
        for t in np.atleast_1d(metrics["return_period"].values)
    ]
    payload["init_months"] = months
    payload["return_periods"] = periods
    by_month: dict[str, Any] = {}
    for month in months:
        row = metrics.sel(init_month=month)
        leads = np.atleast_1d(row["lead"].values) if "lead" in row.coords else []
        crps_vals = np.atleast_1d(row["crps"].values) if "crps" in row else []
        crpss_vals = np.atleast_1d(row["crpss"].values) if "crpss" in row else []
        entry: dict[str, Any] = {
            "n_forecasts": _json_safe(np.asarray(row["n_forecasts"].values).item()),
            "n_pairs": _json_safe(np.asarray(row["n_pairs"].values).item()),
            "n_members": _json_safe(np.asarray(row["n_members"].values).item()),
            "crps_mean": _json_safe(np.asarray(row["crps_mean"].values).item()),
            "crpss_mean": _json_safe(np.asarray(row["crpss_mean"].values).item()),
            "lead": _json_safe(leads),
            "crps_by_lead": _json_safe(crps_vals),
            "crpss_by_lead": _json_safe(crpss_vals),
            "bs": {},
            "bss": {},
            "pod": {},
            "far": {},
            "yp": {},
        }
        for period in periods:
            cell = row.sel(return_period=period)
            key = str(period)
            entry["bs"][key] = _json_safe(np.asarray(cell["bs"].values).item())
            entry["bss"][key] = _json_safe(np.asarray(cell["bss"].values).item())
            entry["pod"][key] = _json_safe(np.asarray(cell["pod"].values).item())
            entry["far"][key] = _json_safe(np.asarray(cell["far"].values).item())
            entry["yp"][key] = _json_safe(np.asarray(cell["yp"].values).item())
        if "peak_crps" in row:
            entry["peak_crps_mean"] = _json_safe(
                np.asarray(row["peak_crps_mean"].values).item()
            )
            entry["peak_crpss_mean"] = _json_safe(
                np.asarray(row["peak_crpss_mean"].values).item()
            )
            entry["peak_crps"] = _peak_scores_by_init(row, "peak_crps")
            entry["peak_crpss"] = _peak_scores_by_init(row, "peak_crpss")
        by_month[month] = entry
    payload["by_init_month"] = by_month
    return payload


def _prepare_metrics_for_netcdf(metrics: xr.Dataset) -> xr.Dataset:
    ds = metrics.copy()
    if "init_month" in ds.coords:
        ds = ds.assign_coords(init_month=ds["init_month"].astype("U5"))
    for name, da in list(ds.data_vars.items()):
        if da.dtype == object:
            ds[name] = da.astype("U32")
    return ds


def write_esp_evaluation_products(
    metrics: xr.Dataset,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dict = esp_metrics_to_dict(metrics)
    _write_eval_netcdf(_prepare_metrics_for_netcdf(metrics), output_dir / "metrics.nc")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_dict, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote ESP streamflow evaluation to %s", output_dir)
    log.info("Metrics: %s", json.dumps(metrics_dict, indent=2))
    return metrics_dict


def evaluate_esp_run(
    run_dir: str | Path,
    obs_csv: str | Path,
    thresholds_csv: str | Path,
    *,
    output_dir: str | Path,
    return_periods: Sequence[float] = DEFAULT_RETURN_PERIODS,
    event_probability_threshold: float = 0.5,
    exclude_verification_year: bool = True,
    crps_method: str = DEFAULT_CRPS_METHOD,
    fair_brier: bool = True,
    score_peak: bool = False,
    resample: str | None = DEFAULT_RESAMPLE,
    variable: str = "IRFroutedRunoff",
    seg: int = 0,
    reach_id: Any | None = None,
    date_col: str | None = None,
    flow_col: str | None = None,
    case_name: str = "esp_streamflow_eval",
    config_path: str | Path | None = None,
    resolved_config: dict[str, Any] | None = None,
) -> Path:
    """Load an ESP run, score it, and write a timestamped evaluation directory."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    eval_dir = output_root / _now_utc_tag()
    eval_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(
        level=logging.INFO, log_file=eval_dir / "evaluate_esp_streamflow.log"
    )
    log.info("snowda ESP streamflow evaluation starting")
    if config_path:
        log.info("Config: %s", config_path)
    log.info("Eval dir: %s", eval_dir)
    log.info("ESP run: %s", run_dir)
    log.info("Observations: %s", obs_csv)
    log.info("Thresholds: %s", thresholds_csv)

    snapshot = resolved_config or {
        "case_name": case_name,
        "run_dir": str(run_dir),
        "obs_csv": str(obs_csv),
        "thresholds_csv": str(thresholds_csv),
        "return_periods": list(return_periods),
        "event_probability_threshold": event_probability_threshold,
        "exclude_verification_year": exclude_verification_year,
        "crps_method": crps_method,
        "fair_brier": fair_brier,
        "score_peak": score_peak,
        "resample": resample,
        "variable": variable,
        "seg": seg,
        "reach_id": reach_id,
        "date_col": date_col,
        "flow_col": flow_col,
        "output_dir": str(output_root),
    }
    (eval_dir / "config.resolved.json").write_text(
        json.dumps(_json_safe(snapshot), indent=2),
        encoding="utf-8",
    )

    forecasts, obs = load_esp_forecasts(
        run_dir,
        obs_csv,
        variable=variable,
        resample=resample,
        seg=seg,
        reach_id=reach_id,
        date_col=date_col,
        flow_col=flow_col,
    )
    thresholds = read_return_period_thresholds(thresholds_csv)
    metrics = compute_esp_metrics(
        forecasts,
        obs,
        thresholds,
        return_periods=return_periods,
        event_probability_threshold=event_probability_threshold,
        exclude_verification_year=exclude_verification_year,
        crps_method=crps_method,
        fair_brier=fair_brier,
        score_peak=score_peak,
    )
    metrics_dict = write_esp_evaluation_products(metrics, eval_dir)

    repo_root = Path(__file__).resolve().parents[3]
    provenance = {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat() + "Z",
        "git_commit": _git_commit_hash(repo_root),
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "eval_dir": str(eval_dir),
        "run_dir": str(run_dir),
        "obs_csv": str(obs_csv),
        "thresholds_csv": str(thresholds_csv),
        "n_forecasts": metrics_dict.get("n_forecasts"),
        "scores_version": metrics_dict.get("scores_version"),
        "crps_method": metrics_dict.get("crps_method"),
    }
    (eval_dir / "provenance.json").write_text(
        json.dumps(_json_safe(provenance), indent=2),
        encoding="utf-8",
    )
    log.info("Wrote provenance: %s", eval_dir / "provenance.json")
    log.info("snowda ESP streamflow evaluation complete")
    return eval_dir


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _load_esp_eval_config(config_path: str | Path) -> dict[str, Any]:
    raw = load_config(config_path)
    base_dir = Path(raw.get("_meta", {}).get("config_dir", Path(config_path).parent))
    output_dir = raw.get("output_dir") or raw.get("paths", {}).get("output_dir")
    if not output_dir:
        raise ValueError("Config must define `output_dir`.")
    run_dir = raw.get("run_dir")
    obs_csv = raw.get("obs_csv")
    thresholds_csv = raw.get("thresholds_csv")
    if not run_dir:
        raise ValueError("Config must define `run_dir`.")
    if not obs_csv:
        raise ValueError("Config must define `obs_csv`.")
    if not thresholds_csv:
        raise ValueError("Config must define `thresholds_csv`.")
    resample = raw.get("resample", DEFAULT_RESAMPLE)
    if resample in (None, "null"):
        resample = None
    return_periods = raw.get("return_periods", list(DEFAULT_RETURN_PERIODS))
    if isinstance(return_periods, (int, float)):
        return_periods = [return_periods]
    return {
        "case_name": raw.get("case_name", "esp_streamflow_eval"),
        "output_dir": _normalize_output_dir(str(output_dir), base_dir),
        "run_dir": _require_absolute_input(str(run_dir), "run_dir"),
        "obs_csv": _require_absolute_input(str(obs_csv), "obs_csv"),
        "thresholds_csv": _require_absolute_input(
            str(thresholds_csv), "thresholds_csv"
        ),
        "variable": raw.get("variable", "IRFroutedRunoff"),
        "resample": resample,
        "seg": int(raw.get("seg", 0)),
        "reach_id": raw.get("reach_id"),
        "date_col": raw.get("date_col") or raw.get("obs_date_col"),
        "flow_col": raw.get("flow_col") or raw.get("obs_flow_col"),
        "return_periods": tuple(return_periods),
        "event_probability_threshold": float(
            raw.get("event_probability_threshold", 0.5)
        ),
        "exclude_verification_year": _as_bool(
            raw.get("exclude_verification_year"), True
        ),
        "crps_method": str(raw.get("crps_method", DEFAULT_CRPS_METHOD)),
        "fair_brier": _as_bool(raw.get("fair_brier"), True),
        "score_peak": _as_bool(raw.get("score_peak"), False),
        "_raw": raw,
        "_config_path": str(Path(config_path).resolve()),
    }


def evaluate_esp_streamflow(config_path: str | Path) -> Path:
    """Run ESP evaluation from a YAML config."""
    cfg = _load_esp_eval_config(config_path)
    snapshot = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    snapshot["config_path"] = cfg["_config_path"]
    return evaluate_esp_run(
        cfg["run_dir"],
        cfg["obs_csv"],
        cfg["thresholds_csv"],
        output_dir=cfg["output_dir"],
        return_periods=cfg["return_periods"],
        event_probability_threshold=cfg["event_probability_threshold"],
        exclude_verification_year=cfg["exclude_verification_year"],
        crps_method=cfg["crps_method"],
        fair_brier=cfg["fair_brier"],
        score_peak=cfg["score_peak"],
        resample=cfg["resample"],
        variable=cfg["variable"],
        seg=cfg["seg"],
        reach_id=cfg["reach_id"],
        date_col=cfg["date_col"],
        flow_col=cfg["flow_col"],
        case_name=cfg["case_name"],
        config_path=cfg["_config_path"],
        resolved_config=snapshot,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate ESP streamflow ensembles against observations."
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    evaluate_esp_streamflow(args.config)


if __name__ == "__main__":
    main()
