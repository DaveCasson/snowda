"""Evaluate a finished run directory into ``<run_dir>/evaluation``.

Layout written::

    evaluation/
      evaluation.json                    what was scored, with which settings
      historic/scores_by_water_year.csv  one row per water year, window, estimate
      historic/summary.json              pooled scores per estimate and window
      historic/series.nc                 daily obs, means, quantile bands, weights
      historic/figures/wy_<year>.png     one figure per water year
      forecast/scores_by_issue.csv       one row per forecast issue date
      forecast/summary.json              scores per issue month
      forecast/metrics.nc, metrics.json  lead-time, threshold and rank scores
      da/updates.csv                     filter diagnostics per assimilation date

``historic`` is written for every run mode. ``forecast`` is written for the
ESP modes and ``da`` for the assimilating ones.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from snowda.config.pysummada import PysummadaConfig, load_pysummada_config
from snowda.config.run_dir import git_commit_hash, utc_timestamp
from snowda.config.run_modes import (
    ASSIMILATION_MODES,
    FORECAST_MODES,
    run_mode_from_provenance,
)
from snowda.evaluation import figures
from snowda.evaluation.da import update_diagnostics
from snowda.evaluation.forecast import ForecastEvaluation, evaluate_forecasts
from snowda.evaluation.historic import evaluate_historic
from snowda.evaluation.loaders import (
    load_historic_flow,
    load_observations,
    load_weight_updates,
    pair_historic,
    read_json,
)
from snowda.evaluation.weights import filter_weight_timeline
from snowda.io.esp_streamflow import discover_esp_mizuroute_outputs
from snowda.utils.logging import get_logger

log = get_logger(__name__)

EVALUATION_DIRNAME = "evaluation"


def _plain(value: Any) -> Any:
    """JSON-ready copy: paths and times as text, NaN and infinity as null."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (Path, pd.Timestamp, datetime, np.datetime64)):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2) + "\n", encoding="utf-8")
    return path


def _resolve_config(run_dir: Path, cfg: Optional[PysummadaConfig]) -> PysummadaConfig:
    if cfg is not None:
        return cfg
    config_path = read_json(run_dir / "provenance.json").get("config_path")
    if not config_path or not Path(str(config_path)).is_file():
        raise FileNotFoundError(
            f"Cannot find the config for {run_dir}: provenance.json names "
            f"{config_path!r}. Pass the run's YAML config."
        )
    return load_pysummada_config(str(config_path))


def _resolve_run_mode(
    run_dir: Path, cfg: PysummadaConfig, run_mode: Optional[str]
) -> Optional[str]:
    return (
        run_mode
        or run_mode_from_provenance(read_json(run_dir / "provenance.json"))
        or cfg.run_mode
    )


def _evaluate_historic(
    run_dir: Path, cfg: PysummadaConfig, out_dir: Path, updates
) -> tuple[Any, list[str]]:
    evaluation = cfg.evaluation
    flow = load_historic_flow(run_dir, evaluation)
    frame, observed, units = pair_historic(flow, load_observations(evaluation), evaluation)
    weights = None
    if updates is not None and frame.shape[1] > 1:
        weights = filter_weight_timeline(
            updates.weights,
            pd.DatetimeIndex(frame.index),
            resampled=updates.resampled.to_numpy(),
            season_start_month=cfg.season_start_month,
        )
    result = evaluate_historic(frame, observed, evaluation, weights=weights, units=units)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.by_water_year.to_csv(out_dir / "scores_by_water_year.csv", index=False)
    _write_json(out_dir / "summary.json", result.summary)
    result.series.to_netcdf(out_dir / "series.nc", engine="scipy")
    return result, [
        "historic/scores_by_water_year.csv",
        "historic/summary.json",
        "historic/series.nc",
    ]


def _evaluate_forecast(
    run_dir: Path, cfg: PysummadaConfig, out_dir: Path
) -> tuple[ForecastEvaluation, list[str]]:
    result = evaluate_forecasts(run_dir, cfg.evaluation, out_dir)
    result.by_issue.to_csv(out_dir / "scores_by_issue.csv", index=False)
    _write_json(out_dir / "summary.json", result.summary)
    return result, [
        "forecast/scores_by_issue.csv",
        "forecast/summary.json",
        "forecast/metrics.nc",
        "forecast/metrics.json",
    ]


def _write_figures(
    run_dir: Path,
    cfg: PysummadaConfig,
    out_dir: Path,
    historic,
    forecast: Optional[ForecastEvaluation],
    updates,
) -> list[str]:
    written: list[Path] = []
    swe = None
    if updates is not None and cfg.evaluation.figures_swe:
        try:
            from snowda.io.da_traces import load_da_swe

            swe = load_da_swe(run_dir)
        except Exception as exc:  # the SWE panel is optional
            log.warning("SWE panel skipped: %s", exc)
    if historic is not None:
        figure_dir = out_dir / "historic" / "figures"
        written.extend(
            figures.write_water_year_figures(
                historic.series,
                historic.by_water_year,
                figure_dir,
                forecasts=forecast.forecasts if forecast is not None else (),
                update_dates=list(updates.weights.index) if updates is not None else (),
                swe=swe,
                title=cfg.case_name,
            )
        )
        if not historic.by_water_year.empty:
            written.append(
                figures.save_figure(
                    figures.plot_scores_by_water_year(historic.by_water_year),
                    figure_dir / "scores_by_water_year.png",
                )
            )
        written.append(
            figures.save_figure(
                figures.plot_pit_histograms(historic.summary),
                figure_dir / "pit_histogram.png",
            )
        )
    if forecast is not None and not forecast.by_issue.empty:
        written.append(
            figures.save_figure(
                figures.plot_forecast_scores(forecast.by_issue),
                out_dir / "forecast" / "figures" / "volume_scores.png",
            )
        )
    return [str(path.relative_to(out_dir)) for path in written if path is not None]


def evaluate_run(
    run_dir: str | Path,
    cfg: Optional[PysummadaConfig] = None,
    *,
    run_mode: Optional[str] = None,
) -> dict[str, Any]:
    """Score a finished run and write ``<run_dir>/evaluation``.

    ``cfg`` defaults to the config named in the run's ``provenance.json``.
    ``run_mode`` defaults to the mode recorded there. Returns the contents of
    ``evaluation/evaluation.json``.
    """
    root = Path(run_dir)
    cfg = _resolve_config(root, cfg)
    if cfg.evaluation is None:
        raise ValueError("The config has no `evaluation` section.")
    mode = _resolve_run_mode(root, cfg, run_mode)
    out_dir = root / EVALUATION_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Evaluating %s (run_mode %s) into %s", root, mode, out_dir)

    updates = load_weight_updates(root)
    if updates is not None and mode is not None and mode not in ASSIMILATION_MODES:
        log.warning("Run mode %s has particle weights in %s; they are used.", mode, root)
    outputs: list[str] = []

    historic, written = _evaluate_historic(root, cfg, out_dir / "historic", updates)
    outputs.extend(written)

    forecast = None
    has_forecasts = bool(discover_esp_mizuroute_outputs(root))
    if mode in FORECAST_MODES and not has_forecasts:
        raise FileNotFoundError(f"Run mode {mode} but no ESP mizuRoute traces in {root}.")
    if has_forecasts:
        forecast, written = _evaluate_forecast(root, cfg, out_dir / "forecast")
        outputs.extend(written)

    if updates is not None:
        (out_dir / "da").mkdir(parents=True, exist_ok=True)
        update_diagnostics(root, updates).to_csv(out_dir / "da" / "updates.csv", index=False)
        outputs.append("da/updates.csv")

    if cfg.evaluation.figures:
        outputs.extend(_write_figures(root, cfg, out_dir, historic, forecast, updates))

    record = {
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit_hash(Path(__file__).resolve().parents[3]),
        "run_dir": str(root),
        "run_mode": mode,
        "case_name": cfg.case_name,
        "num_ens": cfg.num_ens,
        "pf_scheme": cfg.pf_scheme if updates is not None else None,
        "season_start_month": cfg.season_start_month,
        "evaluation": cfg.evaluation,
        "primary_estimate": historic.summary["primary_estimate"],
        "n_water_years": int(historic.by_water_year["water_year"].nunique())
        if not historic.by_water_year.empty
        else 0,
        "n_forecasts": int(len(forecast.by_issue)) if forecast is not None else 0,
        "outputs": outputs,
    }
    _write_json(out_dir / "evaluation.json", record)
    log.info("Evaluation complete: %s", out_dir)
    return _plain(record)


def evaluate_run_safely(
    run_dir: str | Path,
    cfg: PysummadaConfig,
    *,
    run_mode: Optional[str],
    run_log: Optional[logging.Logger] = None,
) -> Optional[dict[str, Any]]:
    """Evaluate at the end of a workflow without risking the run.

    A run without an ``evaluation`` section is skipped. A scoring error is
    logged and recorded in ``evaluation/FAILED.txt``; the model output is kept
    and can be scored again with ``python -m snowda.workflows.evaluate_run``.
    """
    logger = run_log or log
    if cfg.evaluation is None:
        logger.info("No `evaluation` section; skipping automatic evaluation.")
        return None
    failed = Path(run_dir) / EVALUATION_DIRNAME / "FAILED.txt"
    try:
        record = evaluate_run(run_dir, cfg, run_mode=run_mode)
    except Exception as exc:
        logger.exception("Automatic evaluation failed; the run output is intact.")
        failed.parent.mkdir(parents=True, exist_ok=True)
        failed.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return None
    failed.unlink(missing_ok=True)
    return record
