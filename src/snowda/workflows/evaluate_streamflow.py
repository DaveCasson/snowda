"""Evaluate deterministic routed streamflow against gauge observations."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr

from snowda.adapters.scores import compute_deterministic_metrics
from snowda.config.loader import load_config
from snowda.config.pysummada import StreamflowEvaluationConfig
from snowda.io.streamflow import DEFAULT_RESAMPLE, load_paired_streamflow
from snowda.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


def _now_utc_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _git_commit_hash(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _require_absolute_input(raw: str | Path, field_name: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path (got: {raw})")
    return path.resolve()


def _normalize_output_dir(raw: str | Path, base_dir: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_eval_netcdf(ds: xr.Dataset, path: Path) -> None:
    """Write a small evaluation dataset without probing a missing HDF5 path."""
    ds.to_netcdf(path, engine="scipy")


def metrics_to_dict(metrics: xr.Dataset) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, da in metrics.data_vars.items():
        payload[name] = _json_safe(np.asarray(da.values).item())
    payload.update({str(k): _json_safe(v) for k, v in metrics.attrs.items()})
    return payload


def write_evaluation_products(
    sim_nc: str | Path,
    obs_csv: str | Path,
    output_dir: str | Path,
    *,
    variable: str = "IRFroutedRunoff",
    resample: str | None = DEFAULT_RESAMPLE,
    seg: int = 0,
    reach_id: Any | None = None,
    date_col: str | None = None,
    flow_col: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    """Pair files, score metrics, and write artifacts into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sim, obs = load_paired_streamflow(
        sim_nc,
        obs_csv,
        variable=variable,
        seg=seg,
        reach_id=reach_id,
        resample=resample,
        date_col=date_col,
        flow_col=flow_col,
        start_time=start_time,
        end_time=end_time,
    )
    metrics = compute_deterministic_metrics(sim, obs)
    metrics_dict = metrics_to_dict(metrics)
    paired = xr.Dataset({"sim": sim, "obs": obs})
    paired.attrs["n_pairs"] = metrics.attrs["n_pairs"]
    _write_eval_netcdf(paired, output_dir / "paired_streamflow.nc")
    _write_eval_netcdf(metrics, output_dir / "metrics.nc")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_dict, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote streamflow evaluation to %s", output_dir)
    log.info("Metrics: %s", json.dumps(metrics_dict, indent=2))
    return metrics_dict


def evaluate_deterministic_run(
    sim_nc: str | Path,
    evaluation: StreamflowEvaluationConfig,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Score routed streamflow into ``<run_dir>/evaluation/``."""
    return write_evaluation_products(
        sim_nc,
        evaluation.obs_csv,
        Path(run_dir) / "evaluation",
        variable=evaluation.variable,
        resample=evaluation.resample,
        seg=evaluation.seg,
        reach_id=evaluation.reach_id,
        date_col=evaluation.date_col,
        flow_col=evaluation.flow_col,
        start_time=evaluation.start_time,
        end_time=evaluation.end_time,
    )


def _load_eval_config(config_path: str | Path) -> dict[str, Any]:
    raw = load_config(config_path)
    base_dir = Path(raw.get("_meta", {}).get("config_dir", Path(config_path).parent))
    output_dir = raw.get("output_dir") or raw.get("paths", {}).get("output_dir")
    if not output_dir:
        raise ValueError("Config must define `output_dir`.")
    sim_nc = raw.get("sim_nc")
    obs_csv = raw.get("obs_csv")
    if not sim_nc:
        raise ValueError("Config must define `sim_nc`.")
    if not obs_csv:
        raise ValueError("Config must define `obs_csv`.")
    resample = raw.get("resample", DEFAULT_RESAMPLE)
    if resample in (None, "null"):
        resample = None
    return {
        "case_name": raw.get("case_name", "streamflow_eval"),
        "output_dir": _normalize_output_dir(str(output_dir), base_dir),
        "sim_nc": _require_absolute_input(str(sim_nc), "sim_nc"),
        "obs_csv": _require_absolute_input(str(obs_csv), "obs_csv"),
        "variable": raw.get("variable", "IRFroutedRunoff"),
        "resample": resample,
        "seg": int(raw.get("seg", 0)),
        "reach_id": raw.get("reach_id"),
        "date_col": raw.get("date_col") or raw.get("obs_date_col"),
        "flow_col": raw.get("flow_col") or raw.get("obs_flow_col"),
        "start_time": raw.get("start_time"),
        "end_time": raw.get("end_time"),
        "_raw": raw,
        "_config_path": str(Path(config_path).resolve()),
    }


def run_evaluation(
    sim_nc: str | Path,
    obs_csv: str | Path,
    *,
    output_dir: str | Path,
    variable: str = "IRFroutedRunoff",
    resample: str | None = DEFAULT_RESAMPLE,
    seg: int = 0,
    reach_id: Any | None = None,
    date_col: str | None = None,
    flow_col: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    case_name: str = "streamflow_eval",
    config_path: str | Path | None = None,
    resolved_config: dict[str, Any] | None = None,
) -> Path:
    """Pair files, score deterministic hydrology metrics, and write a run directory."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / _now_utc_tag()
    run_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(level=logging.INFO, log_file=run_dir / "evaluate_streamflow.log")
    log.info("snowda streamflow evaluation starting")
    if config_path:
        log.info("Config: %s", config_path)
    log.info("Run dir: %s", run_dir)
    log.info("Simulation: %s", sim_nc)
    log.info("Observations: %s", obs_csv)

    snapshot = resolved_config or {
        "case_name": case_name,
        "sim_nc": str(sim_nc),
        "obs_csv": str(obs_csv),
        "variable": variable,
        "resample": resample,
        "seg": seg,
        "reach_id": reach_id,
        "date_col": date_col,
        "flow_col": flow_col,
        "start_time": start_time,
        "end_time": end_time,
        "output_dir": str(output_root),
    }
    config_out = run_dir / "config.resolved.json"
    config_out.write_text(
        json.dumps(_json_safe(snapshot), indent=2),
        encoding="utf-8",
    )
    log.info("Wrote config snapshot: %s", config_out)

    metrics_dict = write_evaluation_products(
        sim_nc,
        obs_csv,
        run_dir,
        variable=variable,
        resample=resample,
        seg=seg,
        reach_id=reach_id,
        date_col=date_col,
        flow_col=flow_col,
        start_time=start_time,
        end_time=end_time,
    )

    repo_root = Path(__file__).resolve().parents[3]
    provenance = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit_hash(repo_root),
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "run_dir": str(run_dir),
        "sim_nc": str(sim_nc),
        "obs_csv": str(obs_csv),
        "n_pairs": metrics_dict.get("n_pairs"),
        "scores_version": metrics_dict.get("scores_version"),
        "kge_method": metrics_dict.get("kge_method"),
    }
    (run_dir / "provenance.json").write_text(
        json.dumps(_json_safe(provenance), indent=2),
        encoding="utf-8",
    )
    log.info("Wrote provenance: %s", run_dir / "provenance.json")
    log.info("snowda streamflow evaluation complete")
    return run_dir


def evaluate_streamflow(config_path: str | Path) -> Path:
    """Run evaluation from a YAML config."""
    cfg = _load_eval_config(config_path)
    snapshot = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    snapshot["config_path"] = cfg["_config_path"]
    return run_evaluation(
        cfg["sim_nc"],
        cfg["obs_csv"],
        output_dir=cfg["output_dir"],
        variable=cfg["variable"],
        resample=cfg["resample"],
        seg=cfg["seg"],
        reach_id=cfg["reach_id"],
        date_col=cfg["date_col"],
        flow_col=cfg["flow_col"],
        start_time=cfg["start_time"],
        end_time=cfg["end_time"],
        case_name=cfg["case_name"],
        config_path=cfg["_config_path"],
        resolved_config=snapshot,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate deterministic streamflow against observations."
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    evaluate_streamflow(args.config)


if __name__ == "__main__":
    main()
