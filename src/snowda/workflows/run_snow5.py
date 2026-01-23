from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import xarray as xr

from snowda.config.loader import load_config
from snowda.core.snow5 import Snow5Parameters, run_snow5_simulation
from snowda.io.snow5 import (
    convert_arrays_to_dataset,
    read_snow5_forcing,
    write_results,
)
from snowda.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


def _now_utc_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_output_dir(raw: str, base_dir: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _parse_time(value: str | None, field_name: str) -> np.datetime64 | None:
    if value is None:
        return None
    try:
        return np.datetime64(pd.to_datetime(value))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid date/time string (got: {value})") from exc


def _apply_time_window(
    ds: xr.Dataset,
    start: np.datetime64 | None,
    end: np.datetime64 | None,
) -> xr.Dataset:
    if start is None and end is None:
        return ds
    if start is not None and end is not None and start > end:
        raise ValueError("`start_time` must be <= `end_time`.")
    subset = ds.sel(time=slice(start, end))
    if subset.sizes.get("time", 0) == 0:
        raise ValueError(
            "No timesteps left after applying start_time/end_time constraints."
        )
    return subset


def _load_cv_array(cfg: Dict[str, Any], forcing_ds: xr.Dataset) -> np.ndarray:
    hru_vals = forcing_ds["hru"].values
    cv_path = cfg.get("paths", {}).get("cv_nc")
    if cv_path:
        cv_var = cfg.get("cv_var", "cv")
        with xr.open_dataset(cv_path) as cv_ds:
            if cv_var not in cv_ds:
                raise ValueError(f"CV variable '{cv_var}' not found in {cv_path}.")
            cv_da = cv_ds[cv_var]
            if "hru" not in cv_da.dims:
                raise ValueError("CV dataset must have a 'hru' dimension.")
            missing = set(hru_vals) - set(cv_da["hru"].values)
            if missing:
                raise ValueError(f"CV dataset missing HRUs: {sorted(missing)}")
            ordered = cv_da.sel(hru=hru_vals)
            return ordered.values.astype("float64")
    fallback = float(cfg.get("fsca_cv", 0.5))
    log.info("Using uniform CV=%s for %d HRUs", fallback, len(hru_vals))
    return np.full(len(hru_vals), fallback, dtype="float64")


def _load_helbig_params(
    params_path: str | Path, forcing_ds: xr.Dataset
) -> dict[str, np.ndarray]:
    with xr.open_dataset(params_path) as params_ds:
        if "hru" not in params_ds.dims:
            raise ValueError("Helbig params dataset must have an 'hru' dimension.")
        target_hrus = forcing_ds["hruId"].values
        missing = set(target_hrus) - set(params_ds["hru"].values)
        if missing:
            raise ValueError(f"Helbig params missing HRUs: {sorted(missing)}")
        params_sel = params_ds.sel(hru=target_hrus)
        if "mu" not in params_sel or "xi" not in params_sel:
            raise ValueError("Helbig params dataset requires 'mu' and 'xi' variables.")
        return {
            "mu": params_sel["mu"].values.astype("float64"),
            "xi": params_sel["xi"].values.astype("float64"),
        }


def run_snow5(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    base_dir = Path(cfg.get("_meta", {}).get("config_dir", Path(config_path).resolve().parent))
    output_dir = cfg.get("output_dir")
    if not output_dir:
        raise ValueError("Config must define `output_dir`.")
    output_dir = _normalize_output_dir(str(output_dir), base_dir)
    run_dir = _ensure_dir(output_dir / _now_utc_tag())

    log_file_cfg = cfg.get("log_file")
    log_file = None
    if isinstance(log_file_cfg, str) and log_file_cfg:
        log_file = run_dir / Path(log_file_cfg).name

    configure_logging(level=logging.INFO, log_file=log_file)
    log.info("snowda snow_5 run starting")
    log.info("Config: %s", cfg.get("_meta", {}).get("config_path", str(config_path)))
    log.info("Run dir: %s", run_dir)

    config_out = run_dir / "config.resolved.yaml"
    try:
        import yaml

        with config_out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        log.info("Wrote config snapshot: %s", config_out)
    except Exception as exc:  # pragma: no cover - this should not fail in normal runs
        log.warning("Failed to write config snapshot (%s): %s", config_out, exc)

    forcing_dir = cfg.get("paths", {}).get("forcing_dir")
    if not forcing_dir:
        raise ValueError("Config must define `paths.forcing_dir` pointing to netCDF forcing files.")
    forcing_pattern = cfg.get("forcing_pattern", "*.nc")
    forcing_ds = read_snow5_forcing(Path(forcing_dir), pattern=str(forcing_pattern))

    start_time = _parse_time(cfg.get("start_time"), "start_time")
    end_time = _parse_time(cfg.get("end_time"), "end_time")
    if start_time or end_time:
        log.info("Constraining forcing to start=%s end=%s", start_time, end_time)
    forcing_ds = _apply_time_window(forcing_ds, start_time, end_time)

    params_cfg = cfg.get("model_parameters")
    if not isinstance(params_cfg, dict):
        raise ValueError("`model_parameters` must be provided as a mapping.")
    params = Snow5Parameters.from_dict(params_cfg)

    cv_array = _load_cv_array(cfg, forcing_ds)

    fsca_model = cfg.get("fsca_model", "ssnowd")
    helbig_settings = cfg.get("helbig_settings")
    if fsca_model == "helbig" and not helbig_settings:
        raise ValueError("helbig_settings are required when fsca_model is 'helbig'.")
    snow_density = float(cfg.get("snow_density", 300.0))
    helbig_params = None
    if fsca_model == "helbig":
        params_path = cfg.get("helbig_params_nc")
        if not params_path:
            raise ValueError("`helbig_params_nc` is required when `fsca_model` is 'helbig'.")
        helbig_params = _load_helbig_params(params_path, forcing_ds)

    output_arrays, _, _ = run_snow5_simulation(
        forcing_ds,
        params,
        cv_array,
        fsca_model=str(fsca_model),
        helbig_settings=helbig_settings,
        helbig_params=helbig_params,
        snow_density=snow_density,
        initial_model_arrays=None,
        initial_states=None,
    )

    result_ds = convert_arrays_to_dataset(output_arrays, forcing_ds)
    result_file = cfg.get("output_file", "snow5_output.nc")
    result_path = Path(result_file)
    if not result_path.is_absolute():
        result_path = run_dir / result_path

    write_results(result_ds, result_path)

    prov = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": cfg.get("_meta", {}).get("config_path", str(config_path)),
        "run_dir": str(run_dir),
        "forcing_dir": forcing_dir,
        "forcing_pattern": forcing_pattern,
        "fsca_model": fsca_model,
        "model_parameters": params_cfg,
        "cv_source": cfg.get("paths", {}).get("cv_nc") or "uniform",
        "output_file": str(result_path),
        "snow_density": snow_density,
        "start_time": str(start_time) if start_time is not None else None,
        "end_time": str(end_time) if end_time is not None else None,
        "helbig_params": cfg.get("helbig_params_nc"),
    }
    prov_out = run_dir / "provenance.json"
    prov_out.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    log.info("Wrote provenance: %s", prov_out)

    log.info("snowda snow_5 run complete")
    return run_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the snow_5 test model from a YAML config.")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    run_snow5(args.config)


if __name__ == "__main__":
    main()
