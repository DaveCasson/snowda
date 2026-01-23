from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import xarray as xr

from snowda.config.loader import load_config
from snowda.core.snow5 import Snow5Parameters, run_snow5_simulation
from snowda.io.snow5 import convert_arrays_to_dataset, read_snow5_forcing, write_results
from snowda.utils.logging import configure_logging, get_logger
from snowda.workflows.run_snow5 import (
    _apply_time_window,
    _ensure_dir,
    _load_cv_array,
    _load_helbig_params,
    _now_utc_tag,
    _parse_time,
)

log = get_logger(__name__)


def _resolve_forcing_paths(cfg: Dict[str, Any], base_dir: Path) -> List[Path]:
    files = cfg.get("ensemble_forcing_files")
    if not files:
        raise ValueError("Config must define `ensemble_forcing_files`.")
    resolved = []
    for raw in files:
        p = Path(raw)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Forcing file not found: {p}")
        resolved.append(p)
    return resolved


def _run_assimilation(
    result_ds: xr.Dataset,
    obs_nc: Path | str,
    obs_var: str,
    out_dir: Path,
    forcing_label: str,
) -> Path:
    with xr.open_dataset(obs_nc) as obs_ds:
        if obs_var not in obs_ds:
            raise ValueError(f"Observation variable '{obs_var}' not found in {obs_nc}.")
        obs = obs_ds[obs_var]

    model_var = "swe" if obs_var == "swe" else obs_var
    if model_var not in result_ds:
        raise ValueError(f"Modeled variable '{model_var}' is not available in the output dataset.")

    modeled, observed = xr.align(result_ds[model_var], obs, join="inner")
    error = modeled - observed
    rmse = float(np.sqrt(np.mean(error**2)))

    out = xr.Dataset(
        data_vars={
            f"{obs_var}_modeled": modeled,
            f"{obs_var}_obs": observed,
            f"{obs_var}_error": error,
        }
    )
    out.attrs["rmse"] = rmse
    out_path = out_dir / f"{forcing_label}_{obs_var}_assimilation.nc"
    out.to_netcdf(out_path)
    log.info(
        "Wrote assimilation diagnostics for %s (%s): rmse=%s -> %s",
        forcing_label,
        obs_var,
        rmse,
        out_path,
    )
    return out_path


def run_snow5_ensemble(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    base_dir = Path(cfg.get("_meta", {}).get("config_dir", Path(config_path).resolve().parent))

    output_dir = cfg.get("output_dir")
    if not output_dir:
        raise ValueError("Config must define `output_dir`.")
    output_dir = _ensure_dir(Path(output_dir) / _now_utc_tag())

    log_file_cfg = cfg.get("log_file")
    log_file = None
    if isinstance(log_file_cfg, str) and log_file_cfg:
        log_file = output_dir / Path(log_file_cfg).name

    configure_logging(level=logging.INFO, log_file=log_file)
    log.info("snowda snow5 ensemble run starting")
    log.info("Config: %s", cfg.get("_meta", {}).get("config_path", str(config_path)))
    log.info("Run dir: %s", output_dir)

    forcing_files = _resolve_forcing_paths(cfg, base_dir)
    start_time = _parse_time(cfg.get("start_time"), "start_time")
    end_time = _parse_time(cfg.get("end_time"), "end_time")

    params_cfg = cfg.get("model_parameters")
    if not isinstance(params_cfg, dict):
        raise ValueError("`model_parameters` must be provided as a mapping.")
    params = Snow5Parameters.from_dict(params_cfg)

    fsca_model = cfg.get("fsca_model", "ssnowd")
    helbig_settings = cfg.get("helbig_settings")
    helbig_params = None
    if fsca_model == "helbig":
        if not helbig_settings:
            raise ValueError("Helbig settings are required when fsca_model is 'helbig'.")
        params_nc = cfg.get("helbig_params_nc")
        if not params_nc:
            raise ValueError("`helbig_params_nc` is required for Helbig ensemble runs.")

    assimilation_nc = cfg.get("assimilation_nc")
    assimilation_var = cfg.get("assimilation_var", "swe")
    snow_density = float(cfg.get("snow_density", 300.0))

    ensemble_info = []
    for idx, forcing_path in enumerate(forcing_files, start=1):
        label = forcing_path.stem or f"forcing_{idx}"
        run_dir = _ensure_dir(output_dir / label)
        forcing_ds = read_snow5_forcing(forcing_path)
        if start_time or end_time:
            forcing_ds = _apply_time_window(forcing_ds, start_time, end_time)
        cv_array = _load_cv_array(cfg, forcing_ds)

        helbig_params_data = _load_helbig_params(params_nc, forcing_ds) if fsca_model == "helbig" else None

        output_arrays, _, _ = run_snow5_simulation(
            forcing_ds,
            params,
            cv_array,
            fsca_model=str(fsca_model),
            helbig_settings=helbig_settings,
            helbig_params=helbig_params_data,
            snow_density=snow_density,
            initial_model_arrays=None,
            initial_states=None,
        )

        result_ds = convert_arrays_to_dataset(output_arrays, forcing_ds)
        result_out = run_dir / f"{label}_snow5_output.nc"
        write_results(result_ds, result_out)

        assimilation_result = None
        if assimilation_nc:
            assimilation_result = _run_assimilation(
                result_ds,
                Path(assimilation_nc),
                assimilation_var,
                run_dir,
                label,
            )

        ensemble_info.append(
            {
                "forcing": str(forcing_path),
                "run_dir": str(run_dir),
                "output": str(result_out),
                "assimilation": str(assimilation_result) if assimilation_result else None,
            }
        )

    prov = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": cfg.get("_meta", {}).get("config_path", str(config_path)),
        "run_dir": str(output_dir),
        "ensemble_forcing_files": [str(p) for p in forcing_files],
        "fsca_model": fsca_model,
        "snow_density": snow_density,
        "helbig_params": cfg.get("helbig_params_nc"),
        "assimilation_nc": assimilation_nc,
        "assimilation_var": assimilation_var,
        "ensemble_runs": ensemble_info,
    }
    prov_out = output_dir / "provenance.json"
    prov_out.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    log.info("Wrote provenance: %s", prov_out)

    log.info("snowda snow5 ensemble run complete")
    return output_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run snow5 ensemble + assimilation workflow.")
    parser.add_argument("config", type=str, help="Path to ensemble YAML config file")
    args = parser.parse_args()
    run_snow5_ensemble(args.config)


if __name__ == "__main__":
    main()
