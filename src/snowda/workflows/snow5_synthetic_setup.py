"""Build a snow5 synthetic (twin) experiment from SUMMA ensemble forcing.

One ensemble member is held out as the truth. snow5 driven by that member
gives the true SWE. Synthetic SWE observations are the truth plus Gaussian
noise with the same error model the particle filter assumes:
``sigma = max(relative_uncertainty * |truth|, sigma_floor)``, clipped at zero.
The remaining members form the forecast ensemble, and their hourly mean is
written as the forcing for the deterministic (historic mean) run.

Run with:
    python -m snowda.workflows.snow5_synthetic_setup <setup.yaml>
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import xarray as xr

from snowda.config.loader import load_config
from snowda.config.run_dir import utc_timestamp
from snowda.core.snow5 import Snow5Parameters, run_snow5_simulation
from snowda.io.snow5 import convert_arrays_to_dataset, read_snow5_forcing, write_results
from snowda.utils.logging import configure_logging, get_logger
from snowda.workflows.run_snow5 import _load_cv_array

log = get_logger(__name__)


def _resolve(raw: str, base_dir: Path) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


def _ensemble_paths(cfg: Dict[str, Any]) -> list[Path]:
    root = Path(cfg["ensemble_forcing_dir"])
    paths = sorted(root.glob(cfg.get("ensemble_forcing_pattern", "*/*.nc")))
    if not paths:
        raise FileNotFoundError(f"No ensemble forcing in {root}")
    return paths


def write_mean_forcing(
    member_paths: list[Path], start: str, end: str, out_path: Path
) -> Path:
    """Hourly mean airtemp and pptrate over ``member_paths`` in SUMMA units."""
    total_t = None
    total_p = None
    template = None
    for i, path in enumerate(member_paths, start=1):
        with xr.open_dataset(path) as ds:
            sub = ds[["airtemp", "pptrate", "hruId"]].sel(time=slice(start, end)).load()
        if total_t is None:
            template = sub
            total_t = sub["airtemp"].astype("float64")
            total_p = sub["pptrate"].astype("float64")
        else:
            total_t = total_t + sub["airtemp"]
            total_p = total_p + sub["pptrate"]
        log.info("Mean forcing: added member %d/%d (%s)", i, len(member_paths), path.name)
    n = float(len(member_paths))
    out = xr.Dataset(
        {
            "airtemp": (total_t / n).astype("float32"),
            "pptrate": (total_p / n).astype("float32"),
            "hruId": template["hruId"],
        }
    )
    out["airtemp"].attrs = {"units": "K", "long_name": "ensemble-mean air temperature"}
    out["pptrate"].attrs = {
        "units": "kg m-2 s-1",
        "long_name": "ensemble-mean precipitation rate",
    }
    out.attrs["comment"] = (
        f"Hourly mean of {len(member_paths)} members, truth member excluded"
    )
    out.attrs["members"] = json.dumps([p.name for p in member_paths])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path)
    log.info("Wrote mean forcing: %s", out_path)
    return out_path


def make_synthetic_obs(
    truth: xr.Dataset,
    relative_uncertainty: float,
    sigma_floor: float,
    seed: int,
) -> xr.Dataset:
    """SWE observations = truth + N(0, sigma), clipped at 0, on the hruId axis."""
    swe = truth["swe"].values
    sigma = np.maximum(relative_uncertainty * np.abs(swe), sigma_floor)
    rng = np.random.default_rng(seed)
    obs = np.clip(swe + rng.normal(0.0, 1.0, swe.shape) * sigma, 0.0, None)
    hru_ids = truth["hruId"].values if "hruId" in truth.coords else truth["hru"].values
    ds = xr.Dataset(
        {
            "swe": (("time", "hru"), obs),
            "swe_truth": (("time", "hru"), swe),
            "swe_sigma": (("time", "hru"), sigma),
        },
        coords={"time": truth["time"].values, "hru": hru_ids},
    )
    ds["swe"].attrs = {"units": "kg m-2", "long_name": "synthetic SWE observation"}
    ds["swe_truth"].attrs = {"units": "kg m-2", "long_name": "true SWE"}
    ds["swe_sigma"].attrs = {"units": "kg m-2", "long_name": "observation error std"}
    ds.attrs.update(
        relative_uncertainty=relative_uncertainty,
        sigma_floor=sigma_floor,
        seed=seed,
    )
    return ds


def run_setup(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    base_dir = Path(cfg["_meta"]["config_dir"])
    out_dir = _resolve(cfg["data_dir"], base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(level=logging.INFO, log_file=out_dir / "setup.log")

    start, end = str(cfg["start_time"]), str(cfg["end_time"])
    truth_path = Path(cfg["truth_forcing_file"])
    members = [p for p in _ensemble_paths(cfg) if p.resolve() != truth_path.resolve()]
    log.info("Truth member: %s; %d ensemble members", truth_path, len(members))

    params = Snow5Parameters.from_dict(cfg["model_parameters"])
    forcing = read_snow5_forcing(truth_path).sel(time=slice(start, end))
    cv_array = _load_cv_array(cfg, forcing)
    arrays, _, _ = run_snow5_simulation(
        forcing,
        params,
        cv_array,
        fsca_model=str(cfg.get("fsca_model", "ssnowd")),
        snow_density=float(cfg.get("snow_density", 300.0)),
    )
    truth = convert_arrays_to_dataset(arrays, forcing)
    write_results(truth, out_dir / "truth_snow5.nc")

    obs_cfg = cfg["synthetic_obs"]
    obs = make_synthetic_obs(
        truth,
        float(obs_cfg["swe_relative_uncertainty"]),
        float(obs_cfg["swe_sigma_floor"]),
        int(obs_cfg.get("seed", 0)),
    )
    obs.to_netcdf(out_dir / "synthetic_swe_obs.nc")
    log.info("Wrote synthetic observations: %s", out_dir / "synthetic_swe_obs.nc")

    # Pad a day so the daily resample of the last day is complete.
    end_pad = str(pd.Timestamp(end) + pd.Timedelta(hours=23))
    write_mean_forcing(members, start, end_pad, out_dir / "mean_forcing_excl_truth.nc")

    prov = {
        "timestamp_utc": utc_timestamp(),
        "config_path": cfg["_meta"]["config_path"],
        "truth_forcing_file": str(truth_path),
        "ensemble_members": [str(p) for p in members],
        "start_time": start,
        "end_time": end,
        "model_parameters": cfg["model_parameters"],
        "synthetic_obs": obs_cfg,
        "peak_truth_swe": float(truth["swe"].max()),
    }
    (out_dir / "setup_provenance.json").write_text(json.dumps(prov, indent=2))
    log.info("Synthetic setup complete: %s", out_dir)
    return out_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a snow5 synthetic experiment.")
    parser.add_argument("config", type=str, help="Path to setup YAML")
    run_setup(parser.parse_args().config)


if __name__ == "__main__":
    main()
