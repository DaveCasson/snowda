"""Calibrate snow5 parameters against station SWE (split sample).

The objective is RMSE between simulated SWE and observed SWE on survey dates
in the calibration period; the validation period is scored but not fitted.
SWE does not depend on fSCA, so the objective uses a SWE-only loop that
reproduces ``run_snow5_simulation`` SWE exactly (checked at start-up).

Identifiability: ``compute_melt_factor`` uses (f_max+f_min)/2 * (f_max-f_min)/2
* sin, so only f_max^2 - f_min^2 matters. Keep one of them in ``fixed``.

Run with:
    python -m snowda.workflows.calibrate_snow5 <calibration.yaml>
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.optimize import differential_evolution

from snowda.config.loader import load_config
from snowda.core.snow5 import (
    Snow5Parameters,
    compute_melt_factor,
    compute_potential_melt,
    generate_snowfall,
    phase_shift_day_of_year,
    run_snow5_simulation,
)
from snowda.io.snow5 import read_snow5_forcing
from snowda.utils.logging import configure_logging, get_logger

log = get_logger(__name__)

PARAM_NAMES = ("guc", "src", "f_max", "f_min", "t_mlt")


def simulate_swe(precip: np.ndarray, airtemp: np.ndarray, doy: np.ndarray, p: Snow5Parameters) -> np.ndarray:
    """End-of-day SWE for one HRU, same arithmetic as ``run_snow5_timestep``."""
    snowfall = generate_snowfall(precip, airtemp, p.guc, p.src)
    melt_factor = compute_melt_factor(phase_shift_day_of_year(doy), p.f_max, p.f_min)
    potential = compute_potential_melt(airtemp, melt_factor, p.t_mlt)
    out = np.empty(len(precip))
    swe = 0.0
    for t, (sf, pm) in enumerate(zip(snowfall.tolist(), potential.tolist())):
        swe += sf
        swe -= min(swe, pm)
        out[t] = swe
    return out


def metrics(sim: np.ndarray, obs: np.ndarray) -> dict:
    err = sim - obs
    nse = 1.0 - np.sum(err**2) / np.sum((obs - obs.mean()) ** 2)
    return {
        "n": int(len(obs)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "r": float(np.corrcoef(sim, obs)[0, 1]),
        "nse": float(nse),
    }


def _wy(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(index.year + (index.month >= 10))


def run_calibration(config_path: str | Path) -> Path:
    cfg: Dict[str, Any] = load_config(config_path)
    out_dir = Path(cfg["output_dir"]) / datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(level=logging.INFO, log_file=out_dir / "calibration.log")

    forcing_hru = read_snow5_forcing(cfg["forcing_file"]).sel(
        time=slice(cfg["start_time"], cfg["end_time"])
    ).load()
    forcing = forcing_hru.isel(hru=0)
    time = pd.DatetimeIndex(forcing["time"].values)
    precip = forcing["pptrate"].values.astype("float64")
    airtemp = forcing["airtemp"].values.astype("float64")
    doy = forcing["doy"].values.astype("float64")

    with xr.open_dataset(cfg["obs_nc"]) as ds:
        obs = ds[cfg.get("obs_var", "swe_obs")].sel(station_id=str(cfg["station_id"])).to_series()
    obs = obs.dropna()
    obs = obs[(obs.index >= time[0]) & (obs.index <= time[-1])]
    idx = time.get_indexer(obs.index)
    wy = _wy(obs.index)
    cal_years = cfg["calibration_water_years"]
    cal = (wy >= cal_years[0]) & (wy <= cal_years[1])
    val = ~cal
    log.info("Obs: %d calibration, %d validation dates", cal.sum(), val.sum())

    fixed = {k: float(v) for k, v in (cfg.get("fixed") or {}).items()}
    bounds_cfg = cfg["bounds"]
    free = [k for k in PARAM_NAMES if k in bounds_cfg]
    if set(free) | set(fixed) != set(PARAM_NAMES):
        raise ValueError(f"Each of {PARAM_NAMES} must be in `bounds` or `fixed`.")

    def params_from(vec) -> Snow5Parameters:
        values = dict(fixed)
        values.update({k: float(v) for k, v in zip(free, vec)})
        return Snow5Parameters.from_dict(values)

    # The fast loop must match the full model.
    initial = Snow5Parameters.from_dict(cfg["initial_parameters"])
    full, _, _ = run_snow5_simulation(forcing_hru, initial, np.array([0.5]))
    fast = simulate_swe(precip, airtemp, doy, initial)
    # Round-off accumulates to ~1e-3 kg m-2 over decades; 0.01 is far below obs error.
    if not np.allclose(full["swe"][:, 0], fast, atol=1e-2, rtol=0.0):
        raise RuntimeError("SWE-only loop does not match run_snow5_simulation.")

    y = obs.values

    def objective(vec) -> float:
        sim = simulate_swe(precip, airtemp, doy, params_from(vec))[idx]
        return float(np.sqrt(np.mean((sim[cal] - y[cal]) ** 2)))

    opt = cfg.get("optimizer", {})
    result = differential_evolution(
        objective,
        bounds=[tuple(bounds_cfg[k]) for k in free],
        seed=int(opt.get("seed", 1)),
        popsize=int(opt.get("popsize", 20)),
        maxiter=int(opt.get("maxiter", 200)),
        tol=float(opt.get("tol", 1e-6)),
        polish=True,
    )
    best = params_from(result.x)
    log.info("Best %s: cal RMSE %.2f (%d evals)", best, result.fun, result.nfev)

    rows = []
    sims = {}
    for label, p in (("initial", initial), ("calibrated", best)):
        sim_all = simulate_swe(precip, airtemp, doy, p)
        sims[label] = sim_all
        sim = sim_all[idx]
        for period, mask in (("calibration", cal), ("validation", val), ("all", np.ones_like(cal))):
            rows.append({"parameters": label, "period": period, **metrics(sim[mask], y[mask])})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "metrics.csv", index=False, float_format="%.3f")
    log.info("\n%s", table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    best_dict = {k: round(float(getattr(best, k)), 4) for k in PARAM_NAMES}
    (out_dir / "calibrated_parameters.yaml").write_text(
        yaml.safe_dump({"model_parameters": best_dict}, sort_keys=False)
    )
    pd.DataFrame({k: v for k, v in sims.items()}, index=time).to_csv(out_dir / "swe_initial_vs_calibrated.csv")
    obs.to_frame("swe_obs").assign(calibration=cal).to_csv(out_dir / "obs_used.csv")
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "config_path": cfg["_meta"]["config_path"],
                "forcing_file": cfg["forcing_file"],
                "obs_nc": cfg["obs_nc"],
                "obs_var": cfg.get("obs_var", "swe_obs"),
                "station_id": cfg["station_id"],
                "calibration_water_years": cal_years,
                "free": free,
                "fixed": fixed,
                "bounds": bounds_cfg,
                "best": best_dict,
                "objective_rmse": float(result.fun),
                "nfev": int(result.nfev),
                "message": str(result.message),
            },
            indent=2,
        )
    )
    log.info("Calibration outputs: %s", out_dir)
    return out_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate snow5 against station SWE.")
    parser.add_argument("config", type=str)
    run_calibration(parser.parse_args().config)


if __name__ == "__main__":
    main()
