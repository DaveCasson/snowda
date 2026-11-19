"""Figures and scores for the Ross snow5 synthetic experiment.

Uses the latest run of each mode under outputs/snow5/ross_synthetic/<mode>/.
Writes PNGs and scores.csv to outputs/snow5/ross_synthetic/figures/.

    python scripts/snow5/plot_ross_synthetic.py [--water-year 2013]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from snowda.viz import snow5_synthetic as v

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/snow5/ross_synthetic"
OUT = REPO / "outputs/snow5/ross_synthetic"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--water-year", type=int, default=2013)
    parser.add_argument("--outputs", type=Path, default=OUT)
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()

    with xr.open_dataset(args.data / "synthetic_swe_obs.nc") as obs_ds:
        truth = obs_ds["swe_truth"].isel(hru=0).to_series()
        obs = obs_ds["swe"].isel(hru=0).to_series()

    results = []
    loaders = {
        "historic_mean": v.load_mean,
        "historic_ensemble": v.load_ensemble,
        "sir": lambda d: v.load_da("sir", d),
        "sis": lambda d: v.load_da("sis", d),
    }
    for mode, loader in loaders.items():
        mode_dir = args.outputs / mode
        if not mode_dir.exists():
            print(f"skip {mode}: no runs in {mode_dir}")
            continue
        res = loader(mode_dir)
        print(f"{mode}: {res.run_dir} ({res.members.shape[1]} members)")
        results.append(res)

    fig_dir = args.outputs / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([v.scores(r, truth) for r in results])
    table.to_csv(fig_dir / "scores.csv", index=False, float_format="%.3f")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    v.plot_timeseries(results, truth, obs, fig_dir / "swe_timeseries_by_mode.png")
    v.plot_water_year(results, truth, obs, args.water_year, fig_dir / f"swe_wy{args.water_year}_all_modes.png")
    if any(r.updates is not None for r in results):
        v.plot_neff(results, fig_dir / "n_eff.png")
    v.plot_scores(table, fig_dir / "scores.png")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
