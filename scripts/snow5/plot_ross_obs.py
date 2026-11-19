"""Score and plot the Ross snow5 historic modes against snow-survey SWE.

Uses the latest run of each mode under outputs/snow5/ross_obs/<mode>/.
Writes PNGs and scores.csv to outputs/snow5/ross_obs/figures/.

    python scripts/snow5/plot_ross_obs.py [--start 2008-10-01 --end 2016-09-30]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from snowda.viz import snow5_obs_eval as ev
from snowda.viz import snow5_synthetic as v

REPO = Path(__file__).resolve().parents[2]
OBS_NC = REPO / "data/obs_qm/ross_yukon_swe_model_equivalent.nc"
OUT = REPO / "outputs/snow5/ross_obs"
STATION = "09BA-SC02"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=OUT)
    parser.add_argument("--obs", type=Path, default=OBS_NC)
    parser.add_argument("--station", default=STATION)
    parser.add_argument("--start", default="2008-10-01", help="Time-series window start")
    parser.add_argument("--end", default="2016-09-30", help="Time-series window end")
    parser.add_argument("--primary-obs", default="swe_obs", choices=["swe_obs", "swe_model_equivalent"])
    parser.add_argument("--split-wy", type=int, default=2004, help="First validation water year")
    args = parser.parse_args()

    with xr.open_dataset(args.obs) as ds:
        st = ds.sel(station_id=args.station)
        obs = {
            "swe_model_equivalent": st["swe_model_equivalent"].to_series(),
            "swe_obs": st["swe_obs"].to_series(),
        }
        summa = st["swe_model"].to_series()

    loaders = {
        "historic_mean": v.load_mean,
        "historic_ensemble": v.load_ensemble,
        "sir": lambda d: v.load_da("sir", d),
        "sis": lambda d: v.load_da("sis", d),
    }
    results = []
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

    def wy(index):
        return index.year + (index.month >= 10)

    parts = []
    for period, (a, b) in {"all": (0, 9999), "calibration": (1982, args.split_wy - 1), "validation": (args.split_wy, 9999)}.items():
        sub = {k: s_[(wy(s_.index) >= a) & (wy(s_.index) <= b)] for k, s_ in obs.items()}
        parts.append(ev.score_table(results, sub).assign(period=period))
    table = pd.concat(parts, ignore_index=True)
    table.to_csv(fig_dir / "scores.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 200)
    show = table[table["obs"] == args.primary_obs].drop(columns=["obs"])
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # The model-equivalent obs are mapped to SUMMA's climatology: compare it with snow5's.
    mean_res = next((r for r in results if r.mode == "historic_mean"), None)
    if mean_res is not None:
        s = summa.dropna()
        s = s[(s.index >= mean_res.time[0]) & (s.index <= mean_res.time[-1])]
        snow5 = pd.Series(mean_res.members[:, 0], index=mean_res.time).reindex(s.index)
        print(
            f"\nClimatology at survey dates: SUMMA mean {s.mean():.1f}, "
            f"snow5 historic-mean {snow5.mean():.1f}, station obs {obs['swe_obs'].reindex(s.index).mean():.1f} kg m-2"
        )

    primary = obs[args.primary_obs]
    v.plot_timeseries(
        results, None, primary, fig_dir / "swe_timeseries_by_mode.png",
        title=f"snow5 historic modes, Ross — vs snow surveys at {args.station} ({args.primary_obs})",
        start=args.start, end=args.end,
    )
    ev.plot_scatter(results, obs["swe_model_equivalent"], fig_dir / "scatter_model_equivalent.png", "Model-equivalent SWE")
    ev.plot_scatter(results, obs["swe_obs"], fig_dir / "scatter_station_obs.png", "Station SWE")
    for period in ("all", "validation"):
        ev.plot_skill_bars(table[table["period"] == period], args.primary_obs, fig_dir / f"skill_{args.primary_obs}_{period}.png")
    if any(r.updates is not None for r in results):
        v.plot_neff(results, fig_dir / "n_eff.png")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
