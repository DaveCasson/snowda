"""Estimate the SWE observation-error budget for the particle filter.

Reads the quantile-mapped observation file written by ``snowda.obs_qm``
(``swe_obs``, ``swe_model``, ``swe_model_equivalent``, ``mapping_flag`` on
``time`` x ``station_id``) and splits the error into parts that can be
measured from the data:

1. Innovation, per station: ``d = swe_model_equivalent - swe_model``. The
   mapping is cross-validated (leave one water year out), so ``d`` is an
   out-of-sample error. Its variance is ``sigma_o^2 + sigma_m^2``: station
   observation error plus open-loop model error.
2. Station error ``sigma_o``: on each survey campaign, every station is
   compared with the same lumped model state, so the model error is common
   to all of them and cancels in the spread of ``d`` across stations. The
   pooled within-campaign variance of ``d`` estimates ``sigma_o^2``
   (instrument + representativeness + mapping error). Errors shared by every
   station cannot be seen this way, so this is a lower bound.
3. Local error ``sigma_local`` from a collocated pair (Twin Creeks A and B):
   ``var(d_A - d_B) / 2``. What is left of ``sigma_o^2`` after removing it is
   representativeness.
4. Model error ``sigma_m^2 = var(campaign-mean d) - mean(sigma_o^2 / n)``.
   Compare its square root with the open-loop ensemble spread.
5. Timing ``sigma_t`` (optional): the filter compares a sample with model SWE
   on the assimilation date, up to ``obs_window_days`` away. With a daily
   model series, ``sigma_t`` is the RMS of
   ``model(sample time) - model(assimilation date)``.

``sigma_o`` is fitted as ``sigma^2 = (a * SWE)^2 + b^2`` over campaigns, so
``a`` and ``b`` translate to ``swe_relative_uncertainty`` and
``swe_sigma_floor``. The filter uses ``max(a * |z|, b)``, which is within a
factor of sqrt(2) of the quadrature form and equal to it away from the
crossover.

With one station, ``sigma_o`` and ``sigma_m`` cannot be separated; the
script reports sd(d) and splits it only when ``--sigma-o`` is given.
``--model-nc`` must be the run whose SWE is ``swe_model`` in the file.
Without ``--dates``, timing uses the 1st of ``--months`` in every model year.

Run from the repository root:

    python scripts/swe_error_budget.py data/obs_qm/ross_yukon_swe_model_equivalent.nc \
        --exclude 09BA-SC02 --pair 09BA-SC02A 09BA-SC02B

    python scripts/swe_error_budget.py outputs/obs_qm/ross_yukon_swe_model_equivalent.nc \
        --sigma-o 35 --window-days 7 --model-nc \
        outputs/ross_summa_deterministic/complete_ensemble_mean_result/summa/001_2024_12_29_timestep.nc
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

USABLE_FLAGS = (0, 1)  # mapped, observed zero


@dataclass(frozen=True)
class StationErrorFit:
    relative: float  # a, fraction of SWE
    floor: float  # b, kg m-2
    pooled_sigma: float  # sqrt of the dof-weighted mean within-campaign variance
    n_campaigns: int
    dof: int


def innovations(frame: pd.DataFrame) -> pd.DataFrame:
    """Usable samples with ``d = swe_model_equivalent - swe_model``.

    ``frame`` has columns ``time``, ``station_id``, ``swe_model_equivalent``,
    ``swe_model``, ``mapping_flag``.
    """
    usable = frame["mapping_flag"].isin(USABLE_FLAGS)
    finite = np.isfinite(frame["swe_model_equivalent"]) & np.isfinite(
        frame["swe_model"]
    )
    out = frame.loc[
        usable & finite, ["time", "station_id", "swe_model_equivalent", "swe_model"]
    ]
    out = out.rename(columns={"swe_model_equivalent": "z", "swe_model": "model"}).copy()
    out["time"] = pd.to_datetime(out["time"])
    out["d"] = out["z"] - out["model"]
    return out.sort_values("time").reset_index(drop=True)


def assign_campaigns(samples: pd.DataFrame, max_gap_days: int = 5) -> pd.DataFrame:
    """Group samples into survey campaigns.

    A new campaign starts when the gap to the previous sample time exceeds
    ``max_gap_days``. A station sampled twice in one campaign keeps its mean.
    """
    times = samples["time"].sort_values()
    new = times.diff() > pd.Timedelta(days=max_gap_days)
    campaign = new.cumsum().reindex(samples.index)
    out = samples.assign(campaign=campaign.to_numpy())
    return out.groupby(["campaign", "station_id"], as_index=False).agg(
        time=("time", "mean"), z=("z", "mean"), model=("model", "mean"), d=("d", "mean")
    )


def campaign_spread(per_station: pd.DataFrame) -> pd.DataFrame:
    """Per campaign: number of stations, mean SWE, mean and variance of ``d``."""
    grouped = per_station.groupby("campaign")
    table = pd.DataFrame(
        {
            "time": grouped["time"].min(),
            "n": grouped["d"].size(),
            "mean_z": grouped["z"].mean(),
            "mean_d": grouped["d"].mean(),
            "var_d": grouped["d"].var(ddof=1),
        }
    )
    return table


def fit_relative_and_floor(
    mean_swe: np.ndarray, variance: np.ndarray, dof: np.ndarray
) -> tuple[float, float]:
    """Weighted non-negative fit of ``variance = a^2 * mean_swe^2 + b^2``.

    A sample variance with ``k`` degrees of freedom has variance proportional
    to ``1 / k``, so each campaign is weighted by its degrees of freedom.
    """
    design = np.column_stack([np.square(mean_swe), np.ones_like(mean_swe)])
    w = np.sqrt(dof)
    coef, *_ = np.linalg.lstsq(design * w[:, None], variance * w, rcond=None)
    if coef[0] < 0:  # no growth with SWE: floor only
        coef = np.array([0.0, np.average(variance, weights=dof)])
    elif coef[1] < 0:  # no floor: relative only
        a2 = np.sum(dof * design[:, 0] * variance) / np.sum(dof * design[:, 0] ** 2)
        coef = np.array([a2, 0.0])
    return float(np.sqrt(coef[0])), float(np.sqrt(coef[1]))


def station_error(per_station: pd.DataFrame) -> tuple[StationErrorFit, pd.DataFrame]:
    """``sigma_o`` from the spread of ``d`` across stations within campaigns."""
    table = campaign_spread(per_station)
    multi = table[table["n"] >= 2]
    if multi.empty:
        raise ValueError(
            "No campaign has two or more stations; sigma_o cannot be estimated."
        )
    dof = (multi["n"] - 1).to_numpy(dtype=float)
    var = multi["var_d"].to_numpy(dtype=float)
    relative, floor = fit_relative_and_floor(
        np.abs(multi["mean_z"].to_numpy(dtype=float)), var, dof
    )
    fit = StationErrorFit(
        relative=relative,
        floor=floor,
        pooled_sigma=float(np.sqrt(np.average(var, weights=dof))),
        n_campaigns=len(multi),
        dof=int(dof.sum()),
    )
    return fit, table


def model_error(table: pd.DataFrame, sigma_o2: np.ndarray) -> float:
    """``sigma_m`` from campaign means of ``d``: ``var(mean d) - mean(sigma_o^2 / n)``.

    Campaign-mean ``d`` is centred first, so any remaining bias is excluded.
    """
    centred = table["mean_d"] - table["mean_d"].mean()
    excess = float(np.mean(np.square(centred)) - np.mean(sigma_o2 / table["n"]))
    return float(np.sqrt(max(excess, 0.0)))


def sigma_by_swe_band(table: pd.DataFrame, n_bands: int = 4) -> pd.DataFrame:
    """Pooled within-campaign ``sigma_o`` per SWE quantile band."""
    multi = table[table["n"] >= 2].copy()
    multi["band"] = pd.qcut(multi["mean_z"], n_bands, duplicates="drop")
    return multi.groupby("band", observed=True).apply(
        lambda g: pd.Series(
            {
                "campaigns": len(g),
                "mean_swe": g["mean_z"].mean(),
                "sigma_o": np.sqrt(np.average(g["var_d"], weights=g["n"] - 1)),
            }
        )
    )


def default_dates(model: pd.Series, months: Sequence[int]) -> list[pd.Timestamp]:
    """First of each month in ``months`` for every year the model covers."""
    index = pd.to_datetime(model.index)
    return [
        pd.Timestamp(year, month, 1)
        for year in range(index.min().year, index.max().year + 1)
        for month in months
        if index.min() <= pd.Timestamp(year, month, 1) <= index.max()
    ]


def pair_error(per_station: pd.DataFrame, first: str, second: str) -> tuple[float, int]:
    """``sigma_local = sqrt(var(d_first - d_second) / 2)`` over shared campaigns."""
    wide = per_station.pivot_table(index="campaign", columns="station_id", values="d")
    missing = [station for station in (first, second) if station not in wide]
    if missing:
        raise ValueError(f"no usable samples (flag 0 or 1) for {', '.join(missing)}")
    diff = (wide[first] - wide[second]).dropna()
    if len(diff) < 2:
        raise ValueError("Fewer than two shared campaigns for the pair.")
    return float(np.sqrt(diff.var(ddof=1) / 2.0)), len(diff)


def timing_error(
    model: pd.Series,
    sample_times: Sequence[pd.Timestamp],
    dates: Sequence[pd.Timestamp],
    window_days: int,
) -> tuple[float, int]:
    """RMS of ``model(sample time) - model(nearest assimilation date)``.

    Samples are assigned as in ``snowda.core.swe_assimilation``: the nearest
    date within ``window_days``, ties to the earlier date. ``model`` is a
    daily (midnight) series indexed by time; sample times are floored to
    the day.
    """
    ordered = sorted(pd.Timestamp(d) for d in dates)
    window = pd.Timedelta(days=window_days)
    series = model.copy()
    series.index = pd.to_datetime(series.index).normalize()
    deltas = []
    for raw in sample_times:
        when = pd.Timestamp(raw).normalize()
        candidates = [d for d in ordered if abs(when - d) <= window]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda d: (abs(when - d), d))
        if when in series.index and nearest in series.index:
            deltas.append(float(series.loc[when] - series.loc[nearest]))
    if not deltas:
        return float("nan"), 0
    return float(np.sqrt(np.mean(np.square(deltas)))), len(deltas)


def _load_frame(path: str) -> pd.DataFrame:
    import xarray as xr

    names = ["swe_model_equivalent", "swe_model", "mapping_flag"]
    with xr.open_dataset(path) as ds:
        frame = ds[names].to_dataframe().reset_index()
    frame["station_id"] = frame["station_id"].astype(str)
    return frame


def _load_model(path: str, variable: str) -> pd.Series:
    from snowda.obs_qm.io import read_midnight_model_swe

    return read_midnight_model_swe(path, variable=variable).to_series()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mapped_nc", help="obs_qm output with swe_model_equivalent")
    parser.add_argument(
        "--exclude", nargs="*", default=[], help="station ids to leave out"
    )
    parser.add_argument(
        "--pair", nargs=2, metavar=("A", "B"), help="collocated station pair"
    )
    parser.add_argument("--campaign-gap-days", type=int, default=5)
    parser.add_argument("--model-nc", help="daily model SWE for the timing term")
    parser.add_argument("--model-var", default="scalarSWE")
    parser.add_argument(
        "--dates",
        nargs="*",
        default=[],
        help="assimilation dates (default: 1st of --months in every model year)",
    )
    parser.add_argument("--months", nargs="*", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--window-days", type=int, default=15)
    parser.add_argument(
        "--sigma-o",
        type=float,
        help="station error (kg m-2) to use when the file has one station, "
        "e.g. the pooled value from a multi-station file",
    )
    args = parser.parse_args(argv)

    frame = _load_frame(args.mapped_nc)
    samples = innovations(frame)
    per_station_all = assign_campaigns(samples, args.campaign_gap_days)
    per_station = per_station_all[~per_station_all["station_id"].isin(args.exclude)]

    print("1. Innovation d = model-equivalent - model, by station (kg m-2)")
    by_station = per_station.groupby("station_id")["d"]
    print(
        pd.DataFrame(
            {
                "n": by_station.size(),
                "bias": by_station.mean(),
                "rms": np.sqrt(by_station.apply(lambda x: np.mean(x**2))),
            }
        )
        .round(1)
        .to_string()
    )

    if per_station.empty:
        raise SystemExit(
            "No usable samples left after --exclude. A single-station file needs no --exclude."
        )
    try:
        fit, table = station_error(per_station)
    except ValueError:
        fit, table = None, campaign_spread(per_station)

    if fit is None:
        d = per_station["d"]
        print(
            "\n2. Station error sigma_o: not identifiable, no campaign has two stations."
            f"\n   var(d) = sigma_o^2 + sigma_m^2; sd(d) = {d.std(ddof=1):6.1f} kg m-2"
            f" over {len(d)} samples"
        )
        if args.sigma_o is not None:
            sigma_m = np.sqrt(max(d.var(ddof=1) - args.sigma_o**2, 0.0))
            print(f"   with --sigma-o {args.sigma_o:.1f}: sigma_m {sigma_m:6.1f} kg m-2")
        else:
            print("   Pass --sigma-o (e.g. from a multi-station file) to split it.")
    else:
        print(
            f"\n2. Station error sigma_o from {fit.n_campaigns} campaigns ({fit.dof} dof)"
        )
        print(f"   pooled sigma_o        {fit.pooled_sigma:6.1f} kg m-2")
        print(f"   fit relative a        {fit.relative:6.3f}  (swe_relative_uncertainty)")
        print(f"   fit floor b           {fit.floor:6.1f} kg m-2 (swe_sigma_floor)")
        print("   sigma_o by SWE band:")
        print(sigma_by_swe_band(table).round(1).to_string())
        if fit.relative == 0.0 or fit.floor == 0.0:
            print(
                "   The (a*SWE)^2 + b^2 form does not fit; one term was clipped to 0. "
                "Read the bands, not a and b."
            )

    if args.pair and fit is not None:
        try:
            local, n_pair = pair_error(per_station_all, *args.pair)
            repr_var = fit.pooled_sigma**2 - local**2
            print(
                f"\n3. Local error from {args.pair[0]} / {args.pair[1]} ({n_pair} campaigns)"
            )
            print(f"   sigma_local           {local:6.1f} kg m-2")
            print(
                f"   representativeness    {np.sqrt(max(repr_var, 0.0)):6.1f} kg m-2 (pooled, by difference)"
            )
        except ValueError as error:
            print(f"\n3. Pair skipped: {error}")

    if fit is not None:
        sigma_o2 = np.square(fit.relative * table["mean_z"].abs()) + fit.floor**2
        print(
            f"\n4. Open-loop model error sigma_m {model_error(table, sigma_o2.to_numpy()):6.1f} kg m-2"
        )
        print("   Compare with the open-loop ensemble spread on the same dates.")

    if args.model_nc:
        model = _load_model(args.model_nc, args.model_var)
        dates = (
            list(pd.to_datetime(args.dates))
            if args.dates
            else default_dates(model, args.months)
        )
        kept = samples[~samples["station_id"].isin(args.exclude)]
        timing, n_t = timing_error(model, kept["time"], dates, args.window_days)
        print(
            f"\n5. Timing sigma_t over {n_t} samples within +/-{args.window_days} days"
            f" of {len(dates)} dates: {timing:6.1f} kg m-2"
        )

    print(
        "\nFor one station on an HRU: sigma^2 = sigma_o^2 + sigma_t^2."
        "\nFor n independent stations averaged: sigma^2 = sigma_o^2 / n + sigma_t^2."
    )


if __name__ == "__main__":
    main()
