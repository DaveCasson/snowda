from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class QuantileBands:
    low: float
    high: float
    inner_low: float
    inner_high: float


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ESP streamflow ensembles against observations."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=str,
        help="ESP run directory (e.g., outputs/.../<run_id>).",
    )
    parser.add_argument(
        "--obs-csv",
        required=True,
        type=str,
        help="Path to observations CSV.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--obs-date-col",
        default=None,
        type=str,
        help="Observation date column name (auto-detected if omitted).",
    )
    parser.add_argument(
        "--obs-flow-col",
        default=None,
        type=str,
        help="Observation flow column name (auto-detected if omitted).",
    )
    parser.add_argument(
        "--var",
        default="IRFroutedRunoff",
        type=str,
        help="mizuRoute variable name to plot.",
    )
    parser.add_argument(
        "--exclude-climo-year",
        default=False,
        action="store_true",
        help="If set, exclude the observation year from the climatology calculation for each plot.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=["combined", "separate", "both"],
        default="combined",
        help="Specify plot mode: 'combined' for all forecasts overlay, 'separate' for plots per ESP run, or 'both' for both types."
    )
    return parser.parse_args(argv)


def _detect_date_column(df: pd.DataFrame, name: str | None) -> str:
    if name and name in df.columns:
        return name
    candidates = ["date", "time", "datetime", "Date", "Time", "Datetime"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("Could not auto-detect observation date column.")


def _detect_flow_column(df: pd.DataFrame, name: str | None, date_col: str) -> str:
    if name and name in df.columns:
        return name
    numeric_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    candidates = ["streamflow", "flow", "discharge", "q", "Q"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("Could not auto-detect observation flow column.")


def _load_observations(path: Path, date_col: str | None, flow_col: str | None) -> pd.Series:
    df = pd.read_csv(path)
    date_col = _detect_date_column(df, date_col)
    flow_col = _detect_flow_column(df, flow_col, date_col)
    obs = df[[date_col, flow_col]].copy()
    obs[date_col] = pd.to_datetime(obs[date_col])
    obs = obs.set_index(date_col).sort_index()
    return obs[flow_col].astype(float)


def _climatology(series: pd.Series, exclude_years: set[int] | None = None) -> pd.DataFrame:
    # If exclude_years is provided, filter out those years.
    if exclude_years is not None and len(exclude_years) > 0:
        mask = ~series.index.year.isin(exclude_years)
        filtered = series[mask]
    else:
        filtered = series
    df = filtered.to_frame("flow")
    df["dayofyear"] = df.index.dayofyear
    grouped = df.groupby("dayofyear")["flow"]
    return pd.DataFrame(
        {
            "min": grouped.min(),
            "p10": grouped.quantile(0.10),
            "p25": grouped.quantile(0.25),
            "p50": grouped.quantile(0.50),
            "p75": grouped.quantile(0.75),
            "p90": grouped.quantile(0.90),
            "max": grouped.max(),
        }
    )


def _discover_mizuroute_outputs(run_dir: Path) -> list[Path]:
    return sorted(run_dir.rglob("esp/*/*/*/mizuroute/mizuroute_*.nc"))


def _group_trace_files(paths: Iterable[Path]) -> dict[tuple[str, str], list[Path]]:
    grouped: dict[tuple[str, str], list[Path]] = {}
    for path in paths:
        parts = list(path.parts)
        if "esp" not in parts:
            continue
        idx = parts.index("esp")
        try:
            member = parts[idx + 1]
            state_date_key = parts[idx + 2]
        except IndexError:
            continue
        grouped.setdefault((member, state_date_key), []).append(path)
    return grouped


def _read_esp_ensemble(paths: list[Path], variable: str) -> pd.DataFrame:
    """
    Reads an ESP ensemble from a list of mizuRoute NetCDF output files.
    The mizuroute_*.nc files have variables of shape (time, seg) or (time, hru).
    This function assumes variable (e.g., "IRFroutedRunoff") is of shape (time, seg),
    and selects the first segment along axis 1 if multiple.

    Each path corresponds to a different trace year (ensemble member).
    The returned DataFrame has a DateTimeIndex and columns per ensemble member (trace year).

    Parameters:
        paths: list of paths to mizuroute_*.nc files.
        variable: variable name to extract, e.g., "IRFroutedRunoff"

    Returns:
        pd.DataFrame: index = datetime, columns = trace_years
    """
    members = {}
    for path in sorted(paths):
        trace_year = path.stem.split("_")[-1]
        with xr.open_dataset(path) as ds:
            if variable not in ds:
                raise KeyError(f"{variable} not found in {path}")
            data = ds[variable]
            if "seg" in data.dims:
                arr = data.isel(seg=0)
            elif "hru" in data.dims:
                arr = data.isel(hru=0)
            else:
                arr = data  # fallback, if it's just (time,), unlikely for mizuroute outputs
            # Handle time axis
            time = ds["time"]
            times = xr.decode_cf(ds).indexes["time"]
            values = arr.values
        # Make a pandas Series with datetime index
        series = pd.Series(data=values, index=pd.to_datetime(times))
        members[trace_year] = series
    df = pd.DataFrame(members).sort_index()
    return df


def _ensemble_quantiles(df: pd.DataFrame, bands: QuantileBands) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "low": df.quantile(bands.low, axis=1),
            "high": df.quantile(bands.high, axis=1),
            "inner_low": df.quantile(bands.inner_low, axis=1),
            "inner_high": df.quantile(bands.inner_high, axis=1),
            "median": df.quantile(0.5, axis=1),
        }
    )


def _climatology_for_dates(climo: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    lookup = climo.copy()
    lookup.index = lookup.index.astype(int)
    doy = dates.dayofyear
    return lookup.reindex(doy).set_index(dates)


def _latest_run_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"ESP output base directory not found: {base_dir}")
    candidates = [p for p in base_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No ESP run directories found in {base_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _plot_ensemble_all_forecasts(
    output_path: Path,
    obs: pd.Series,
    climo: pd.DataFrame,
    forecasts: dict[tuple[str, str], pd.DataFrame],
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    # Find global date range for all forecasts
    all_dates = pd.Index([])
    for esp_df in forecasts.values():
        all_dates = all_dates.union(esp_df.index)
    if all_dates.empty:
        raise RuntimeError("No ESP forecast dates found for plotting.")
    obs = obs[(obs.index >= all_dates.min()) & (obs.index <= all_dates.max())]

    # Plot obs and climatology for full date span; overlays all ESP envelopes
    climo_q = climo.loc[:, ["min", "p10", "p25", "p50", "p75", "p90", "max"]]
    climo_plot = _climatology_for_dates(climo_q, obs.index)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot climatology bands
    ax.fill_between(
        climo_plot.index,
        climo_plot["min"],
        climo_plot["max"],
        color="lightgray",
        alpha=0.49,
        label="Climatology min-max",
    )
    ax.fill_between(
        climo_plot.index,
        climo_plot["p10"],
        climo_plot["p90"],
        color="lightgray",
        alpha=0.6,
        label="Climatology p10-p90",
    )
    ax.fill_between(
        climo_plot.index,
        climo_plot["p25"],
        climo_plot["p75"],
        color="gray",
        alpha=0.4,
        label="Climatology p25-p75",
    )
    ax.plot(climo_plot.index, climo_plot["p50"], color="black", linestyle="--", label="Climatology median")

    # Plot all ESP forecasts as envelopes, with legend on all
    bands = QuantileBands(low=0.10, high=0.90, inner_low=0.25, inner_high=0.75)
    cmap = plt.get_cmap("tab20" if len(forecasts) > 8 else "tab10")
    colors = [cmap(i % cmap.N) for i in range(len(forecasts))]
    legend_handles = []
    from matplotlib.patches import Patch

    for idx, ((member, state_date_key), esp_df) in enumerate(forecasts.items()):
        if esp_df.empty:
            continue
        esp_q = _ensemble_quantiles(esp_df, bands)
        label = f"ESP member {member}, state date {state_date_key}"
        ax.fill_between(
            esp_q.index,
            esp_q["low"],
            esp_q["high"],
            color=colors[idx],
            alpha=0.34,
            label=None,  # We'll manage legend handles below for clarity
        )
        ax.fill_between(
            esp_q.index,
            esp_q["inner_low"],
            esp_q["inner_high"],
            color=colors[idx],
            alpha=0.50,
            label=None,
        )
        ax.plot(esp_q.index, esp_q["median"], color=colors[idx], alpha=0.9, linewidth=1.5, label=None)
        # Legend handle for the ESP run label:
        legend_handles.append(Patch(facecolor=colors[idx], edgecolor='k', alpha=0.52, label=label))

    # Plot obs
    obs_line, = ax.plot(obs.index, obs.values, color="darkorange", label="Observed", linewidth=2.0)

    # Compose legend
    climo_handles = [
        Patch(facecolor="whitesmoke", edgecolor="none", alpha=0.49, label="Climatology min-max"),
        Patch(facecolor="lightgray", edgecolor="none", alpha=0.6, label="Climatology p10-p90"),
        Patch(facecolor="gray", edgecolor="none", alpha=0.4, label="Climatology p25-p75"),
        ]
    handles = climo_handles + legend_handles + [obs_line]
    labels = [h.get_label() for h in handles]
    ax.set_title(title)
    ax.set_ylabel("Streamflow")
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=9, ncol=2 if len(handles) > 10 else 1)
    ax.grid(True, alpha=0.23)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ensemble_separate_forecasts(
    output_dir: Path,
    obs: pd.Series,
    climo: pd.DataFrame,
    forecasts: dict[tuple[str, str], pd.DataFrame],
    exclude_years: set[int],
    args
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    bands = QuantileBands(low=0.10, high=0.90, inner_low=0.25, inner_high=0.75)
    climo_q = climo.loc[:, ["min", "p10", "p25", "p50", "p75", "p90", "max"]]

    for idx, ((member, state_date_key), esp_df) in enumerate(forecasts.items()):
        if esp_df.empty:
            continue

        forecast_start = esp_df.index.min()
        forecast_end = esp_df.index.max()

        obs_period = obs[(obs.index >= forecast_start) & (obs.index <= forecast_end)]
        climo_plot = _climatology_for_dates(climo_q, obs_period.index)

        fig, ax = plt.subplots(figsize=(13, 6))

        # Plot climatology bands
        ax.fill_between(
            climo_plot.index,
            climo_plot["min"],
            climo_plot["max"],
            color="whitesmoke",
            alpha=0.49,
            label="Climatology min-max",
        )
        ax.fill_between(
            climo_plot.index,
            climo_plot["p10"],
            climo_plot["p90"],
            color="lightgray",
            alpha=0.6,
            label="Climatology p10-p90",
        )
        ax.fill_between(
            climo_plot.index,
            climo_plot["p25"],
            climo_plot["p75"],
            color="gray",
            alpha=0.4,
            label="Climatology p25-p75",
        )
        ax.plot(climo_plot.index, climo_plot["p50"], color="black", linestyle="--", label="Climatology median")

        # Plot ESP forecast envelope for this run
        color = "#3777b5"  # blueish for ESP envelope
        esp_q = _ensemble_quantiles(esp_df, bands)

        ax.fill_between(
            esp_q.index,
            esp_q["low"],
            esp_q["high"],
            color=color,
            alpha=0.36,
            label=f"ESP {member}, state date {state_date_key} (p10-p90)",
        )
        ax.fill_between(
            esp_q.index,
            esp_q["inner_low"],
            esp_q["inner_high"],
            color=color,
            alpha=0.52,
            label=f"ESP {member}, state date {state_date_key} (p25-p75)",
        )
        ax.plot(esp_q.index, esp_q["median"], color=color, alpha=0.97, linewidth=2.1,
                label=f"ESP {member}, state date {state_date_key} median")

        # Plot obs
        ax.plot(obs_period.index, obs_period.values, color="darkorange", linewidth=2.0, label="Observed")

        title = (
            f"ESP State Date: {state_date_key}\n"
            f"Forecast Range: {forecast_start.strftime('%Y-%m-%d')} to {forecast_end.strftime('%Y-%m-%d')}"
        )
        if args.exclude_climo_year and len(exclude_years) > 0:
            title += f"\n(Climatology excludes: {', '.join(map(str, sorted(exclude_years)))})"
        ax.set_title(title)
        ax.set_ylabel("Streamflow")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.18)

        output_path = output_dir / f"esp_streamflow_{member}_{state_date_key}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()

    output_dir = Path(args.output_dir).expanduser().resolve()
    obs_path = Path(args.obs_csv).expanduser().resolve()

    obs = _load_observations(obs_path, args.obs_date_col, args.obs_flow_col)

    trace_files = _discover_mizuroute_outputs(run_dir)
    if not trace_files:
        raise FileNotFoundError(f"No ESP mizuRoute outputs found in {run_dir}")
    grouped = _group_trace_files(trace_files)

    # Read in all ESP forecasts
    all_esp_forecasts: dict[tuple[str, str], pd.DataFrame] = {}
    global_start, global_end = None, None
    exclude_years = set()
    for (member, state_date_key), files in grouped.items():
        esp_df = _read_esp_ensemble(files, args.var)
        if esp_df.empty:
            continue

        start = esp_df.index.min()
        end = esp_df.index.max()
        if global_start is None or start < global_start:
            global_start = start
        if global_end is None or end > global_end:
            global_end = end

        # For climatology exclusion, gather *all* years that would be observed by any forecast's obs period
        obs_period = obs[(obs.index >= start) & (obs.index <= end)]
        if args.exclude_climo_year:
            exclude_years.update(obs_period.index.year.unique())

        all_esp_forecasts[(member, state_date_key)] = esp_df

    # If no valid ESP forecasts, exit
    if not all_esp_forecasts:
        raise RuntimeError("No non-empty ESP forecast dataframes found to plot.")

    # By default, include the actual observation year in climatology unless --exclude-climo-year is set
    climo = _climatology(obs, exclude_years=exclude_years if args.exclude_climo_year else None)

    # Combined (overlay) plot of all forecasts
    if args.plot_mode in ["combined", "both"]:
        title = (
            f"ESP Streamflow - All Forecasts\nDate Range: {global_start.strftime('%Y-%m-%d')} to {global_end.strftime('%Y-%m-%d')}"
        )
        if args.exclude_climo_year and len(exclude_years) > 0:
            title += f"\n(Climatology excludes: {', '.join(map(str, sorted(exclude_years)))})"
        plot_path = output_dir / "esp_streamflow_all_forecasts.png"
        _plot_ensemble_all_forecasts(plot_path, obs, climo, all_esp_forecasts, title)

    # Separate plot per run
    if args.plot_mode in ["separate", "both"]:
        _plot_ensemble_separate_forecasts(
            output_dir=output_dir,
            obs=obs,
            climo=climo,
            forecasts=all_esp_forecasts,
            exclude_years=exclude_years,
            args=args,
        )


if __name__ == "__main__":
    SAMPLE_ARGS = [
        "--run-dir",
        "/Users/dcasson/GitHub/snowda/outputs/ross_summa_esp/20260917T022218Z",
        "--output-dir",
        "/Users/dcasson/GitHub/snowda/outputs/ross_summa_esp/20260917T022218Z/evaluation",
        "--obs-csv",
        "/Users/dcasson/Data/snowda/summa/ross_lumped/observations/streamflow.csv",
        "--var",
        "IRFroutedRunoff",
        #"--exclude-climo-year",  # Uncomment to test exclusion of the obs year from climatology
        "--plot-mode", "both"   # Uncomment to output both combined and separate plots
    ]
    main(SAMPLE_ARGS)
