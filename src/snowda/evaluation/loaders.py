"""Read what evaluation needs from a finished run directory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from snowda.config.pysummada import StreamflowEvaluationConfig
from snowda.io.da_traces import _MIZUROUTE_FILE_RE, _historic_mizuroute_files
from snowda.io.streamflow import (
    _ensure_unique_sorted_time,
    _maybe_convert_units,
    _normalize_resample,
    _resample_mean,
    read_routed_streamflow,
    read_streamflow_observations,
)
from snowda.utils.logging import get_logger

log = get_logger(__name__)

DETERMINISTIC_MEMBER = "deterministic"
UPDATES_FILENAME = "updates.csv"


@dataclass(frozen=True)
class WeightUpdates:
    """Posterior weights at each assimilation date of a run.

    ``weights`` has one row per date and one column per member. ``table`` is
    ``assimilation/updates.csv`` (one row per date), empty for older runs.
    """

    weights: pd.DataFrame
    resampled: pd.Series
    table: pd.DataFrame


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_observations(evaluation: StreamflowEvaluationConfig) -> xr.DataArray:
    """Observed streamflow on the scoring time step (daily means by default)."""
    obs = read_streamflow_observations(
        evaluation.obs_csv,
        date_col=evaluation.date_col,
        flow_col=evaluation.flow_col,
    )
    obs = _ensure_unique_sorted_time(obs)
    rule = _normalize_resample(evaluation.resample)
    if rule is not None:
        obs = _resample_mean(obs, rule)
    return obs


def load_historic_flow(
    run_dir: str | Path, evaluation: StreamflowEvaluationConfig
) -> xr.DataArray:
    """Routed historic streamflow as ``(time, member)``.

    A deterministic run is one member. An ensemble run has one file per
    member and segment; a member's segments are joined in time. Where two
    segments share a time step, the later segment is kept: it starts from the
    state the filter left behind.
    """
    root = Path(run_dir)
    files = _historic_mizuroute_files(root)
    if not files:
        raise FileNotFoundError(
            f"No routed historic streamflow in {root / 'mizuroute'}. "
            "Evaluation needs mizuRoute output."
        )
    grouped: dict[str, list[Path]] = {}
    for path in files:
        match = _MIZUROUTE_FILE_RE.match(path.name)
        member = match.group("member") if match else DETERMINISTIC_MEMBER
        grouped.setdefault(member, []).append(path)

    rule = _normalize_resample(evaluation.resample)
    columns: dict[str, pd.Series] = {}
    units = None
    for member, paths in sorted(grouped.items()):
        pieces = []
        for path in sorted(paths, key=lambda item: item.name):
            da = read_routed_streamflow(
                path,
                variable=evaluation.variable,
                seg=evaluation.seg,
                reach_id=evaluation.reach_id,
            )
            units = units or da.attrs.get("units")
            pieces.append(da.to_series())
        series = pd.concat(pieces)
        series = series[~series.index.duplicated(keep="last")].sort_index()
        if rule is not None:
            series = series.resample(rule).mean()
        columns[member] = series
    frame = pd.DataFrame(columns).sort_index()
    flow = xr.DataArray(
        frame.to_numpy(dtype=float),
        dims=("time", "member"),
        coords={"time": frame.index.to_numpy(), "member": list(frame.columns)},
        name="sim",
        attrs={"units": units or "m3 s-1", "long_name": "simulated streamflow"},
    )
    return flow


def pair_historic(
    flow: xr.DataArray,
    obs: xr.DataArray,
    evaluation: StreamflowEvaluationConfig,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Simulation and observations on the simulation's time axis.

    Returns the ``(time, member)`` frame, the observed series reindexed to it
    (NaN where there is no observation) and the flow units. ``start_time`` and
    ``end_time`` of the evaluation config limit the period.
    """
    template = flow.isel(member=0, drop=True)
    template, obs = _maybe_convert_units(template, obs)
    frame = flow.to_pandas()
    frame.index = pd.DatetimeIndex(frame.index)
    frame.columns = [str(member) for member in frame.columns]
    start = pd.Timestamp(evaluation.start_time) if evaluation.start_time else None
    end = pd.Timestamp(evaluation.end_time) if evaluation.end_time else None
    frame = frame.loc[start:end]
    observed = obs.to_series().reindex(frame.index)
    return frame, observed, str(template.attrs.get("units", "m3 s-1"))


def load_weight_updates(run_dir: str | Path) -> WeightUpdates | None:
    """Particle weights written by an assimilating run, or ``None`` without any."""
    folder = Path(run_dir) / "assimilation"
    if not folder.is_dir():
        return None
    rows = []
    for path in sorted(folder.glob("*_weights.csv")):
        frame = pd.read_csv(path, index_col=0)
        if frame.empty:
            continue
        frame.index = pd.to_datetime(frame.index)
        frame.columns = [str(column) for column in frame.columns]
        rows.append(frame)
    if not rows:
        return None
    weights = pd.concat(rows).sort_index()
    weights = weights[~weights.index.duplicated(keep="last")]

    table_path = folder / UPDATES_FILENAME
    if table_path.is_file():
        table = pd.read_csv(table_path, parse_dates=["date"]).set_index("date")
    else:
        records = read_json(Path(run_dir) / "provenance.json").get("swe_updates") or []
        table = pd.DataFrame(records)
        if not table.empty:
            table["date"] = pd.to_datetime(table["date"])
            table = table.set_index("date")
    if "resampled" in table.columns:
        resampled = (
            table["resampled"].astype(bool).reindex(weights.index).fillna(False)
        )
    else:
        log.warning(
            "No resampling record in %s; treating every update as not resampled.",
            folder,
        )
        resampled = pd.Series(False, index=weights.index)
    return WeightUpdates(
        weights=weights, resampled=resampled.astype(bool), table=table
    )


def load_particle_swe(run_dir: str | Path) -> pd.DataFrame:
    """Member SWE at each assimilation date, averaged over observed HRUs.

    One row per date, one column per member. Empty for runs that did not
    write ``assimilation/<date>_swe_particles.csv``.
    """
    folder = Path(run_dir) / "assimilation"
    rows = {}
    for path in sorted(folder.glob("*_swe_particles.csv")):
        frame = pd.read_csv(path)
        if frame.empty or "time" not in frame.columns:
            continue
        when = pd.Timestamp(frame["time"].iloc[0])
        members = frame.drop(columns=[c for c in ("hru", "time") if c in frame.columns])
        rows[when] = members.mean(axis=0, skipna=True)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).T.sort_index()
    out.columns = [str(column) for column in out.columns]
    return out


def empirical_return_period_thresholds(
    obs: xr.DataArray, return_periods: list[float]
) -> xr.DataArray:
    """Peak-flow thresholds from the observed annual maxima.

    The threshold for return period ``T`` is the ``1 - 1/T`` quantile of the
    calendar-year maxima. Used when no thresholds file is configured.
    """
    series = obs.to_series().dropna()
    annual = series.groupby(series.index.year).max()
    if annual.size < 2:
        raise ValueError("Need at least two years of observations for thresholds.")
    periods = np.asarray(sorted(float(t) for t in return_periods), dtype=float)
    values = np.quantile(annual.to_numpy(dtype=float), 1.0 - 1.0 / periods)
    return xr.DataArray(
        values,
        dims=("return_period",),
        coords={"return_period": periods},
        name="yp",
        attrs={
            "units": obs.attrs.get("units", "m3 s-1"),
            "long_name": "return-period streamflow threshold",
            "source": "empirical quantile of observed annual maxima",
        },
    )
