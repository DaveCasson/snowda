"""Figures and skill scores for the snow5 synthetic (twin) experiment.

Each mode gives an ensemble of SWE trajectories and a weight per member and
day. The historic mean is a one-member ensemble; the open-loop ensemble has
equal weights. For SIR and SIS, the weight at an update date is that update's
posterior (the stored trajectory value there is the pre-resample forecast).
Between updates it is the prior carried to the next update: equal after a
resample, otherwise the posterior. Before the first update weights are equal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Reference categorical palette (light), fixed order. Text stays in ink.
MODE_COLORS = {
    "historic_mean": "#2a78d6",
    "historic_ensemble": "#eb6834",
    "sir": "#1baf7a",
    "sis": "#eda100",
}
MODE_LABELS = {
    "historic_mean": "Historic mean",
    "historic_ensemble": "Historic ensemble",
    "sir": "SIR",
    "sis": "SIS",
}
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"


@dataclass
class ModeResult:
    mode: str
    run_dir: Path
    time: pd.DatetimeIndex
    members: np.ndarray  # (time, member)
    weights: np.ndarray  # (time, member), rows sum to 1
    updates: pd.DataFrame | None = None  # time, n_eff, resampled, updated
    # Prior weights at each update date (before that date's observation).
    prior: dict | None = None


def latest_run_dir(mode_dir: Path) -> Path:
    runs = sorted(p for p in Path(mode_dir).iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No runs in {mode_dir}")
    return runs[-1]


def _swe(path: Path, hru: int = 0) -> pd.Series:
    with xr.open_dataset(path) as ds:
        return ds["swe"].isel(hru=hru).to_series()


def load_mean(mode_dir: Path) -> ModeResult:
    run = latest_run_dir(mode_dir)
    s = _swe(run / "snow5_output.nc")
    return ModeResult("historic_mean", run, s.index, s.values[:, None], np.ones((len(s), 1)))


def load_ensemble(mode_dir: Path) -> ModeResult:
    run = latest_run_dir(mode_dir)
    files = sorted(run.glob("*/*_snow5_output.nc"))
    series = [_swe(f) for f in files]
    x = np.column_stack([s.values for s in series])
    w = np.full_like(x, 1.0 / x.shape[1])
    return ModeResult("historic_ensemble", run, series[0].index, x, w)


def load_da(mode: str, mode_dir: Path) -> ModeResult:
    run = latest_run_dir(mode_dir)
    prov = json.loads((run / "provenance.json").read_text())
    files = sorted(run.glob("*_snow5_output.nc"))
    labels = [f.name.split("_")[0] for f in files]
    series = [_swe(f) for f in files]
    time = series[0].index
    x = np.column_stack([s.values for s in series])
    n = x.shape[1]
    w = np.full_like(x, 1.0 / n)
    rows = []
    prior = {}
    records = sorted(prov["records"], key=lambda r: r["time"])
    for j, rec in enumerate(records):
        t = pd.Timestamp(rec["time"])
        post = np.array([rec["weights"][lab] for lab in labels])
        if "prior_weights" in rec:
            prior[t] = np.array([rec["prior_weights"][lab] for lab in labels])
        resampled = rec["resample"] is not None
        carry = np.full(n, 1.0 / n) if resampled else post
        nxt = pd.Timestamp(records[j + 1]["time"]) if j + 1 < len(records) else time[-1] + pd.Timedelta(days=1)
        w[time == t] = post
        w[(time > t) & (time < nxt)] = carry
        rows.append(
            {
                "time": t,
                "n_eff": rec["n_eff"],
                "resampled": resampled,
                "updated": rec.get("updated", True),
            }
        )
    updates = pd.DataFrame(rows)
    updates = updates[updates["updated"]].reset_index(drop=True)
    return ModeResult(mode, run, time, x, w, updates, prior or None)


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> np.ndarray:
    """Per-row weighted quantile of ``x`` (time, member)."""
    order = np.argsort(x, axis=1)
    xs = np.take_along_axis(x, order, axis=1)
    ws = np.take_along_axis(w, order, axis=1)
    cdf = np.cumsum(ws, axis=1) - 0.5 * ws
    cdf /= ws.sum(axis=1, keepdims=True)
    return np.array([np.interp(q, cdf[i], xs[i]) for i in range(len(xs))])


def weighted_crps(x: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of a weighted ensemble per row: E|X-y| - 0.5 E|X-X'|."""
    term1 = np.sum(w * np.abs(x - y[:, None]), axis=1)
    pair = np.abs(x[:, :, None] - x[:, None, :])
    term2 = 0.5 * np.einsum("ti,tij,tj->t", w, pair, w)
    return term1 - term2


def scores(res: ModeResult, truth: pd.Series) -> dict:
    y = truth.reindex(res.time).values
    est = np.sum(res.weights * res.members, axis=1)
    spread = np.sqrt(np.sum(res.weights * (res.members - est[:, None]) ** 2, axis=1))
    err = est - y
    snow = y > 0
    wy = pd.Series(res.time.year + (res.time.month >= 10), index=res.time)
    peak_err = (
        pd.Series(est, index=res.time).groupby(wy.values).max()
        - pd.Series(y, index=res.time).groupby(wy.values).max()
    )
    out = {
        "mode": MODE_LABELS[res.mode],
        "members": res.members.shape[1],
        "rmse": float(np.sqrt(np.mean(err**2))),
        "rmse_snow_on": float(np.sqrt(np.mean(err[snow] ** 2))),
        "bias": float(np.mean(err)),
        "crps": float(np.mean(weighted_crps(res.members, res.weights, y))),
        "mean_spread": float(np.mean(spread[snow])),
        "peak_swe_mae": float(np.mean(np.abs(peak_err))),
    }
    if res.updates is not None:
        out["mean_n_eff_frac"] = float(res.updates["n_eff"].mean() / res.members.shape[1])
        out["n_resamples"] = int(res.updates["resampled"].sum())
    return out


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)


def _window(res: ModeResult, start, end) -> ModeResult:
    sel = np.ones(len(res.time), bool)
    if start is not None:
        sel &= res.time >= pd.Timestamp(start)
    if end is not None:
        sel &= res.time <= pd.Timestamp(end)
    upd = res.updates
    if upd is not None:
        upd = upd[(upd["time"] >= res.time[sel][0]) & (upd["time"] <= res.time[sel][-1])]
    return ModeResult(res.mode, res.run_dir, res.time[sel], res.members[sel], res.weights[sel], upd, res.prior)


def plot_timeseries(
    results: list[ModeResult],
    truth: pd.Series | None,
    obs: pd.Series,
    path: Path,
    *,
    title: str = "snow5 synthetic experiment, Ross — SWE vs truth (member 050)",
    start=None,
    end=None,
    truth_label: str = "Truth",
) -> Path:
    results = [_window(r, start, end) for r in results]
    if truth is not None:
        truth = truth[slice(start, end)]
    fig, axes = plt.subplots(len(results), 1, figsize=(13, 2.6 * len(results)), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, res in zip(np.atleast_1d(axes), results):
        _style(ax)
        c = MODE_COLORS[res.mode]
        est = np.sum(res.weights * res.members, axis=1)
        if res.members.shape[1] > 1:
            lo = weighted_quantile(res.members, res.weights, 0.05)
            hi = weighted_quantile(res.members, res.weights, 0.95)
            ax.fill_between(res.time, lo, hi, color=c, alpha=0.22, linewidth=0, label="5–95%")
        if res.updates is not None:
            ot = res.updates["time"]
            ax.scatter(ot, obs.reindex(ot).values, s=10, color=MUTED, zorder=3, label="Assimilated obs")
            rs = res.updates.loc[res.updates["resampled"], "time"]
            ax.vlines(rs, 0, 0.04, transform=ax.get_xaxis_transform(), color=INK_2, linewidth=0.8, label="Resample")
        elif truth is None:
            ot = obs[(obs.index >= res.time[0]) & (obs.index <= res.time[-1])].dropna()
            ax.scatter(ot.index, ot.values, s=10, color=MUTED, zorder=3, label="Survey obs")
        if truth is not None:
            ax.plot(truth.index, truth.values, color=INK, linewidth=1.2, label=truth_label)
        ax.plot(res.time, est, color=c, linewidth=2, label="Estimate (weighted mean)")
        ax.set_title(MODE_LABELS[res.mode], loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_ylabel("SWE (kg m⁻²)", color=INK_2, fontsize=9)
        ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=5)
    fig.suptitle(title, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_water_year(results: list[ModeResult], truth: pd.Series, obs: pd.Series, wy: int, path: Path) -> Path:
    start, end = pd.Timestamp(f"{wy - 1}-10-01"), pd.Timestamp(f"{wy}-09-30")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot(truth[start:end].index, truth[start:end].values, color=INK, linewidth=1.4, label="Truth")
    for res in results:
        sel = (res.time >= start) & (res.time <= end)
        est = np.sum(res.weights * res.members, axis=1)[sel]
        ax.plot(res.time[sel], est, color=MODE_COLORS[res.mode], linewidth=2, label=MODE_LABELS[res.mode])
    da = next((r for r in results if r.updates is not None), None)
    if da is not None:
        ot = da.updates["time"][(da.updates["time"] >= start) & (da.updates["time"] <= end)]
        ax.scatter(ot, obs.reindex(ot).values, s=14, color=MUTED, zorder=3, label="Assimilated obs")
    ax.set_ylabel("SWE (kg m⁻²)", color=INK_2)
    ax.set_title(f"Water year {wy}: point estimates", loc="left", color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_neff(results: list[ModeResult], path: Path) -> Path:
    da = [r for r in results if r.updates is not None]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for res in da:
        u = res.updates
        frac = u["n_eff"] / res.members.shape[1]
        ax.plot(u["time"], frac, color=MODE_COLORS[res.mode], linewidth=2, marker="o", markersize=4,
                label=MODE_LABELS[res.mode])
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.text(da[0].updates["time"].iloc[0], 0.52, "SIR resample threshold", fontsize=8, color=INK_2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("N_eff / N (posterior)", color=INK_2)
    ax.set_title("Effective sample size at each update", loc="left", color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_scores(table: pd.DataFrame, path: Path) -> Path:
    metrics = [("rmse", "RMSE"), ("crps", "CRPS"), ("peak_swe_mae", "Peak SWE abs. error")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3.4))
    fig.patch.set_facecolor(SURFACE)
    keys = [k for k, v in MODE_LABELS.items() if v in set(table["mode"])]
    for ax, (col, title) in zip(axes, metrics):
        _style(ax)
        vals = table.set_index("mode").loc[[MODE_LABELS[k] for k in keys], col].values
        bars = ax.bar(range(len(keys)), vals, color=[MODE_COLORS[k] for k in keys], width=0.6,
                      edgecolor=SURFACE, linewidth=2)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, color=INK_2, padding=2)
        ax.set_xticks(range(len(keys)), [MODE_LABELS[k].replace(" ", "\n") for k in keys], fontsize=8)
        ax.set_title(f"{title} (kg m⁻²)", loc="left", fontsize=10, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
