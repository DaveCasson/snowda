"""Score snow5 historic modes against snow-survey SWE.

Scores are on survey dates only. For the DA modes two estimates are scored:

- ``prior``: weights from earlier surveys only (before the survey on that date
  is assimilated). Independent of the observation being scored; this is the
  skill measure. The first survey of each season uses equal weights.
- ``analysis``: posterior weights after assimilating that survey. This is the
  fit to the assimilated data, not independent skill.

Open-loop modes have one estimate (equal weights), reported as both.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from snowda.viz.snow5_synthetic import (
    INK,
    INK_2,
    MODE_COLORS,
    MODE_LABELS,
    MUTED,
    SURFACE,
    ModeResult,
    _style,
    weighted_crps,
    weighted_quantile,
)


def estimates_at(res: ModeResult, dates: pd.DatetimeIndex, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Members (date, member) and weights at ``dates`` for ``kind`` prior/analysis."""
    idx = res.time.get_indexer(dates)
    if (idx < 0).any():
        raise ValueError(f"{res.mode}: survey dates outside the run period")
    x = res.members[idx]
    w = res.weights[idx].copy()
    if kind == "prior" and res.prior is not None:
        for i, d in enumerate(dates):
            if d in res.prior:
                w[i] = res.prior[d]
    return x, w / w.sum(axis=1, keepdims=True)


def score(x: np.ndarray, w: np.ndarray, y: np.ndarray) -> dict:
    est = np.sum(w * x, axis=1)
    err = est - y
    lo = weighted_quantile(x, w, 0.05)
    hi = weighted_quantile(x, w, 0.95)
    spread = np.sqrt(np.sum(w * (x - est[:, None]) ** 2, axis=1))
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r": float(np.corrcoef(est, y)[0, 1]) if len(y) > 2 else np.nan,
        "crps": float(np.mean(weighted_crps(x, w, y))),
        "mean_spread": float(np.mean(spread)),
        "coverage_90": float(np.mean((y >= lo) & (y <= hi))) if x.shape[1] > 1 else np.nan,
    }


def score_table(results: list[ModeResult], obs: dict[str, pd.Series]) -> pd.DataFrame:
    """One row per mode x estimate x observation variable."""
    rows = []
    for res in results:
        kinds = ["prior", "analysis"] if res.prior is not None else ["open_loop"]
        for obs_name, series in obs.items():
            s = series.dropna()
            s = s[(s.index >= res.time[0]) & (s.index <= res.time[-1])]
            for kind in kinds:
                x, w = estimates_at(res, s.index, "prior" if kind != "analysis" else "analysis")
                rows.append({"mode": MODE_LABELS[res.mode], "estimate": kind, "obs": obs_name, **score(x, w, s.values)})
    return pd.DataFrame(rows)


def plot_scatter(results: list[ModeResult], obs: pd.Series, path: Path, obs_label: str) -> Path:
    s = obs.dropna()
    fig, axes = plt.subplots(1, len(results), figsize=(3.4 * len(results), 3.6), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    top = float(np.nanmax(s.values)) * 1.1
    for ax, res in zip(np.atleast_1d(axes), results):
        _style(ax)
        x, w = estimates_at(res, s.index, "prior")
        est = np.sum(w * x, axis=1)
        ax.plot([0, top], [0, top], color=MUTED, linewidth=1, linestyle="--")
        ax.scatter(s.values, est, s=14, color=MODE_COLORS[res.mode], edgecolor=SURFACE, linewidth=0.5)
        rmse = np.sqrt(np.mean((est - s.values) ** 2))
        name = MODE_LABELS[res.mode] + (" (prior)" if res.prior is not None else "")
        ax.set_title(name, loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.text(0.04, 0.92, f"RMSE {rmse:.0f}", transform=ax.transAxes, fontsize=9, color=INK_2)
        ax.set_xlabel(f"{obs_label} (kg m⁻²)", color=INK_2, fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("snow5 estimate (kg m⁻²)", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_skill_bars(table: pd.DataFrame, obs_name: str, path: Path) -> Path:
    """RMSE and CRPS per mode: open loop, DA prior (skill) and DA analysis (fit)."""
    t = table[table["obs"] == obs_name]
    order = [m for m in MODE_LABELS.values() if m in set(t["mode"])]
    key_of = {v: k for k, v in MODE_LABELS.items()}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    fig.patch.set_facecolor(SURFACE)
    for ax, col in zip(axes, ["rmse", "crps"]):
        _style(ax)
        xs, labels = [], []
        pos = 0
        for mode in order:
            sub = t[t["mode"] == mode].set_index("estimate")
            for kind in [k for k in ("open_loop", "prior", "analysis") if k in sub.index]:
                val = sub.loc[kind, col]
                hatch = "///" if kind == "analysis" else None
                alpha = 0.55 if kind == "analysis" else 1.0
                b = ax.bar(pos, val, color=MODE_COLORS[key_of[mode]], alpha=alpha, hatch=hatch,
                           edgecolor=SURFACE, linewidth=2, width=0.8)
                ax.bar_label(b, fmt="%.0f", fontsize=8, color=INK_2, padding=2)
                xs.append(pos)
                labels.append(mode.replace(" ", "\n") if kind == "open_loop" else f"{mode}\n{kind}")
                pos += 1
            pos += 0.4
        ax.set_xticks(xs, labels, fontsize=8)
        ax.set_title(f"{col.upper()} vs {obs_name} (kg m⁻²)", loc="left", fontsize=10, color=INK, fontweight="bold")
    fig.text(0.01, 0.01, "Prior = before the survey is assimilated (skill). Analysis (hatched) = after (fit to assimilated data).",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
