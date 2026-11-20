"""Diagnostics of the particle-filter updates in an assimilating run.

These describe how the filter behaved. They are not independent verification:
the SWE observations here are the ones that were assimilated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from snowda.evaluation.loaders import WeightUpdates, load_particle_swe


def update_diagnostics(run_dir: str | Path, updates: WeightUpdates) -> pd.DataFrame:
    """One row per assimilation date.

    ``prior_mean`` and ``prior_std`` are the equally weighted ensemble SWE at
    the update. ``posterior_mean`` applies the posterior weights to the same
    members. ``innovation`` is observation minus prior mean.
    """
    table = updates.table.copy()
    table = table.reindex(updates.weights.index)
    table.index.name = "date"
    keep = [
        column
        for column in ("lumped_swe", "n_samples", "tempering_beta", "season_reset")
        if column in table.columns
    ]
    out = table[keep].rename(columns={"lumped_swe": "obs_swe"})
    weights = updates.weights
    out["n_members"] = weights.shape[1]
    out["n_eff"] = 1.0 / np.square(
        weights.div(weights.sum(axis=1), axis=0)
    ).sum(axis=1)
    out["max_weight"] = weights.div(weights.sum(axis=1), axis=0).max(axis=1)
    out["resampled"] = updates.resampled.reindex(weights.index).astype(bool)

    particles = load_particle_swe(run_dir)
    if not particles.empty:
        particles = particles.reindex(index=weights.index, columns=weights.columns)
        normalised = weights.div(weights.sum(axis=1), axis=0)
        out["prior_mean"] = particles.mean(axis=1)
        out["prior_std"] = particles.std(axis=1, ddof=1)
        out["posterior_mean"] = (particles * normalised).sum(axis=1, min_count=1)
        if "obs_swe" in out.columns:
            out["innovation"] = out["obs_swe"] - out["prior_mean"]
            out["posterior_residual"] = out["obs_swe"] - out["posterior_mean"]
    return out.reset_index()
