"""Particle weights on a daily time axis, and weighted ensemble statistics.

SIS never changes a member's trace. Its streamflow estimate is the unchanged
historic ensemble with the filter weights applied, so the weighting is done
here, when a run is evaluated, and not while the run is in progress.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from snowda.core.particle_filter import season_of

# Outer band, inner band and median used in every table and figure.
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def filter_weight_timeline(
    updates: pd.DataFrame,
    times: pd.DatetimeIndex,
    *,
    resampled: Optional[Sequence[bool]] = None,
    season_start_month: Optional[int] = None,
) -> pd.DataFrame:
    """Filter weights in force on each day of ``times``, one column per member.

    ``updates`` holds one row of posterior weights per assimilation date. A
    day uses the latest update on or before it, so a weight only reflects
    observations already made. Days use equal weights when:

    - no update has happened yet;
    - that update resampled the ensemble (``resampled``), because the states
      then already carry the information and the weights were reset to 1/N;
    - ``season_start_month`` is set and the day is in a later season than the
      update. The reset happens on the first day of the season, not at the
      next update, so one season's weights never reach the following winter.
    """
    members = list(updates.columns)
    n_members = len(members)
    if n_members == 0:
        raise ValueError("Weight updates have no member columns.")
    days = pd.DatetimeIndex(times).normalize()
    out = np.full((days.size, n_members), 1.0 / n_members, dtype=float)
    if len(updates) == 0:
        return pd.DataFrame(out, index=times, columns=members)

    order = np.argsort(pd.DatetimeIndex(updates.index).to_numpy())
    update_days = pd.DatetimeIndex(updates.index)[order].normalize()
    values = updates.to_numpy(dtype=float)[order]
    totals = values.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(totals <= 0):
        raise ValueError("Weight updates must be finite, non-negative and not all zero.")
    values = values / totals
    if resampled is None:
        was_resampled = np.zeros(len(update_days), dtype=bool)
    else:
        was_resampled = np.asarray(list(resampled), dtype=bool)[order]
        if was_resampled.size != len(update_days):
            raise ValueError("`resampled` needs one flag per weight update.")

    latest = np.searchsorted(update_days.to_numpy(), days.to_numpy(), side="right") - 1
    use = latest >= 0
    use[use] &= ~was_resampled[latest[use]]
    if season_start_month is not None:
        update_season = np.asarray(
            [season_of(day, season_start_month) for day in update_days]
        )
        day_season = np.where(
            days.month >= int(season_start_month), days.year, days.year - 1
        )
        use[use] &= update_season[latest[use]] == day_season[use]
    out[use] = values[latest[use]]
    return pd.DataFrame(out, index=times, columns=members)


def effective_weights(values: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    """Row-normalised weights with zero on missing members.

    ``values`` is ``(time, member)``. ``weights`` is the same shape, one row
    of shape ``(member,)``, or ``None`` for equal weights. A row with no
    finite member is all NaN.
    """
    data = np.asarray(values, dtype=float)
    if weights is None:
        w = np.ones_like(data)
    else:
        w = np.broadcast_to(np.asarray(weights, dtype=float), data.shape).copy()
    w = np.where(np.isfinite(data), w, 0.0)
    total = w.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, w / total, np.nan)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean across members for each time. ``weights`` from ``effective_weights``."""
    data = np.where(np.isfinite(values), values, 0.0)
    return np.sum(data * weights, axis=1)


def weighted_variance(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = weighted_mean(values, weights)
    deviation = np.where(np.isfinite(values), values - mean[:, None], 0.0)
    return np.sum(weights * deviation**2, axis=1)


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float] = QUANTILES,
) -> np.ndarray:
    """Weighted quantiles for each time, shape ``(time, quantile)``.

    A quantile is the smallest member value whose cumulative weight reaches
    ``q``, the same rule as ``core.particle_filter.weighted_ensemble_summary``.
    Equal weights give the unweighted ensemble quantile under that rule.
    """
    data = np.asarray(values, dtype=float)
    filled = np.where(np.isfinite(data), data, np.inf)
    order = np.argsort(filled, axis=1)
    sorted_values = np.take_along_axis(filled, order, axis=1)
    cdf = np.cumsum(np.take_along_axis(np.nan_to_num(weights), order, axis=1), axis=1)
    n_valid = np.isfinite(data).sum(axis=1)
    rows = np.arange(data.shape[0])
    out = np.full((data.shape[0], len(quantiles)), np.nan, dtype=float)
    for j, q in enumerate(quantiles):
        index = (cdf < q - 1e-12).sum(axis=1)
        index = np.minimum(index, np.maximum(n_valid - 1, 0))
        picked = sorted_values[rows, index]
        out[:, j] = np.where(n_valid > 0, picked, np.nan)
    return out


def n_eff(weights: np.ndarray) -> np.ndarray:
    """Effective sample size ``1 / sum(w^2)`` for each time."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 / np.sum(np.square(weights), axis=1)
