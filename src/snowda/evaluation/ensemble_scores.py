"""Scores for a weighted ensemble.

Every function takes ``values`` shaped ``(time, member)`` and ``weights``
from ``snowda.evaluation.weights.effective_weights``. One member with weight
one is a deterministic run, and equal weights are an unweighted ensemble, so
a deterministic run, an ensemble and an SIS-weighted ensemble are all scored
by the same code and on the same scale.
"""

from __future__ import annotations

import numpy as np

from snowda.evaluation.weights import weighted_mean, weighted_variance

PIT_BINS = 10


def weighted_crps(values: np.ndarray, weights: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """CRPS of the weighted empirical distribution, one value per time.

    ``CRPS = sum_i w_i |x_i - y| - 0.5 sum_ij w_i w_j |x_i - x_j|``
    (Gneiting and Raftery 2007, Eq. 21, with weights). With one member this
    is the absolute error. It is the plain, not the "fair", estimator: the
    fair correction is undefined for one member and for unequal weights.
    """
    data = np.asarray(values, dtype=float)
    y = np.asarray(obs, dtype=float)
    w = np.nan_to_num(np.asarray(weights, dtype=float))
    filled = np.where(np.isfinite(data), data, 0.0)
    accuracy = np.sum(w * np.abs(filled - y[:, None]), axis=1)
    order = np.argsort(filled, axis=1)
    x_sorted = np.take_along_axis(filled, order, axis=1)
    w_sorted = np.take_along_axis(w, order, axis=1)
    cumulative = np.cumsum(w_sorted, axis=1)
    # 0.5 * sum_ij w_i w_j |x_i - x_j| = sum_i w_i x_i (2 W_i - w_i - 1)
    spread = np.sum(w_sorted * x_sorted * (2.0 * cumulative - w_sorted - 1.0), axis=1)
    crps = accuracy - spread
    valid = np.isfinite(y) & (np.isfinite(data).any(axis=1))
    return np.where(valid, crps, np.nan)


def weighted_pit(
    values: np.ndarray, weights: np.ndarray, obs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interval of the probability integral transform for each time.

    With ``N`` equally weighted members the observation has one of ``N + 1``
    ranks, each covering ``1 / (N + 1)`` of [0, 1]. This is that rank
    interval, generalised to unequal weights: ``b`` is the weight below the
    observation (half of any tied weight included) and ``N`` is the
    effective sample size ``1 / sum(w^2)``. The interval is
    ``[b N / (N + 1), b N / (N + 1) + 1 / (N + 1)]``. Returns its lower and
    upper bounds, NaN where the observation or every member is missing.
    """
    data = np.asarray(values, dtype=float)
    y = np.asarray(obs, dtype=float)[:, None]
    w = np.nan_to_num(np.asarray(weights, dtype=float))
    with np.errstate(invalid="ignore", divide="ignore"):
        below = np.sum(w * (data < y), axis=1) + 0.5 * np.sum(w * (data == y), axis=1)
        size = 1.0 / np.sum(np.square(w), axis=1)
    lower = np.clip(below, 0.0, 1.0) * size / (size + 1.0)
    upper = lower + 1.0 / (size + 1.0)
    valid = np.isfinite(y[:, 0]) & np.isfinite(data).any(axis=1)
    return np.where(valid, lower, np.nan), np.where(valid, upper, np.nan)


def pit_histogram(
    lower: np.ndarray, upper: np.ndarray, bins: int = PIT_BINS
) -> np.ndarray:
    """Relative frequency of PIT intervals in ``bins`` equal bins on [0, 1].

    Each interval is spread over the bins it overlaps, so the result does not
    depend on how the ensemble size divides the number of bins. A calibrated
    ensemble gives a flat histogram.
    """
    valid = np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
    if not valid.any():
        return np.full(bins, np.nan)
    low = lower[valid][:, None]
    high = upper[valid][:, None]
    edges = np.linspace(0.0, 1.0, bins + 1)
    overlap = np.clip(
        np.minimum(high, edges[None, 1:]) - np.maximum(low, edges[None, :-1]), 0.0, None
    )
    return (overlap / (high - low)).sum(axis=0) / int(valid.sum())


def spread_skill_ratio(values: np.ndarray, weights: np.ndarray, obs: np.ndarray) -> float:
    """Root mean ensemble variance over RMSE of the ensemble mean.

    Near one for a well-dispersed ensemble, below one when it is too narrow.
    NaN for a single member, which has no spread.
    """
    if np.isfinite(values).sum(axis=1).max(initial=0) < 2:
        return float("nan")
    y = np.asarray(obs, dtype=float)
    mean = weighted_mean(values, weights)
    variance = weighted_variance(values, weights)
    valid = np.isfinite(y) & np.isfinite(mean) & np.isfinite(variance)
    if not valid.any():
        return float("nan")
    rmse = np.sqrt(np.mean((mean[valid] - y[valid]) ** 2))
    if rmse == 0:
        return float("nan")
    return float(np.sqrt(np.mean(variance[valid])) / rmse)


def interval_coverage(lower: np.ndarray, upper: np.ndarray, obs: np.ndarray) -> float:
    """Fraction of observations inside ``[lower, upper]``."""
    y = np.asarray(obs, dtype=float)
    valid = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
    if not valid.any():
        return float("nan")
    inside = (y[valid] >= lower[valid]) & (y[valid] <= upper[valid])
    return float(np.mean(inside))
