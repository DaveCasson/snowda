"""Empirical quantile mapping from observation SWE to model SWE space.

The transformation is always

    y_model_equiv = F_mod^{-1}(F_obs(y_obs))

using empirical distributions (no parametric fit). Observed and modeled
samples are sorted independently. Contemporaneous pairing is used only to
choose which values enter the two samples, not to match pair-by-pair.

Plotting positions follow the Hazen convention

    p_i = (i - 0.5) / n    for sorted rank i = 1, ..., n.

Linear interpolation is used between empirical quantiles. Values outside
the calibration range are not extrapolated: the percentile is clipped to
the supported empirical range [p_min, p_max] = [0.5/n, (n - 0.5)/n]
(after collapsing ties, the endpoints are the mean Hazen positions of the
extreme unique values).
"""

from __future__ import annotations

import numpy as np

# Hazen plotting position: p_i = (i - 0.5) / n
PLOTTING_POSITION = "hazen_(i-0.5)/n"


def hazen_plotting_positions(n: int) -> np.ndarray:
    """Return Hazen plotting positions for ranks 1..n."""
    if n < 1:
        raise ValueError("n must be >= 1 to form plotting positions.")
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return (ranks - 0.5) / n


def empirical_quantiles(samples) -> tuple[np.ndarray, np.ndarray, int]:
    """Return unique sorted values and mean Hazen positions.

    Tied sample values are collapsed so the quantile grid is strictly
    increasing. The plotting position of a tied value is the mean of the
    Hazen positions of the tied ranks.

    Returns
    -------
    x_unique : ndarray
        Unique sample values in ascending order.
    p_unique : ndarray
        Mean Hazen plotting positions for each unique value.
    n : int
        Number of finite samples before collapsing ties.
    """
    values = np.asarray(samples, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < 1:
        raise ValueError("Need at least one finite sample for an empirical CDF.")

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    p = hazen_plotting_positions(n)

    x_unique, inverse = np.unique(sorted_values, return_inverse=True)
    counts = np.bincount(inverse)
    p_unique = np.bincount(inverse, weights=p) / counts
    return x_unique, p_unique, n


def empirical_cdf_quantile(value: float, samples) -> float:
    """Estimate F(value) by linear interpolation on the empirical CDF.

    If ``value`` lies below the smallest sample, the result is clipped to
    ``p_min``. If it lies above the largest sample, the result is clipped
    to ``p_max``. ``numpy.interp`` performs this clipping; it is applied
    explicitly as well so the bound is documented in the return path.
    """
    x, p, _n = empirical_quantiles(samples)
    q = float(np.interp(value, x, p))
    return float(np.clip(q, p[0], p[-1]))


def empirical_inverse_cdf(quantile: float, samples) -> float:
    """Evaluate the empirical quantile function F^{-1}(quantile).

    ``quantile`` is clipped to the supported empirical range before
    interpolation, so the result never extrapolates beyond the sample min
    or max.
    """
    x, p, _n = empirical_quantiles(samples)
    q = float(np.clip(quantile, p[0], p[-1]))
    return float(np.interp(q, p, x))


def quantile_map_obs_to_model(
    obs_value: float,
    obs_samples,
    model_samples,
) -> tuple[float, float]:
    """Map one observation into model SWE space.

    Parameters
    ----------
    obs_value :
        Observed SWE (must already be positive; zeros are handled by the
        caller).
    obs_samples, model_samples :
        Finite calibration samples. They should come from the same dates
        but are sorted independently.

    Returns
    -------
    y_model_equiv : float
        ``F_mod^{-1}(F_obs(obs_value))``.
    quantile_obs : float
        Clipped empirical percentile of ``obs_value`` in ``F_obs``.
    """
    q = empirical_cdf_quantile(obs_value, obs_samples)
    y_star = empirical_inverse_cdf(q, model_samples)
    return y_star, q
