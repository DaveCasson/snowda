"""Laugh tests for empirical quantile mapping.

The map is y* = F_model^{-1}(F_obs(y)), with Hazen positions p_i = (i - 0.5) / n,
linear interpolation, and no extrapolation past the sample.

Panofsky and Brier (1968), Some Applications of Statistics to Meteorology.
    The empirical quantile-quantile correction.
Boé, Terray, Habets, and Martin (2007), Int. J. Climatol. 27:1643-1655.
    doi:10.1002/joc.1602
    Empirical percentiles with linear interpolation.
Gudmundsson, Bremnes, Haugen, and Engen-Skaugen (2012), HESS 16:3383-3390.
    doi:10.5194/hess-16-3383-2012
    QUANT: empirical CDFs, linear interpolation, constant beyond the sample.
Hazen (1914), Trans. ASCE 77. Plotting position (i - 0.5) / n.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from laugh_tests._figures import save_figure
from snowda.obs_qm.mapping import (
    empirical_cdf_quantile,
    hazen_plotting_positions,
    quantile_map_obs_to_model,
)

_QUANT = (
    "Gudmundsson et al. (2012) QUANT: empirical CDFs, linear interpolation, "
    "no extrapolation"
)
_HAZEN = "Hazen plotting position p_i = (i - 0.5) / n"


def _mapped(
    values: np.ndarray, obs_samples: np.ndarray, model_samples: np.ndarray
) -> np.ndarray:
    return np.array(
        [
            quantile_map_obs_to_model(float(value), obs_samples, model_samples)[0]
            for value in values
        ]
    )


def test_hazen_plotting_positions(laugh) -> None:
    laugh.check(
        "positions for n = 4",
        np.array([0.125, 0.375, 0.625, 0.875]),
        hazen_plotting_positions(4),
        atol=0.0,
        reference=_HAZEN,
    )


def test_median_of_an_odd_sample_is_the_50th_percentile(laugh) -> None:
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    laugh.check(
        "F(3) for five sorted samples",
        0.5,
        empirical_cdf_quantile(3.0, samples),
        atol=0.0,
        reference="With n = 5 the middle rank has p = (3 - 0.5) / 5 = 0.5",
    )


def test_identity_mapping(laugh) -> None:
    """The same sample on both sides is the identity, including between knots."""
    samples = np.array([1.0, 2.0, 4.0, 7.0])
    points = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 7.0])
    laugh.check(
        "mapped values",
        points,
        _mapped(points, samples, samples),
        atol=1e-12,
        reference="F^{-1}(F(y)) = y when both distributions use the same knots",
    )


def test_affine_mapping_is_exact(laugh) -> None:
    """Linear interpolation of a linear quantile map has no interpolation error."""
    obs = np.array([1.0, 2.0, 4.0, 8.0])
    slope = 2.5
    intercept = -0.4
    model = slope * obs + intercept
    points = np.array([1.0, 1.5, 3.0, 6.0, 8.0])
    laugh.check(
        "mapped values",
        slope * points + intercept,
        _mapped(points, obs, model),
        atol=1e-12,
        reference="A positive affine QQ relationship is reproduced exactly",
    )


def test_exponential_quantiles_map_by_the_scale_ratio(laugh) -> None:
    """Exp(scale=2) quantiles are twice Exp(scale=1) quantiles at the same probability."""
    probability = hazen_plotting_positions(21)
    obs = stats.expon.ppf(probability, scale=1.0)
    model = stats.expon.ppf(probability, scale=2.0)
    points = np.linspace(float(obs[0]), float(obs[-1]), 11)
    laugh.check(
        "mapped values along y* = 2y",
        2.0 * points,
        _mapped(points, obs, model),
        atol=1e-12,
        reference="Matched exponential quantiles lie on a straight QQ line",
    )


def _mean_absolute_error(n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    obs = rng.exponential(1.0, size=n)
    model = rng.exponential(2.0, size=n)
    low, high = np.quantile(obs, [0.25, 0.75])
    points = np.linspace(low, high, 15)
    error = np.abs(_mapped(points, obs, model) - 2.0 * points)
    return float(np.mean(error))


def test_random_sample_error_shrinks_with_n(laugh) -> None:
    """Independent draws: the empirical map approaches y* = 2y as n grows."""
    seeds = range(30)
    error_10 = float(np.mean([_mean_absolute_error(10, seed) for seed in seeds]))
    error_100 = float(np.mean([_mean_absolute_error(100, seed) for seed in seeds]))
    error_1000 = float(np.mean([_mean_absolute_error(1000, seed) for seed in seeds]))
    laugh.check(
        (
            f"mean absolute error decreases "
            f"({error_10:.3f} at n=10, {error_100:.3f} at n=100, {error_1000:.3f} at n=1000)"
        ),
        True,
        bool(error_10 > error_100 > error_1000),
        reference=_QUANT + "; population map of Exp(1) onto Exp(2) is y* = 2y",
    )


def test_mapping_is_monotone(laugh) -> None:
    obs = np.array([1.0, 2.0, 5.0, 9.0, 12.0])
    model = np.array([0.5, 1.0, 4.0, 8.0, 30.0])
    grid = np.linspace(1.0, 12.0, 41)
    mapped = _mapped(grid, obs, model)
    largest_decrease = float(np.maximum(0.0, -np.min(np.diff(mapped))))
    laugh.check(
        "largest decrease along an increasing grid",
        0.0,
        largest_decrease,
        atol=0.0,
        reference="An empirical quantile map is non-decreasing",
    )


def test_outside_the_sample_is_clipped(laugh) -> None:
    obs = np.array([2.0, 4.0, 6.0, 8.0])
    model = np.array([10.0, 20.0, 30.0, 40.0])
    below, low_probability = quantile_map_obs_to_model(0.0, obs, model)
    above, high_probability = quantile_map_obs_to_model(100.0, obs, model)
    positions = hazen_plotting_positions(4)
    laugh.check(
        "value below the obs sample",
        10.0,
        below,
        atol=1e-12,
        reference=_QUANT,
    )
    laugh.check(
        "value above the obs sample",
        40.0,
        above,
        atol=1e-12,
        reference=_QUANT,
    )
    laugh.check(
        "clipped low probability",
        float(positions[0]),
        low_probability,
        atol=1e-12,
        reference=_HAZEN,
    )
    laugh.check(
        "clipped high probability",
        float(positions[-1]),
        high_probability,
        atol=1e-12,
        reference=_HAZEN,
    )


def test_figure_error_shrinks_with_sample_size(laugh) -> None:
    """Thirty independent samples at each size. The mean error should fall."""
    sizes = (10, 100, 1000)
    seeds = range(30)
    samples = [
        np.array([_mean_absolute_error(n, seed) for seed in seeds]) for n in sizes
    ]
    means = np.array([float(np.mean(values)) for values in samples])

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for index, (n, values) in enumerate(zip(sizes, samples)):
        ax.scatter(
            np.full(values.shape, n),
            values,
            s=16,
            color="0.65",
            zorder=2,
            label="one sample" if index == 0 else None,
        )
    ax.plot(sizes, means, "o-", color="C0", ms=6, zorder=3, label="mean over 30 samples")
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.set_xlabel("sample size n")
    ax.set_ylabel("mean absolute error from y* = 2y")
    ax.set_title("Empirical quantile map, Exp(1) onto Exp(2)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "quantile_mapping_convergence.png")
    laugh.check(
        "mean absolute error decreases with n",
        True,
        bool(means[0] > means[1] > means[2]),
        reference=_QUANT + "; population map of Exp(1) onto Exp(2) is y* = 2y",
    )


def test_figure_transfer_function(laugh) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    largest_gap = 0.0
    for n, marker, size in ((20, "o", 5), (200, ".", 8)):
        probability = hazen_plotting_positions(n)
        obs = stats.expon.ppf(probability, scale=1.0)
        model = stats.expon.ppf(probability, scale=2.0)
        points = np.linspace(float(obs[1]), float(obs[-2]), 60)
        mapped = _mapped(points, obs, model)
        largest_gap = max(largest_gap, float(np.max(np.abs(mapped - 2.0 * points))))
        ax.plot(points, mapped, marker, ms=size, label=f"empirical map, n={n}")
    line = np.linspace(0.05, 3.0, 40)
    ax.plot(line, 2.0 * line, "k--", lw=1.0, label="analytical y* = 2y")
    ax.set_xlabel("observed value")
    ax.set_ylabel("value in model space")
    ax.set_title("Exponential quantile map, scale 1 onto scale 2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "quantile_mapping_transfer.png")
    laugh.check(
        "largest gap from y* = 2y at n = 20 and n = 200",
        0.0,
        largest_gap,
        atol=1e-10,
        reference="Matched quantiles of Exp(1) and Exp(2) lie on y* = 2y",
    )
