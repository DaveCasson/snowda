"""Laugh tests for the Liston (2004) lognormal snow depletion curve.

Liston (2004), J. Climate 17(6):1381-1397.
doi:10.1175/1520-0442(2004)017<1381:RSSCHI>2.0.CO;2

Subgrid SWE is lognormal with mean mu and coefficient of variation CV.
After a uniform melt depth dm, fSCA is the survival function P(SWE > dm)
and the grid-mean SWE is the partial expectation E[(SWE - dm)+].
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, stats

from laugh_tests._figures import save_figure
from snowda.core.ssnowd import (
    calc_fsca_and_mean_swe,
    calc_lambda,
    calc_swe_frac_area,
    calc_zeta,
    solve_dm_for_mean_swe,
)

_LISTON = (
    "Liston (2004): lognormal SWE, zeta = sqrt(ln(1+CV^2)), "
    "lambda = ln(mu) - zeta^2/2"
)


def _analytic_depletion(mu: float, dm: float, cv: float) -> tuple[float, float]:
    """fSCA and mean SWE from scipy's lognormal, not from ``ssnowd``.

    The mean uses the partial-expectation identity
    E[X 1_{X>dm}] = mu * (1 - Phi((ln dm - lambda) / zeta - zeta)).
    """
    zeta = float(np.sqrt(np.log(1.0 + cv**2)))
    lam = float(np.log(mu) - 0.5 * zeta**2)
    if dm <= 0.0:
        return 1.0, float(mu)
    dist = stats.lognorm(s=zeta, scale=np.exp(lam))
    fsca = float(dist.sf(dm))
    z = (np.log(dm) - lam) / zeta
    tail_mean = mu * float(stats.norm.sf(z - zeta))
    return fsca, float(tail_mean - dm * fsca)


def test_parameters_have_the_right_mean(laugh) -> None:
    mu = 2.5
    cv = 0.4
    zeta = calc_zeta(cv)
    lam = calc_lambda(zeta, mu)
    laugh.check(
        "exp(lambda + zeta^2/2)",
        mu,
        float(np.exp(lam + 0.5 * zeta**2)),
        rtol=1e-12,
        reference=_LISTON,
    )
    dist = stats.lognorm(s=zeta, scale=np.exp(lam))
    laugh.check(
        "scipy lognormal mean",
        mu,
        float(dist.mean()),
        rtol=1e-12,
        reference=_LISTON,
    )


def test_depletion_matches_the_lognormal(laugh) -> None:
    mu, cv, dm = 1.0, 0.5, 0.4
    fsca, swe_mean, _z, lam, zeta = calc_fsca_and_mean_swe(mu, dm, cv)
    dist = stats.lognorm(s=zeta, scale=np.exp(lam))
    integral, _err = integrate.quad(
        lambda x: (x - dm) * dist.pdf(x),
        dm,
        np.inf,
        epsabs=1e-12,
    )
    laugh.check(
        "fSCA against the lognormal survival function",
        float(dist.sf(dm)),
        fsca,
        rtol=1e-12,
        reference="fSCA = P(SWE > dm) for the lognormal distribution",
    )
    laugh.check(
        "mean SWE against the numerical partial expectation",
        float(integral),
        swe_mean,
        rtol=1e-10,
        atol=1e-12,
        reference="E[(SWE - dm)+] = integral_{dm}^{inf} (x - dm) f(x) dx",
    )


def test_no_melt_leaves_full_cover(laugh) -> None:
    fsca, swe_mean, *_rest = calc_fsca_and_mean_swe(mu=1.7, dm=0.0, cv=0.5)
    laugh.check(
        "fSCA",
        1.0,
        fsca,
        atol=0.0,
        reference="With dm = 0 the whole grid is snow covered",
    )
    laugh.check(
        "mean SWE",
        1.7,
        swe_mean,
        atol=0.0,
        reference="With dm = 0 the mean is the distribution mean",
    )


def test_erfc_complement_identity(laugh) -> None:
    gaps = []
    for z in (-2.0, -0.5, 0.0, 0.3, 1.7):
        total = calc_swe_frac_area(z) + calc_swe_frac_area(-z)
        gaps.append(abs(total - 1.0))
    laugh.check(
        "largest departure of fSCA(z) + fSCA(-z) from 1",
        0.0,
        float(np.max(gaps)),
        atol=1e-12,
        reference="erfc(x) + erfc(-x) = 2, so the two half-erfc values sum to 1",
    )


def test_tiny_cv_is_a_step_function(laugh) -> None:
    """CV is floored at 0.001. The distribution is then a spike at mu."""
    fsca_low, mean_low, *_rest = calc_fsca_and_mean_swe(mu=1.0, dm=0.5, cv=0.001)
    fsca_high, mean_high, *_rest = calc_fsca_and_mean_swe(mu=1.0, dm=2.0, cv=0.001)
    laugh.check(
        "fSCA when dm < mu",
        1.0,
        fsca_low,
        atol=1e-6,
        reference="A degenerate distribution at mu is fully covered until dm reaches mu",
    )
    laugh.check(
        "mean SWE when dm < mu",
        0.5,
        mean_low,
        atol=1e-6,
        reference="Below the spike, mean SWE = mu - dm",
    )
    laugh.check(
        "fSCA when dm > mu",
        0.0,
        fsca_high,
        atol=1e-6,
        reference="Once dm passes mu, a degenerate distribution is snow free",
    )
    laugh.check(
        "mean SWE when dm > mu",
        0.0,
        mean_high,
        atol=1e-6,
        reference="Snow-free grid mean is 0",
    )


def test_melt_depth_round_trip(laugh) -> None:
    mu, cv, dm = 1.0, 0.5, 0.4
    _fsca, target, *_rest = calc_fsca_and_mean_swe(mu, dm, cv)
    recovered = solve_dm_for_mean_swe(target, mu, cv, dm_guess=0.1)
    _fsca_back, implied, *_rest = calc_fsca_and_mean_swe(mu, recovered, cv)
    laugh.check(
        "recovered melt depth",
        dm,
        recovered,
        atol=1e-8,
        reference="Brent inversion of the partial expectation",
    )
    laugh.check(
        "residual of the inverted mean",
        0.0,
        abs(target - implied),
        atol=1e-8,
        reference="The solver residual should be within Brent's tolerance",
    )


def test_figure_step_and_inversion(laugh) -> None:
    """CV = 0.001 is a step at mu. Brent recovers a prescribed melt depth."""
    mu = 1.0
    melt_depths = np.linspace(0.02, 2.4, 90)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for cv, color, label in ((0.001, "C3", "CV = 0.001"), (0.5, "C0", "CV = 0.5")):
        fsca = [
            calc_fsca_and_mean_swe(mu, float(dm), cv)[0] for dm in melt_depths
        ]
        axes[0].plot(melt_depths, fsca, color=color, label=label)
    axes[0].axvline(mu, color="0.35", ls="--", lw=1.0, label="dm = mu")
    axes[0].set_xlabel("melt depth")
    axes[0].set_ylabel("fSCA")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].set_title("Degenerate cover is a step")
    axes[0].legend(fontsize=7)

    prescribed = np.linspace(0.15, 0.85, 8)
    recovered = []
    for dm in prescribed:
        _fsca, target, *_rest = calc_fsca_and_mean_swe(mu, float(dm), 0.5)
        recovered.append(solve_dm_for_mean_swe(target, mu, 0.5, dm_guess=0.2))
    recovered = np.asarray(recovered)
    axes[1].plot([0.0, 1.0], [0.0, 1.0], "k--", lw=1.0, label="prescribed")
    axes[1].scatter(prescribed, recovered, s=28, color="C0", zorder=3, label="recovered")
    axes[1].set_xlabel("prescribed melt depth")
    axes[1].set_ylabel("recovered melt depth")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Brent inversion, CV = 0.5")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, "ssnowd_step_and_inversion.png")

    below = calc_fsca_and_mean_swe(mu, 0.5, 0.001)[0]
    above = calc_fsca_and_mean_swe(mu, 2.0, 0.001)[0]
    laugh.check(
        "fSCA on either side of mu when CV = 0.001",
        np.array([1.0, 0.0]),
        np.array([below, above]),
        atol=1e-6,
        reference="A degenerate distribution at mu is fully covered until dm reaches mu",
    )
    laugh.check(
        "largest melt-depth recovery error",
        0.0,
        float(np.max(np.abs(recovered - prescribed))),
        atol=1e-8,
        reference="Brent inversion of the partial expectation",
    )


def test_figure_depletion_curve(laugh) -> None:
    mu = 1.0
    melt_depths = np.linspace(0.05, 3.0, 25)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    largest_gap = 0.0
    ax.scatter([], [], s=16, color="0.2", label="code")
    for cv, color in ((0.2, "C0"), (0.5, "C1"), (1.0, "C2")):
        code_mean = []
        code_fsca = []
        analytic_mean = []
        analytic_fsca = []
        for dm in melt_depths:
            fsca, swe_mean, *_rest = calc_fsca_and_mean_swe(mu, float(dm), cv)
            a_fsca, a_mean = _analytic_depletion(mu, float(dm), cv)
            code_fsca.append(fsca)
            code_mean.append(swe_mean / mu)
            analytic_fsca.append(a_fsca)
            analytic_mean.append(a_mean / mu)
            largest_gap = max(largest_gap, abs(fsca - a_fsca), abs(swe_mean - a_mean))
        order = np.argsort(analytic_mean)
        ax.plot(
            np.asarray(analytic_mean)[order],
            np.asarray(analytic_fsca)[order],
            color=color,
            label=f"analytical, CV={cv:g}",
        )
        ax.scatter(
            code_mean,
            code_fsca,
            s=16,
            color=color,
            zorder=3,
        )
    ax.set_xlabel("grid-mean SWE / mu")
    ax.set_ylabel("fSCA")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Lognormal depletion curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "ssnowd_depletion.png")
    laugh.check(
        "largest gap from the lognormal partial expectation",
        0.0,
        largest_gap,
        atol=1e-10,
        reference=_LISTON,
    )
