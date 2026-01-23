from __future__ import annotations

import numpy as np
from scipy import optimize, special

from snowda.utils.logging import get_logger

log = get_logger(__name__)


def calc_lambda(zeta: float, mu: float) -> float:
    """
    Compute the lognormal location parameter (lambda).

    Parameters
    ----------
    zeta : float
        Lognormal shape parameter.
    mu : float
        Distribution scale parameter (reference mean SWE). Must be positive.

    Returns
    -------
    float
        Lognormal location parameter.
    """
    mu = float(mu)
    if mu <= 0.0:
        # Caller should normally guard this; keep safe to avoid log(0) warnings.
        return float("-inf")
    return float(np.log(mu) - 0.5 * zeta**2)


def calc_zeta(cv: float) -> float:
    """
    Compute the lognormal shape parameter (zeta) from coefficient of variation (CV).

    CV is bounded below to avoid singular behavior as CV approaches zero.
    """
    cv = float(cv)
    if not np.isfinite(cv):
        cv = 0.001
    cv = max(cv, 0.001)
    return float(np.sqrt(np.log(1.0 + cv**2)))


def calc_z(depth: float, lambda_: float, zeta: float) -> float:
    """
    Transform positive depth to standard normal space.

    Parameters
    ----------
    depth : float
        Depth at which distribution is evaluated. Must be > 0.
    lambda_ : float
        Lognormal location parameter.
    zeta : float
        Lognormal shape parameter.

    Returns
    -------
    float
        Standard-normal variate.

    Raises
    ------
    ValueError
        If depth <= 0.
    """
    depth = float(depth)
    if depth <= 0.0:
        raise ValueError(f"calc_z requires depth > 0, got {depth}")
    return float((np.log(depth) - lambda_) / zeta)


def calc_swe_frac_area(z: float) -> float:
    """Compute fractional snow-covered area (fSCA) from z."""
    return float(0.5 * special.erfc(z / np.sqrt(2.0)))


def calc_mean_swe(mu: float, dm: float, zeta: float, z: float) -> float:
    """
    Mean SWE implied by (mu, zeta) after applying cumulative melt depth dm.
    """
    fsca = calc_swe_frac_area(z)
    return float(0.5 * mu * special.erfc((z - zeta) / np.sqrt(2.0)) - dm * fsca)


def calc_snowmelt_residual(dm: float, swe_mean_target: float, mu: float, zeta: float, lambda_: float) -> float:
    """
    Residual for melt-depth inversion: target_mean - implied_mean(dm).

    dm must be > 0.
    """
    z = calc_z(dm, lambda_, zeta)
    return float(swe_mean_target - calc_mean_swe(mu, dm, zeta, z))


def solve_dm_for_mean_swe(swe_mean_target: float, mu: float, cv: float, dm_guess: float) -> float:
    """
    Solve for cumulative melt depth (dm) that yields a desired grid-mean SWE.

    Uses a bracketed root finder (Brent) on dm > 0 to avoid invalid log() calls and
    overflow issues common with unconstrained Newton steps.
    """
    mu = float(mu)
    if mu <= 0.0:
        return 0.0

    swe_mean_target = float(swe_mean_target)
    if not np.isfinite(swe_mean_target):
        return float(max(dm_guess, 0.0))

    # Physical bounds for the target mean SWE
    swe_mean_target = float(np.clip(swe_mean_target, 0.0, mu))

    # If target is full snow, dm is effectively zero
    if swe_mean_target >= mu:
        return 0.0

    # If target is essentially snow-free, dm is "very large" (we return a large dm)
    if swe_mean_target <= 0.0:
        return float(max(dm_guess, 0.0))

    zeta = calc_zeta(cv)
    if zeta == 0.0:
        zeta = 1e-6
    lambda_ = calc_lambda(zeta, mu)

    # Residual has known signs:
    # dm -> 0+ : implied_mean -> mu => residual = target - mu < 0
    # dm large : implied_mean -> 0  => residual = target - 0 > 0
    # So a sign change exists on (0, +inf) for target in (0, mu).

    dm_lo = 1e-12
    f_lo = calc_snowmelt_residual(dm_lo, swe_mean_target, mu, zeta, lambda_)

    # Start upper bound from guess or mu-scale and expand until sign change.
    dm_hi = max(float(dm_guess), 1e-6, 0.1 * mu)
    max_expand = 80  # enough to reach extremely large dm without overflow in logs

    f_hi = None
    for _ in range(max_expand):
        try:
            f_hi = calc_snowmelt_residual(dm_hi, swe_mean_target, mu, zeta, lambda_)
        except (ValueError, FloatingPointError, OverflowError):
            # If dm_hi became invalid (shouldn't for log), expand anyway
            f_hi = np.nan

        if np.isfinite(f_hi) and (f_lo * f_hi) <= 0.0:
            break
        dm_hi *= 2.0

    if f_hi is None or not np.isfinite(f_hi) or (f_lo * f_hi) > 0.0:
        log.debug(
            "Could not bracket melt-depth root (mu=%s, target=%s, dm_guess=%s). Using previous dm.",
            mu,
            swe_mean_target,
            dm_guess,
        )
        return float(max(dm_guess, 0.0))

    try:
        sol = optimize.root_scalar(
            calc_snowmelt_residual,
            bracket=(dm_lo, dm_hi),
            args=(swe_mean_target, mu, zeta, lambda_),
            method="brentq",
            maxiter=200,
        )
        if sol.converged:
            return float(max(sol.root, 0.0))
        log.debug("Brent solver did not converge; using previous dm.")
        return float(max(dm_guess, 0.0))
    except (ValueError, RuntimeError, OverflowError) as exc:
        log.debug("Melt-depth solver failed: %s. Using previous dm.", exc)
        return float(max(dm_guess, 0.0))


def calc_fsca_and_mean_swe(mu: float, dm: float, cv: float) -> tuple[float, float, float, float, float]:
    """
    Compute (fSCA, mean SWE, z, lambda, zeta) implied by (mu, dm, cv).
    """
    mu = float(mu)
    dm = float(dm)

    zeta = calc_zeta(cv)
    if zeta == 0.0:
        zeta = 1e-6

    if mu <= 0.0:
        return 0.0, 0.0, np.nan, np.nan, float(zeta)

    lambda_ = calc_lambda(zeta, mu)

    if dm <= 0.0:
        return 1.0, float(mu), np.nan, float(lambda_), float(zeta)

    z = calc_z(dm, lambda_, zeta)
    fsca = calc_swe_frac_area(z)

    swe_mean = (
        0.5 * np.exp(lambda_ + 0.5 * zeta**2) * special.erfc((z - zeta) / np.sqrt(2.0))
        - dm * fsca
    )
    return float(fsca), float(swe_mean), float(z), float(lambda_), float(zeta)


def ssnowd(
    snowfall: float,
    melt_rate: float,
    swe_total_acc: float,
    swe_total_melt: float,
    swe_input: float,
    residual_new_snow: float,
    cv: float,
    *,
    use_model_swe: bool = True,
) -> tuple[float, float, float, float, float, float]:
    """
    Update fractional snow-covered area (fSCA) and mean SWE using a lognormal depletion curve approach.

    State variables
    --------------
    swe_total_acc : float
        Reference accumulation scale (mu) for the subgrid SWE distribution.
    swe_total_melt : float
        Cumulative melt depth (dm) applied uniformly to the distribution.

    Parameters
    ----------
    snowfall : float
        New snowfall over the timestep (SWE units).
    melt_rate : float
        Melt over the timestep (SWE units).
    swe_input : float
        Target grid-mean SWE. If use_model_swe is True, dm is solved such that mean SWE matches this value.
    cv : float
        Coefficient of variation for subgrid SWE.

    Returns
    -------
    tuple
        (fsca, swe_mean, mu, dm, z, lambda)
    """
    _ = residual_new_snow

    if 0.0 < snowfall < 1.0e-7:
        snowfall = 0.0

    mu = float(max(swe_total_acc, 0.0))
    dm = float(max(swe_total_melt, 0.0))

    if snowfall > 0.0:
        mu += float(snowfall)

    if melt_rate < 0.0:
        log.debug("Negative melt_rate encountered: %s", melt_rate)

    if melt_rate > 0.0:
        dm += float(melt_rate)

    if mu <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, np.nan, np.nan

    if use_model_swe:
        dm = solve_dm_for_mean_swe(swe_input, mu, cv, dm_guess=dm)

    fsca, swe_mean, z, lambda_, _zeta = calc_fsca_and_mean_swe(mu, dm, cv)
    swe_mean = float(np.clip(swe_mean, 0.0, mu))

    return float(fsca), float(swe_mean), float(mu), float(dm), float(z), float(lambda_)