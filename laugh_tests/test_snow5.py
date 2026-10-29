"""Laugh tests for one step of the 5-parameter snow model.

Slater et al. (2013), Adv. Water Resour. 55:165-177.
doi:10.1016/j.advwatres.2012.07.006

Snow falls when air temperature is below the rain/snow threshold.
Melt is zero at or below the melt threshold, and it cannot exceed the
snow on the ground. The SNOW-17 melt factor is the additive sinusoid
(f_max + f_min) / 2 + (f_max - f_min) / 2 * sin(phase). The code, like
the legacy port, multiplies those terms instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from snowda.core.snow5 import (
    Snow5Parameters,
    compute_melt_factor,
    compute_potential_melt,
    generate_snowfall,
    melt_swe,
    run_snow5_timestep,
)

_SLATER = "Slater et al. (2013): snowfall = precip * guc when T < src, else 0"
_SNOW17 = (
    "SNOW-17: Mf = (f_max + f_min)/2 + (f_max - f_min)/2 * sin(phase), "
    "which equals f when f_max = f_min"
)


def test_precipitation_is_snow_only_below_the_threshold(laugh) -> None:
    snowfall = generate_snowfall(
        precip=[10.0, 10.0, 10.0, 4.0],
        airtemp=[-5.0, 0.0, 0.1, -1.0],
        guc=0.5,
        src=0.0,
    )
    laugh.check(
        "snowfall",
        np.array([5.0, 0.0, 0.0, 2.0]),
        snowfall,
        atol=0.0,
        reference=_SLATER,
    )


def test_no_melt_at_or_below_the_melt_threshold(laugh) -> None:
    potential = compute_potential_melt(
        airtemp=[-5.0, 0.0, 2.0],
        melt_factor=[4.0, 4.0, 4.0],
        t_mlt=0.0,
    )
    laugh.check(
        "potential melt",
        np.array([0.0, 0.0, 8.0]),
        potential,
        atol=0.0,
        reference="Potential melt = melt factor * max(T - t_mlt, 0)",
    )


def test_melt_cannot_exceed_swe(laugh) -> None:
    swe_after, melt = melt_swe(
        swe=[1.0, 5.0, 0.0],
        melt_potential=[10.0, 2.0, 3.0],
    )
    laugh.check(
        "SWE after melt",
        np.array([0.0, 3.0, 0.0]),
        swe_after,
        atol=0.0,
        reference="Melt = min(SWE, potential), so storage cannot go negative",
    )
    laugh.check(
        "melt",
        np.array([1.0, 2.0, 0.0]),
        melt,
        atol=0.0,
        reference="Melt = min(SWE, potential), so storage cannot go negative",
    )


def test_cold_day_stores_all_snowfall(laugh) -> None:
    """T is below both thresholds, so the answer does not depend on the melt factor."""
    params = Snow5Parameters(guc=0.5, src=0.0, f_max=5.0, f_min=1.0, t_mlt=0.0)
    snowfall, melt, _potential, swe_after = run_snow5_timestep(
        pptrate=[10.0],
        airtemp=[-5.0],
        doy=[180.0],
        swe=[20.0],
        params=params,
    )
    laugh.check("snowfall", 5.0, float(snowfall[0]), atol=0.0, reference=_SLATER)
    laugh.check(
        "melt",
        0.0,
        float(melt[0]),
        atol=0.0,
        reference="T < t_mlt, so potential melt is 0",
    )
    laugh.check(
        "SWE",
        25.0,
        float(swe_after[0]),
        atol=0.0,
        reference="SWE_after = SWE + snowfall - melt = 20 + 5 - 0",
    )


def test_timestep_conserves_swe(laugh) -> None:
    params = Snow5Parameters(guc=0.5, src=0.0, f_max=8.0, f_min=1.0, t_mlt=0.0)
    swe = np.array([30.0, 4.0])
    snowfall, melt, _potential, swe_after = run_snow5_timestep(
        pptrate=[8.0, 0.0],
        airtemp=[-3.0, 5.0],
        doy=[180.0, 180.0],
        swe=swe,
        params=params,
    )
    residual = swe_after - (swe + snowfall - melt)
    laugh.check(
        "mass-balance residual",
        np.zeros(2),
        residual,
        atol=0.0,
        reference="SWE_after = SWE + snowfall - melt at each HRU",
    )
    laugh.check(
        "SWE stays non-negative",
        True,
        bool(np.all(swe_after >= 0.0)),
        reference="Melt is capped by the snow on the ground",
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The SNOW-17 melt factor is (f_max+f_min)/2 + (f_max-f_min)/2*sin(phase), "
        "so equal bounds give that constant all year. compute_melt_factor multiplies "
        "mean * sin * diff, as the legacy port does, and equal bounds give zero melt."
    ),
)
def test_equal_melt_bounds_give_a_constant_factor(laugh) -> None:
    day = np.linspace(0.0, 365.0, 366)
    computed = compute_melt_factor(day, f_max=2.0, f_min=2.0)
    laugh.check(
        "minimum melt factor over the year when f_max = f_min = 2",
        2.0,
        float(np.min(computed)),
        atol=1e-12,
        reference=_SNOW17,
    )
