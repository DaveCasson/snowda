from __future__ import annotations

import numpy as np

from snowda.core.debeer import linear_reversion
from snowda.core.helbig import snow_cover_fraction
from snowda.core.snow5 import compute_melt_factor, phase_shift_day_of_year
from snowda.schaake_shuffle.core import (
    apply_permutation,
    schaake_shuffle,
    shuffle_source,
)


def test_melt_factor_runs_from_f_min_in_december_to_f_max_in_june():
    # 21 December (day 355) and 21 June (day 172, 173 in the 366-day cycle).
    days = phase_shift_day_of_year(np.array([355.0, 173.0]))
    winter, summer = compute_melt_factor(days, f_max=6.0, f_min=2.0)
    assert abs(winter - 2.0) < 1e-2
    assert abs(summer - 6.0) < 1e-2


def test_linear_reversion_ramps_from_full_cover_to_the_earlier_fsca():
    # 10 units of new snow, alpha 0.25: full cover until 7.5 has melted.
    at_threshold, _ = linear_reversion(10.0, 7.5, 0.4, 100.0, 0.25)
    halfway, _ = linear_reversion(10.0, 8.75, 0.4, 100.0, 0.25)
    melted, _ = linear_reversion(10.0, 10.0, 0.4, 100.0, 0.25)
    assert at_threshold == 1
    assert abs(halfway - 0.7) < 1e-12
    assert melted == 0.4


def test_helbig_accepts_a_two_step_buffer():
    fsca = snow_cover_fraction(
        0.5, 150.0, [0.0], 0.0, 0.0, [0.0], 0.0, 0.0, 0.2, 100.0, 1000.0, 0.01, 2
    )[0]
    assert 0.0 <= fsca <= 1.0


def test_secondary_variable_stays_with_its_shuffled_member():
    swe = np.array([[30.0], [10.0], [20.0]])
    depth = swe * 10.0
    template = np.array([[2.0], [3.0], [1.0]])
    shuffled, rank_index = schaake_shuffle(swe, template)
    moved = apply_permutation(depth, shuffle_source(swe, rank_index))
    assert np.array_equal(moved, shuffled * 10.0)
