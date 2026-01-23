from __future__ import annotations

import numpy as np
from scipy import special


def calc_std_y(cv: float) -> float:
    return np.sqrt(np.log(1 + cv**2))


def calc_mean_y(mean_swe: float, cv: float) -> float:
    return np.log(mean_swe) - 0.5 * (np.log(1 + cv**2))


def calc_exceedance_prop(k_y: float) -> float:
    return 0.5 * special.erfc(k_y / np.sqrt(2))


def calculate_k_min(initial_swe: float, current_swe: float, cv: float) -> float:
    swe_ratio = current_swe / initial_swe
    return -swe_ratio / cv


def back_calculate_k_y(k: float, cv: float, std_y: float) -> float:
    return np.log(1 + k * cv) / std_y + std_y / 2


def run_debeer_simple(premelt_swe: float, swe_melt: float, cv: float) -> tuple[float, float, float]:
    std_y = calc_std_y(cv)
    current_swe = premelt_swe - swe_melt
    k_min = calculate_k_min(premelt_swe, current_swe, cv)
    k_y = back_calculate_k_y(k_min, cv, std_y)
    p = calc_exceedance_prop(k_y)
    return k_min, k_y, p


def linear_reversion(
    new_snowfall: float, melt: float, original_fsca: float, snowfall_total: float, alpha: float
) -> tuple[float, float]:
    melt_threshold = (1 - alpha) * new_snowfall

    if np.greater_equal(melt, new_snowfall):
        new_fsca = original_fsca
        residual_new_snow = 0
    elif np.less_equal(melt, melt_threshold):
        new_fsca = 1
        residual_new_snow = new_snowfall - melt
    else:
        residual_melt = melt - melt_threshold
        slope = (1 - original_fsca) / (alpha * new_snowfall)
        y_intercept = original_fsca - slope * snowfall_total
        new_fsca = slope * (snowfall_total - residual_melt) + y_intercept
        residual_new_snow = new_snowfall - melt

    return new_fsca, residual_new_snow


def debeer(
    snowfall: float,
    melt: float,
    snowfall_total: float,
    melt_total: float,
    fsca: float,
    residual_new_snow: float,
    cv: float,
    *,
    alpha: float = 0.25,
    new_snowfall_threshold: float = 2,
) -> tuple[float, float, float, float]:
    snowfall_total = snowfall_total + snowfall
    melt_total = melt_total + melt

    if snowfall_total <= 0:
        if snowfall <= 0:
            fsca = 0
            residual_new_snow = 0
        else:
            if snowfall > melt:
                fsca = 1
                residual_new_snow = 0
            else:
                fsca = 0
                residual_new_snow = 0
    else:
        if melt_total > 0:
            _k_min, _k_y, fsca = run_debeer_simple(snowfall_total, melt_total, cv)
            if snowfall > new_snowfall_threshold:
                new_fsca, residual_new_snow = linear_reversion(
                    snowfall, melt, fsca, snowfall_total, alpha
                )
                fsca = max(new_fsca, fsca)
        else:
            fsca = 1
            residual_new_snow = 0

    return fsca, snowfall_total, melt_total, residual_new_snow
