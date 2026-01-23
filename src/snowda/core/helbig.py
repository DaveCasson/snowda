from __future__ import annotations

import numpy as np


def snow_cover_fraction(
    snow_depth_n,
    swe_tmp,
    swe_hist_n,
    swe_min_n,
    swe_max_n,
    snow_depth_hist_n,
    snow_depth_min_n,
    snow_depth_max_n,
    slope_mu_n,
    xi_n,
    Ld_n,
    snow_depth_threshold,
    buffer_length,
):
    swe_buffer = np.empty(buffer_length)
    snow_depth_buffer = np.empty(buffer_length)
    diff_swe_buffer = np.empty(buffer_length - 1)

    sd_snow_depth_1 = np.exp(-1 / (Ld_n / xi_n) ** 2)
    sd_snow_depth_3 = slope_mu_n ** (0.3193 * (Ld_n**0.1034))

    swe_buffer[0] = swe_tmp
    swe_buffer[1:buffer_length] = swe_hist_n[:]
    snow_depth_buffer[0] = snow_depth_n
    snow_depth_buffer[1:buffer_length] = snow_depth_hist_n[:]

    history_update_length = buffer_length - 1
    swe_hist_n[0:history_update_length] = swe_buffer[0:history_update_length]
    snow_depth_hist_n[0:history_update_length] = snow_depth_buffer[0:history_update_length]

    i_abs_max = np.argmax(swe_buffer)
    i_abs_min = np.argmin(swe_buffer)

    for i in range(1, history_update_length, 1):
        diff_swe_buffer[i] = swe_buffer[i + 1] - swe_buffer[i]
        if diff_swe_buffer[i] > 0.5:
            break

    i_recent_min = np.argmin(swe_buffer[0:i])

    snow_depth_min_buffer = snow_depth_buffer[i_abs_min]
    snow_depth_max_buffer = snow_depth_buffer[i_abs_max]
    snow_depth_min_recent = snow_depth_buffer[i_recent_min]

    d_snow_depth = max(snow_depth_n - snow_depth_min_buffer, 0)
    d_snow_depth_max = max(snow_depth_max_buffer - snow_depth_min_buffer, 0)
    d_snow_depth_max = max(d_snow_depth_max, d_snow_depth)
    d_snow_depth_recent = max(snow_depth_n - snow_depth_min_recent, 0)

    if swe_tmp == 0:
        swe_max_n = 0
        swe_min_n = 0
        snow_depth_max_n = 0
        snow_depth_min_n = 0

    if swe_tmp >= swe_max_n:
        swe_max_n = swe_tmp
        swe_min_n = swe_tmp
        snow_depth_max_n = snow_depth_n
        snow_depth_min_n = snow_depth_n

    if swe_tmp < swe_max_n and swe_tmp < swe_min_n:
        swe_min_n = swe_tmp
        snow_depth_min_n = snow_depth_n

    fsnow_season = 0
    fsnow_nsnow = 0
    fsnow_nsnow_recent = 0

    sd_snow_depth_2 = snow_depth_max_n ** (0.5330 * (Ld_n**0.0389))
    sd_snow_depth_0 = sd_snow_depth_1 * sd_snow_depth_2 * sd_snow_depth_3
    if not slope_mu_n > np.finfo("float").eps:
        sd_snow_depth_0 = snow_depth_max_n**0.84

    if snow_depth_max_n > 0:
        fsnow_season = np.tanh(1.3 * snow_depth_min_n / sd_snow_depth_0)

    if snow_depth_max_n > 0:
        coeff_vari = sd_snow_depth_0 / snow_depth_max_n
    else:
        coeff_vari = np.nan

    sd_snow_depth_0_dhs = d_snow_depth_max**0.84
    if d_snow_depth_max > 0:
        fsnow_nsnow = np.tanh(1.3 * d_snow_depth / sd_snow_depth_0_dhs)

    fsnow_nsnow_14days = fsnow_nsnow
    _ = fsnow_nsnow_14days

    sd_snow_depth_0_dhs_recent = d_snow_depth_recent**0.84
    if d_snow_depth_recent > 0:
        fsnow_nsnow_recent = np.tanh(1.3 * d_snow_depth_recent / sd_snow_depth_0_dhs_recent)

    fsnow_nsnow = max(fsnow_nsnow, fsnow_nsnow_recent)

    if snow_depth_min_n < snow_depth_threshold:
        fsnow_season = 0
    if d_snow_depth < snow_depth_threshold:
        fsnow_nsnow = 0

    fsnow_n = max(fsnow_season, fsnow_nsnow)

    if snow_depth_n < np.finfo("float").eps:
        fsnow_n = 0
    else:
        fsnow_n = min(fsnow_n, 1)

    return (
        fsnow_n,
        snow_depth_min_n,
        snow_depth_max_n,
        snow_depth_hist_n,
        swe_min_n,
        swe_max_n,
        swe_hist_n,
        coeff_vari,
    )
