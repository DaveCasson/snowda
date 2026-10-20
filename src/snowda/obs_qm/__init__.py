"""Observation-to-model empirical quantile mapping for point SWE.

This package is observation **preprocessing**. It estimates, for each
station, seasonal empirical distributions of observed and colocated
modeled SWE and maps each observation onto model-equivalent SWE:

    y_model_equiv = F_mod^{-1}(F_obs(y_obs))

The model state is not bias-corrected. Data-assimilation code does not
import this package.

Method (version 1)
------------------
- Station-specific mappings (no spatial pooling).
- Empirical CDFs with Hazen plotting positions p_i = (i - 0.5) / n.
- Seasonal moving window of ±window_days in circular day-of-year space.
- Leave-one-water-year-out cross-validation by default.
- Zero observed SWE maps to zero; positive SWE uses samples where both
  observed and modeled SWE are positive.
- Linear interpolation; percentiles outside the calibration range are
  clipped (no tail extrapolation).
- Midnight model SWE only (no daily mean, no interpolation).
"""

from __future__ import annotations

from snowda.obs_qm.diagnostics import (
    mapping_curve_arrays,
    plot_station_comparison_timeseries,
    plot_station_cross_validation,
    plot_station_diagnostics,
)
from snowda.obs_qm.io import (
    attach_mapped_variables,
    read_midnight_model_swe,
    read_swe_observations,
    write_mapped_observations,
)
from snowda.obs_qm.mapping import (
    empirical_cdf_quantile,
    empirical_inverse_cdf,
    empirical_quantiles,
    hazen_plotting_positions,
    quantile_map_obs_to_model,
)
from snowda.obs_qm.pairing import (
    align_model_to_obs_time,
    extract_midnight_swe,
    pair_obs_model,
)
from snowda.obs_qm.preprocess import (
    FLAG_INSUFFICIENT_SAMPLES,
    FLAG_MAPPED,
    FLAG_MISSING_MODEL,
    FLAG_MISSING_OBS,
    FLAG_ZERO_OBS,
    combine_stations,
    preprocess_all_stations,
    preprocess_station,
    station_record_table,
    stations_to_plot,
)
from snowda.obs_qm.seasonal import (
    calibration_sample_count_by_water_year_day,
    circular_doy_distance,
    get_seasonal_samples,
    in_seasonal_window,
    observation_year_coverage,
    water_year,
    water_year_day,
)

__all__ = [
    "empirical_cdf_quantile",
    "empirical_inverse_cdf",
    "empirical_quantiles",
    "hazen_plotting_positions",
    "quantile_map_obs_to_model",
    "circular_doy_distance",
    "get_seasonal_samples",
    "in_seasonal_window",
    "water_year",
    "extract_midnight_swe",
    "align_model_to_obs_time",
    "pair_obs_model",
    "preprocess_station",
    "preprocess_all_stations",
    "combine_stations",
    "station_record_table",
    "stations_to_plot",
    "observation_year_coverage",
    "water_year_day",
    "calibration_sample_count_by_water_year_day",
    "FLAG_MAPPED",
    "FLAG_ZERO_OBS",
    "FLAG_INSUFFICIENT_SAMPLES",
    "FLAG_MISSING_OBS",
    "FLAG_MISSING_MODEL",
    "read_swe_observations",
    "read_midnight_model_swe",
    "attach_mapped_variables",
    "write_mapped_observations",
    "plot_station_diagnostics",
    "plot_station_comparison_timeseries",
    "plot_station_cross_validation",
    "mapping_curve_arrays",
]
