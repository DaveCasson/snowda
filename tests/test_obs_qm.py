"""Unit tests for observation-to-model empirical SWE quantile mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from snowda.obs_qm.mapping import (
    empirical_cdf_quantile,
    empirical_inverse_cdf,
    empirical_quantiles,
    hazen_plotting_positions,
    quantile_map_obs_to_model,
)
from snowda.obs_qm.pairing import extract_midnight_swe, pair_obs_model
from snowda.obs_qm.preprocess import (
    FLAG_INSUFFICIENT_SAMPLES,
    FLAG_MAPPED,
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
    get_seasonal_samples,
    in_seasonal_window,
    observation_year_coverage,
    water_year,
    water_year_day,
)


def test_hazen_plotting_positions():
    p = hazen_plotting_positions(4)
    np.testing.assert_allclose(p, np.array([0.125, 0.375, 0.625, 0.875]))


def test_identical_distributions_map_to_identity():
    samples = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    for y in samples:
        y_star, q = quantile_map_obs_to_model(y, samples, samples)
        assert y_star == pytest.approx(y)
        assert 0.0 < q < 1.0


def test_identical_distributions_interior_interpolation():
    samples = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    # Interior point between unique sample values should stay on the 1:1 line.
    y_star, _q = quantile_map_obs_to_model(15.0, samples, samples)
    assert y_star == pytest.approx(15.0)


def test_simple_scale_difference():
    obs_samples = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    model_samples = 0.5 * obs_samples
    for y in obs_samples:
        y_star, _q = quantile_map_obs_to_model(y, obs_samples, model_samples)
        assert y_star == pytest.approx(0.5 * y)


def test_nonlinear_mapping_preserves_rank():
    obs_samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    model_samples = obs_samples**2
    targets = np.array([1.5, 3.2, 6.8, 8.0])
    mapped = np.array(
        [quantile_map_obs_to_model(y, obs_samples, model_samples)[0] for y in targets]
    )
    assert np.all(np.diff(mapped) > 0)
    # Rank order of targets is preserved in the mapped values.
    assert np.array_equal(np.argsort(targets), np.argsort(mapped))


def test_observation_outside_range_clips_percentile():
    obs_samples = np.array([10.0, 20.0, 30.0, 40.0])
    model_samples = np.array([1.0, 2.0, 3.0, 4.0])
    x, p, n = empirical_quantiles(obs_samples)
    assert n == 4
    q_low = empirical_cdf_quantile(0.0, obs_samples)
    q_high = empirical_cdf_quantile(100.0, obs_samples)
    assert q_low == pytest.approx(p[0])
    assert q_high == pytest.approx(p[-1])
    assert q_low == pytest.approx(0.5 / n)
    assert q_high == pytest.approx((n - 0.5) / n)

    y_low, _ = quantile_map_obs_to_model(0.0, obs_samples, model_samples)
    y_high, _ = quantile_map_obs_to_model(100.0, obs_samples, model_samples)
    assert y_low == pytest.approx(model_samples.min())
    assert y_high == pytest.approx(model_samples.max())
    # Inverse CDF also clips rather than extrapolating.
    assert empirical_inverse_cdf(-0.5, model_samples) == pytest.approx(model_samples.min())
    assert empirical_inverse_cdf(1.5, model_samples) == pytest.approx(model_samples.max())


def test_zero_obs_maps_to_zero():
    times = pd.date_range("2001-01-01", periods=80, freq="D")
    obs = np.linspace(5.0, 40.0, times.size)
    obs[10] = 0.0
    model = obs * 0.8
    obs_da = xr.DataArray(obs, dims=("time",), coords={"time": times})
    model_da = xr.DataArray(model, dims=("time",), coords={"time": times})
    mapped = preprocess_station(
        obs_da,
        model_da,
        window_days=20,
        min_samples=5,
        cross_validate=False,
    )
    assert mapped["mapping_flag"].values[10] == FLAG_ZERO_OBS
    assert mapped["swe_model_equivalent"].values[10] == pytest.approx(0.0)
    assert np.isnan(mapped["quantile_obs"].values[10])


def test_insufficient_historical_samples_returns_missing_and_flag():
    times = pd.date_range("2001-01-15", periods=8, freq="D")
    obs = np.arange(8, dtype=float) + 1.0
    model = obs * 2.0
    obs_da = xr.DataArray(obs, dims=("time",), coords={"time": times})
    model_da = xr.DataArray(model, dims=("time",), coords={"time": times})
    mapped = preprocess_station(
        obs_da,
        model_da,
        window_days=5,
        min_samples=20,
        cross_validate=False,
    )
    assert np.all(mapped["mapping_flag"].values == FLAG_INSUFFICIENT_SAMPLES)
    assert np.all(np.isnan(mapped["swe_model_equivalent"].values))
    assert np.all(mapped["n_samples_mapping"].values < 20)


def test_missing_obs_flag():
    times = pd.date_range("2001-02-01", periods=40, freq="D")
    obs = np.linspace(5.0, 25.0, times.size)
    obs[3] = np.nan
    model = obs.copy()
    model[3] = 10.0
    obs_da = xr.DataArray(obs, dims=("time",), coords={"time": times})
    model_da = xr.DataArray(model, dims=("time",), coords={"time": times})
    mapped = preprocess_station(
        obs_da,
        model_da,
        window_days=15,
        min_samples=5,
        cross_validate=False,
    )
    assert mapped["mapping_flag"].values[3] == FLAG_MISSING_OBS
    assert np.isnan(mapped["swe_model_equivalent"].values[3])


def test_leave_one_year_out_excludes_target_year_from_calibration():
    times = pd.to_datetime(
        [
            "2014-01-15",
            "2015-01-15",
            "2016-01-15",
            "2017-01-15",
        ]
    )
    obs = np.array([10.0, 20.0, 999.0, 40.0])
    model = np.array([10.0, 20.0, 999.0, 40.0])
    target = times[2]
    obs_s, mod_s = get_seasonal_samples(
        times,
        obs,
        model,
        target,
        window_days=5,
        exclude_water_year=2016,
    )
    assert 999.0 not in obs_s
    assert 999.0 not in mod_s
    np.testing.assert_array_equal(obs_s, np.array([10.0, 20.0, 40.0]))

    included = get_seasonal_samples(
        times, obs, model, target, window_days=5, exclude_water_year=None
    )[0]
    assert 999.0 in included


def test_leave_one_year_out_preprocess_does_not_use_target_year_sentinel():
    # Daily mid-winter series across three water years, with a unique
    # sentinel only in WY 2016 that would dominate the mapping if leaked.
    times = pd.date_range("2014-01-01", "2016-03-01", freq="D")
    obs = np.full(times.size, 20.0)
    model = np.full(times.size, 10.0)
    wy = water_year(times)
    obs[wy == 2016] = 500.0
    model[wy == 2016] = 500.0
    # Keep a few positive samples every year inside a January window.
    obs_da = xr.DataArray(obs, dims=("time",), coords={"time": times})
    model_da = xr.DataArray(model, dims=("time",), coords={"time": times})
    mapped = preprocess_station(
        obs_da,
        model_da,
        window_days=20,
        min_samples=10,
        cross_validate=True,
    )
    target = np.flatnonzero(times == pd.Timestamp("2016-01-15"))[0]
    assert mapped["mapping_flag"].values[target] == FLAG_MAPPED
    # Without leakage the 2016 observation (500) maps through the 20→10
    # climatology of the other years, not through the 500 sentinel.
    assert mapped["swe_model_equivalent"].values[target] == pytest.approx(10.0)
    obs_cal, _mod_cal = get_seasonal_samples(
        times,
        obs,
        model,
        times[target],
        window_days=20,
        exclude_water_year=2016,
    )
    assert 500.0 not in obs_cal
    assert mapped["n_samples_mapping"].values[target] == obs_cal.size


def test_seasonal_window_spans_water_year_boundary():
    times = pd.to_datetime(
        [
            "2015-09-15",
            "2015-09-20",
            "2015-10-01",
            "2015-10-20",
            "2015-11-15",
            "2016-03-01",
        ]
    )
    target = pd.Timestamp("2015-10-01")
    mask = in_seasonal_window(times, target, window_days=30)
    selected = times[mask]
    assert pd.Timestamp("2015-09-15") in selected
    assert pd.Timestamp("2015-09-20") in selected
    assert pd.Timestamp("2015-10-01") in selected
    assert pd.Timestamp("2015-10-20") in selected
    assert pd.Timestamp("2015-11-15") not in selected
    assert pd.Timestamp("2016-03-01") not in selected

    obs = np.arange(times.size, dtype=float) + 1.0
    model = obs.copy()
    obs_s, _mod_s = get_seasonal_samples(
        times, obs, model, target, window_days=30
    )
    assert obs_s.size == 4


def test_water_year_october_start():
    times = pd.to_datetime(["2015-09-30", "2015-10-01"])
    wy = water_year(times, start_month=10)
    np.testing.assert_array_equal(wy, np.array([2015, 2016]))


def test_midnight_extract_skips_hourly_and_missing_midnight():
    times = pd.to_datetime(
        [
            "2000-01-15 00:00:00",
            "2000-01-15 01:00:00",
            "2000-01-16 01:00:00",
            "2000-01-17 00:00:00",
            "2000-01-17 00:59:59.999986560",
        ],
        format="mixed",
    )
    values = np.array([10.0, 99.0, 88.0, 20.0, 77.0])
    da = xr.DataArray(values, dims=("time",), coords={"time": times}, name="scalarSWE")
    midnight = extract_midnight_swe(da)
    midnight_times = pd.DatetimeIndex(midnight["time"].values)
    np.testing.assert_array_equal(
        midnight_times, pd.to_datetime(["2000-01-15", "2000-01-17"])
    )
    np.testing.assert_allclose(midnight.values, np.array([10.0, 20.0]))
    assert 99.0 not in midnight.values
    assert 88.0 not in midnight.values
    assert 77.0 not in midnight.values


def test_pair_obs_model_does_not_fill_missing_midnight():
    obs_times = pd.to_datetime(["2000-01-15", "2000-01-16", "2000-01-17"])
    obs = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("time",),
        coords={"time": obs_times},
        name="snw",
    )
    model_times = pd.to_datetime(["2000-01-15 00:00:00", "2000-01-16 01:00:00"])
    model = xr.DataArray(
        np.array([10.0, 99.0]),
        dims=("time",),
        coords={"time": model_times},
        name="scalarSWE",
    )
    midnight = extract_midnight_swe(model)
    paired = pair_obs_model(obs, midnight)
    assert paired["paired"].values.tolist() == [True, False, False]
    assert paired["swe_model"].values[0] == pytest.approx(10.0)
    assert np.isnan(paired["swe_model"].values[1])
    assert np.isnan(paired["swe_model"].values[2])
    assert 99.0 not in paired["swe_model"].values


def test_duplicate_midnight_raises():
    times = pd.to_datetime(["2000-01-15 00:00:00", "2000-01-15 00:00:30"])
    da = xr.DataArray([1.0, 2.0], dims=("time",), coords={"time": times})
    with pytest.raises(ValueError, match="duplicate midnight"):
        extract_midnight_swe(da)


def test_tie_collapse_uses_mean_hazen_position():
    samples = np.array([5.0, 5.0, 10.0, 10.0])
    x, p, n = empirical_quantiles(samples)
    np.testing.assert_allclose(x, np.array([5.0, 10.0]))
    # Ranks 1-2 share 5.0; ranks 3-4 share 10.0. Mean Hazen positions:
    # (0.125+0.375)/2 = 0.25, (0.625+0.875)/2 = 0.75.
    np.testing.assert_allclose(p, np.array([0.25, 0.75]))
    assert n == 4
    assert empirical_cdf_quantile(5.0, samples) == pytest.approx(0.25)


def test_preprocess_all_stations_are_independent():
    times = pd.date_range("2001-01-01", periods=60, freq="D")
    obs = np.stack(
        [
            np.linspace(5.0, 25.0, times.size),
            np.linspace(25.0, 5.0, times.size),
        ]
    )
    model = obs[0] * 0.5
    obs_da = xr.DataArray(
        obs,
        dims=("station_id", "time"),
        coords={"station_id": ["a", "b"], "time": times},
    )
    model_da = xr.DataArray(model, dims=("time",), coords={"time": times})
    mapped = preprocess_all_stations(
        obs_da,
        model_da,
        window_days=20,
        min_samples=10,
        cross_validate=False,
    )
    assert list(mapped.station_id.values) == ["a", "b"]
    a = mapped.sel(station_id="a")
    b = mapped.sel(station_id="b")
    assert int((a["mapping_flag"].values == FLAG_MAPPED).sum()) > 0
    assert int((b["mapping_flag"].values == FLAG_MAPPED).sum()) > 0
    assert not np.allclose(a["swe_obs"].values, b["swe_obs"].values)
    assert float(mapped["n_paired"].sel(station_id="a")) == float(
        mapped["n_paired"].sel(station_id="b")
    )
    assert mapped.attrs["station_pooling"] == "none"
    assert mapped.attrs["mapping_direction"] == "observation_to_model"


def test_water_year_day_starts_on_october_1():
    times = pd.to_datetime(["2015-10-01", "2016-01-01", "2016-09-30"])
    wyd = water_year_day(times, start_month=10)
    np.testing.assert_array_equal(wyd, np.array([1, 93, 366]))


def test_observation_year_coverage_counts_unique_water_years():
    times = pd.to_datetime(
        ["2014-11-01", "2015-02-01", "2015-11-01", "2017-03-01"]
    )
    obs = np.array([10.0, 20.0, np.nan, 30.0])
    stats = observation_year_coverage(times, obs, start_month=10)
    # Finite on 2014-11 (WY2015), 2015-02 (WY2015), 2017-03 (WY2017).
    assert stats["n_calendar_years"] == 3
    assert stats["n_water_years"] == 2
    assert stats["n_water_years_positive"] == 2
    assert stats["first_year"] == 2014
    assert stats["last_year"] == 2017


def test_calibration_sample_count_peaks_in_winter_not_summer():
    times = pd.to_datetime(
        ["2014-01-15", "2015-01-15", "2016-01-15", "2014-07-15"]
    )
    obs = np.array([10.0, 20.0, 30.0, np.nan])
    model = np.array([8.0, 16.0, 24.0, np.nan])
    wyd, counts, ref = calibration_sample_count_by_water_year_day(
        times, obs, model, window_days=20
    )
    jan = counts[ref.month == 1]
    jul = counts[ref.month == 7]
    assert jan.max() == 3
    assert jul.max() == 0
    # Winter (Oct-May) must hold more samples than calendar summer (Jun-Sep).
    winter = counts[ref.month.isin([10, 11, 12, 1, 2, 3, 4, 5])]
    summer = counts[ref.month.isin([6, 7, 8, 9])]
    assert winter.max() > summer.max()


def test_station_record_table_year_screen():
    times = pd.date_range("2015-01-01", periods=800, freq="D")
    obs = np.full((2, times.size), np.nan)
    # Station a: two winter observations in different water years.
    obs[0, 0] = 10.0  # 2015-01-01 WY2015
    obs[0, 400] = 12.0  # ~2016-02 WY2016
    # Station b: one water year only.
    obs[1, 0] = 15.0
    obs_da = xr.DataArray(
        obs,
        dims=("station_id", "time"),
        coords={"station_id": ["long", "short"], "time": times},
    )
    table = station_record_table(obs_da, min_years=2)
    assert bool(table.set_index("station_id").loc["long", "passes_year_screen"])
    assert not bool(table.set_index("station_id").loc["short", "passes_year_screen"])


def test_combine_stations_mean_on_overlap_keeps_union():
    times = pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"])
    ds = xr.Dataset(
        {
            "snw": (
                ("time", "station_id"),
                np.array(
                    [
                        [10.0, np.nan, 5.0],
                        [20.0, np.nan, 6.0],
                        [30.0, 50.0, 7.0],
                        [np.nan, 40.0, 8.0],
                    ]
                ),
            ),
            "lat": ("station_id", np.array([60.0, 62.0, 50.0])),
            "lon": ("station_id", np.array([-135.0, -137.0, -140.0])),
            "elevation": ("station_id", np.array([900.0, 1000.0, 500.0])),
            "station_name": ("station_id", np.array(["A", "B", "C"])),
        },
        coords={"time": times, "station_id": ["SC02A", "SC02B", "M5"]},
    )
    out = combine_stations(
        ds,
        ["SC02A", "SC02B"],
        new_id="SC02",
        new_name="TWIN CREEKS combined",
    )
    ids = [str(s) for s in out.station_id.values]
    assert "SC02A" not in ids
    assert "SC02B" not in ids
    assert ids.count("SC02") == 1
    assert "M5" in ids

    combined = out.sel(station_id="SC02")
    np.testing.assert_allclose(combined["snw"].values, np.array([10.0, 20.0, 40.0, 40.0]))
    assert str(combined["station_name"].values) == "TWIN CREEKS combined"
    assert float(combined["lat"]) == pytest.approx(61.0)
    assert float(combined["lon"]) == pytest.approx(-136.0)
    assert float(combined["elevation"]) == pytest.approx(950.0)
    np.testing.assert_allclose(
        out.sel(station_id="M5")["snw"].values, np.array([5.0, 6.0, 7.0, 8.0])
    )
    assert out.attrs["combine_overlap"] == "arithmetic_mean"
    assert "SC02A" in out.attrs["combined_from"]


def test_stations_to_plot_skips_screened_by_default():
    mapped = xr.Dataset(
        {"passes_year_screen": ("station_id", np.array([True, False, True]))},
        coords={"station_id": ["SC02", "M5", "M7"]},
    )
    assert [str(s) for s in stations_to_plot(mapped)] == ["SC02", "M7"]
    assert [str(s) for s in stations_to_plot(mapped, include_screened=True)] == [
        "SC02",
        "M5",
        "M7",
    ]
