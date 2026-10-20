from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from snowda.adapters.scores import (
    brier_score_for_ensemble,
    crps_for_ensemble,
    false_alarm_ratio,
    probability_of_detection,
    rank_histogram,
    reliability_diagram,
    roc_diagram,
    skill_score,
)
from snowda.io.esp_streamflow import (
    EspPairedForecast,
    climatology_window_peaks,
    group_esp_trace_files,
    group_forecasts_by_init_month,
    init_month_key,
    observational_climatology_ensemble,
    parse_state_date_key,
    peak_over_horizon,
    read_return_period_thresholds,
    select_return_period_thresholds,
)
from snowda.workflows.evaluate_esp_streamflow import (
    compute_esp_metrics,
    esp_metrics_to_dict,
    evaluate_esp_run,
    evaluate_esp_streamflow,
)


@pytest.fixture(scope="module", autouse=True)
def _restore_snowda_logging():
    logger = logging.getLogger("snowda")
    yield
    logger.handlers.clear()
    logger.propagate = True


def _series(
    times: pd.DatetimeIndex, values: np.ndarray, name: str = "obs"
) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=("time",),
        coords={"time": times},
        name=name,
        attrs={"units": "m3 s-1"},
    )


def _paired_forecast(
    init: str,
    member_values: dict[str, list[float]],
    obs_values: list[float],
) -> EspPairedForecast:
    init_time = pd.Timestamp(init)
    n_lead = len(obs_values)
    leads = np.arange(n_lead, dtype=int)
    valid = pd.date_range(init_time.normalize(), periods=n_lead, freq="D")
    members = list(member_values)
    data = np.stack([np.asarray(member_values[m], dtype=float) for m in members])
    fcst = xr.DataArray(
        data,
        dims=("member", "lead"),
        coords={
            "member": members,
            "lead": leads,
            "valid_time": ("lead", valid.to_numpy()),
        },
        name="fcst",
        attrs={"units": "m3 s-1"},
    )
    obs = xr.DataArray(
        np.asarray(obs_values, dtype=float),
        dims=("lead",),
        coords={"lead": leads, "valid_time": ("lead", valid.to_numpy())},
        name="obs",
        attrs={"units": "m3 s-1"},
    )
    return EspPairedForecast(
        fcst=fcst,
        obs=obs,
        init_time=init_time,
        init_month=init_month_key(init_time),
        n_pairs=n_lead,
        source_files=(),
    )


def test_skill_score_and_zero_reference():
    score = xr.DataArray([1.0, 2.0], dims=("x",), coords={"x": [0, 1]})
    ref = xr.DataArray([2.0, 0.0], dims=("x",), coords={"x": [0, 1]})
    result = skill_score(score, ref)
    assert float(result.isel(x=0)) == pytest.approx(0.5)
    assert np.isnan(float(result.isel(x=1)))


def test_crps_perfect_ensemble_is_zero():
    obs = xr.DataArray([1.0, 2.0, 3.0], dims=("lead",), coords={"lead": [0, 1, 2]})
    fcst = xr.DataArray(
        np.vstack([obs.values, obs.values, obs.values]),
        dims=("member", "lead"),
        coords={"member": ["a", "b", "c"], "lead": [0, 1, 2]},
    )
    crps = crps_for_ensemble(fcst, obs, preserve_dims="lead", method="ecdf")
    np.testing.assert_allclose(crps.values, 0.0, atol=1e-12)


def test_pod_and_far_from_binary_events():
    fcst = xr.DataArray([1, 1, 0, 0], dims=("init_time",))
    obs = xr.DataArray([1, 0, 1, 0], dims=("init_time",))
    pod = probability_of_detection(fcst, obs, reduce_dims="init_time")
    far = false_alarm_ratio(fcst, obs, reduce_dims="init_time")
    assert float(pod) == pytest.approx(0.5)
    assert float(far) == pytest.approx(0.5)


def test_pod_and_far_match_contingency_ratios():
    """POD = hits / observed events; FAR = false alarms / forecast events.

    The sample is 2 hits, 1 miss, 1 false alarm, and 3 correct negatives.
    One NaN pair is excluded. The false-alarm rate for this table is 1/4,
    so a ratio/rate mix-up fails this check. A miss with no forecast event
    leaves FAR undefined.
    """
    fcst = xr.DataArray([1, 1, 0, 1, 0, 0, 0, np.nan], dims=("case",))
    obs = xr.DataArray([1, 1, 1, 0, 0, 0, 0, 1], dims=("case",))
    pod = probability_of_detection(fcst, obs, reduce_dims="case")
    far = false_alarm_ratio(fcst, obs, reduce_dims="case")
    hits, misses, false_alarms, correct_negatives = 2, 1, 1, 3
    assert float(pod) == pytest.approx(hits / (hits + misses))
    assert float(far) == pytest.approx(false_alarms / (hits + false_alarms))
    false_alarm_rate = false_alarms / (false_alarms + correct_negatives)
    assert float(far) != pytest.approx(false_alarm_rate)

    miss_only_fcst = xr.DataArray([0, 0], dims=("case",))
    miss_only_obs = xr.DataArray([1, 0], dims=("case",))
    assert float(
        probability_of_detection(miss_only_fcst, miss_only_obs, reduce_dims="case")
    ) == pytest.approx(0.0)
    assert np.isnan(
        float(false_alarm_ratio(miss_only_fcst, miss_only_obs, reduce_dims="case"))
    )


def test_reliability_diagram_perfect_probabilities_lie_on_diagonal():
    probability = xr.DataArray(
        [0.0, 0.0, 0.099, 0.1, 1.0, np.nan],
        dims=("init_time",),
    )
    observed = xr.DataArray(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        dims=("init_time",),
    )
    diagram = reliability_diagram(probability, observed)
    low = diagram.isel(probability_bin=0)
    boundary = diagram.isel(probability_bin=1)
    last = diagram.isel(probability_bin=-1)
    assert int(low["count"]) == 3
    assert float(low["mean_forecast"]) == pytest.approx((0.0 + 0.0 + 0.099) / 3)
    assert float(low["observed_frequency"]) == pytest.approx(0.0)
    assert int(boundary["count"]) == 1
    assert float(boundary["mean_forecast"]) == pytest.approx(0.1)
    assert float(boundary["observed_frequency"]) == pytest.approx(1.0)
    assert int(last["count"]) == 1
    assert float(last["mean_forecast"]) == pytest.approx(1.0)
    assert float(last["observed_frequency"]) == pytest.approx(1.0)
    assert int(diagram["count"].isel(probability_bin=5)) == 0
    assert np.isnan(float(diagram["mean_forecast"].isel(probability_bin=5)))
    assert float(diagram["event_frequency"]) == pytest.approx(2 / 5)


def test_roc_diagram_perfect_separation_passes_through_upper_left():
    probability = xr.DataArray([0.0, 0.0, 1.0, 1.0], dims=("init_time",))
    observed = xr.DataArray([0.0, 0.0, 1.0, 1.0], dims=("init_time",))
    diagram = roc_diagram(probability, observed)
    assert "probability_threshold" in diagram.dims
    assert "threshold" not in diagram.dims
    assert float(diagram["AUC"]) == pytest.approx(1.0)
    pofd = np.asarray(diagram["POFD"].values, dtype=float)
    pod = np.asarray(diagram["POD"].values, dtype=float)
    upper_left = np.where(np.isclose(pofd, 0.0) & np.isclose(pod, 1.0))[0]
    origin = np.where(np.isclose(pofd, 0.0) & np.isclose(pod, 0.0))[0]
    assert upper_left.size >= 1
    assert origin.size >= 1
    assert upper_left[0] < origin[0]


def test_rank_histogram_splits_ties_equally():
    fcst = xr.DataArray(
        [[1.0, 1.0, 1.0]],
        dims=("lead", "member"),
        coords={"lead": [0], "member": ["a", "b", "c"]},
    )
    obs = xr.DataArray([1.0], dims=("lead",), coords={"lead": [0]})
    histogram = rank_histogram(fcst, obs, reduce_dims=["lead"])
    np.testing.assert_allclose(histogram.values, [0.25, 0.25, 0.25, 0.25])


def test_esp_rank_histogram_pools_forecasts_and_drops_incomplete_leads():
    obs_times = pd.date_range("1998-01-01", "2003-12-31", freq="D")
    obs = _series(obs_times, np.full(obs_times.size, 1.0))
    below = _paired_forecast(
        "2001-02-01",
        {"a": [5.0, np.nan], "b": [6.0, 8.0], "c": [7.0, 9.0]},
        [1.0, 0.0],
    )
    above = _paired_forecast(
        "2002-02-01",
        {"a": [1.0, 1.0], "b": [2.0, 2.0], "c": [3.0, 3.0]},
        [10.0, 10.0],
    )
    thresholds = xr.DataArray(
        [10.0],
        dims=("return_period",),
        coords={"return_period": [5]},
        name="yp",
    )
    metrics = compute_esp_metrics(
        [below, above],
        obs,
        thresholds,
        return_periods=(5,),
        exclude_verification_year=True,
        fair_brier=False,
    )
    histogram = metrics["rank_histogram"].sel(init_month="02-01")
    assert int(metrics["rank_histogram_n"].sel(init_month="02-01")) == 3
    assert float(histogram.sel(rank=1)) == pytest.approx(1 / 3)
    assert float(histogram.sel(rank=2)) == pytest.approx(0.0)
    assert float(histogram.sel(rank=3)) == pytest.approx(0.0)
    assert float(histogram.sel(rank=4)) == pytest.approx(2 / 3)


def test_brier_score_known_probability():
    fcst = xr.DataArray(
        [[12.0, 12.0, 8.0], [5.0, 6.0, 7.0]],
        dims=("init_time", "member"),
        coords={
            "init_time": pd.to_datetime(["2001-02-01", "2002-02-01"]),
            "member": [0, 1, 2],
        },
    )
    obs = xr.DataArray(
        [11.0, 4.0],
        dims=("init_time",),
        coords={"init_time": fcst["init_time"]},
    )
    bs = brier_score_for_ensemble(
        fcst, obs, 10.0, fair_correction=False, reduce_dims="init_time"
    )
    assert float(bs) == pytest.approx(((2 / 3 - 1) ** 2 + (0 - 0) ** 2) / 2)


def test_parse_state_date_and_month_key():
    ts = parse_state_date_key("2014020100")
    assert ts == pd.Timestamp("2014-02-01")
    assert init_month_key(ts) == "02-01"


def test_group_esp_trace_files_pools_state_members(tmp_path: Path):
    files = [
        tmp_path
        / "esp"
        / "001"
        / "2014020100"
        / "2016"
        / "mizuroute"
        / "mizuroute_2016.nc",
        tmp_path
        / "esp"
        / "002"
        / "2014020100"
        / "2017"
        / "mizuroute"
        / "mizuroute_2017.nc",
        tmp_path
        / "esp"
        / "001"
        / "2014030100"
        / "2016"
        / "mizuroute"
        / "mizuroute_2016.nc",
    ]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    grouped = group_esp_trace_files(files)
    assert set(grouped) == {"2014020100", "2014030100"}
    assert len(grouped["2014020100"]) == 2


def test_threshold_csv_parsing(tmp_path: Path):
    csv_path = tmp_path / "thresholds.csv"
    pd.DataFrame(
        {"T_yrs": [20, 5, 10], "Yp": [30.0, 10.0, 20.0], "CI_Lower": [1, 2, 3]}
    ).to_csv(csv_path, index=False)
    thresholds = read_return_period_thresholds(csv_path)
    np.testing.assert_allclose(thresholds["return_period"].values, [5, 10, 20])
    np.testing.assert_allclose(thresholds.values, [10.0, 20.0, 30.0])
    selected = select_return_period_thresholds(thresholds, [5, 20])
    np.testing.assert_allclose(selected.values, [10.0, 30.0])


def test_plot_ensemble_envelopes_labels_thresholds():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from snowda.viz.esp_streamflow import plot_ensemble_envelopes, threshold_line_label

    forecast = _paired_forecast(
        "2001-02-01",
        {"a": [1.0, 2.0, 3.0], "b": [1.5, 2.5, 3.5]},
        [1.0, 2.0, 3.0],
    )
    obs_times = pd.date_range("2001-01-01", "2001-03-01", freq="D")
    obs = _series(obs_times, np.linspace(1.0, 3.0, len(obs_times)))
    thresholds = xr.DataArray(
        [10.0, 20.0],
        dims=("return_period",),
        coords={"return_period": [5, 10]},
        name="yp",
    )
    fig, ax = plot_ensemble_envelopes(
        forecasts=[forecast], obs=obs, thresholds=thresholds
    )
    texts = [text.get_text().strip() for text in ax.texts]
    assert "5-year" in texts
    assert "10-year" in texts
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert threshold_line_label(5, 10.0) in legend_labels
    assert threshold_line_label(10, 20.0) in legend_labels
    plt.close(fig)


def test_plot_ensemble_envelopes_shares_init_color_and_sets_xlim():
    import matplotlib
    import matplotlib.dates as mdates

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from snowda.viz.esp_streamflow import plot_ensemble_envelopes

    members = {"a": [1.0, 2.0, 3.0], "b": [1.5, 2.5, 3.5]}
    obs_values = [1.0, 2.0, 3.0]
    forecasts = [
        _paired_forecast("2001-02-01", members, obs_values),
        _paired_forecast("2002-02-01", members, obs_values),
        _paired_forecast("2001-03-01", members, obs_values),
    ]
    fig, ax = plot_ensemble_envelopes(
        forecasts,
        start="2001-01-15",
        end="2002-04-01",
    )
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend_labels.count("02-01") == 1
    assert legend_labels.count("03-01") == 1

    def _rgb(collection) -> tuple[float, float, float]:
        return tuple(float(v) for v in collection.get_facecolor()[0][:3])

    # Two fills per forecast: outer then inner. Feb forecasts share a color.
    fills = ax.collections
    assert _rgb(fills[0]) == _rgb(fills[2])
    assert _rgb(fills[0]) != _rgb(fills[4])
    left, right = ax.get_xlim()
    assert left == pytest.approx(mdates.date2num(pd.Timestamp("2001-01-15")))
    assert right == pytest.approx(mdates.date2num(pd.Timestamp("2002-04-01")))
    plt.close(fig)


def test_plot_peak_scores_one_point_per_initialization():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from snowda.viz.esp_streamflow import plot_peak_scores

    init_time = pd.to_datetime(["2001-02-01", "2002-02-01", "2001-03-01"])
    metrics = xr.Dataset(
        {
            "peak_crps": xr.DataArray(
                [[1.5, 0.0, np.nan], [np.nan, np.nan, 2.5]],
                dims=("init_month", "init_time"),
                coords={"init_month": ["02-01", "03-01"], "init_time": init_time},
                name="peak_crps",
            ),
            "peak_crpss": xr.DataArray(
                [[0.4, 1.0, np.nan], [np.nan, np.nan, -0.2]],
                dims=("init_month", "init_time"),
                coords={"init_month": ["02-01", "03-01"], "init_time": init_time},
                name="peak_crpss",
            ),
        }
    )
    fig = plot_peak_scores(metrics)
    crps_ax, crpss_ax = fig.axes
    assert crps_ax.get_ylabel() == "Peak CRPS"
    assert crpss_ax.get_ylabel() == "Peak CRPSS"
    assert crpss_ax.get_xlabel() == "Initialization time"
    crps_lines = [
        line for line in crps_ax.lines if line.get_label() in {"02-01", "03-01"}
    ]
    assert [line.get_label() for line in crps_lines] == ["02-01", "03-01"]
    np.testing.assert_allclose(crps_lines[0].get_ydata(), [1.5, 0.0])
    np.testing.assert_allclose(crps_lines[1].get_ydata(), [2.5])
    np.testing.assert_allclose(crpss_ax.lines[0].get_ydata(), [0.4, 1.0])
    assert any(line.get_linestyle() == "--" for line in crpss_ax.lines)
    legend = [text.get_text() for text in crps_ax.get_legend().get_texts()]
    assert legend == ["02-01", "03-01"]
    plt.close(fig)

    with pytest.raises(KeyError, match="score_peak"):
        plot_peak_scores(xr.Dataset({"crps": metrics["peak_crps"]}))


def test_plot_rank_histogram_one_panel_per_month():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from snowda.viz.esp_streamflow import plot_rank_histogram

    metrics = xr.Dataset(
        {
            "rank_histogram": xr.DataArray(
                [[1.0, 0.0, 0.0, 0.0]],
                dims=("init_month", "rank"),
                coords={"init_month": ["02-01"], "rank": [1, 2, 3, 4]},
            ),
            "rank_histogram_n": xr.DataArray(
                [4],
                dims=("init_month",),
                coords={"init_month": ["02-01"]},
            ),
        }
    )
    fig = plot_rank_histogram(metrics)
    assert len(fig.axes) == 1
    assert fig.axes[0].get_title() == "02-01 (n=4)"
    assert fig.axes[0].get_xlabel() == "Observation rank"
    assert len(fig.axes[0].patches) == 4
    assert any(line.get_linestyle() == "--" for line in fig.axes[0].lines)
    plt.close(fig)

    with pytest.raises(KeyError, match="rank_histogram"):
        plot_rank_histogram(xr.Dataset())


def test_plot_threshold_metrics_plots_brier_scores_only():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from snowda.viz.esp_streamflow import plot_threshold_metrics

    metrics = xr.Dataset(
        {
            "bs": xr.DataArray(
                [[0.1, 0.2]],
                dims=("init_month", "return_period"),
                coords={"init_month": ["02-01"], "return_period": [5, 10]},
            ),
            "bss": xr.DataArray(
                [[0.3, -0.1]],
                dims=("init_month", "return_period"),
                coords={"init_month": ["02-01"], "return_period": [5, 10]},
            ),
        }
    )
    fig = plot_threshold_metrics(metrics)
    assert len(fig.axes) == 2
    assert fig.axes[0].get_title() == "Brier score"
    assert fig.axes[1].get_title() == "Brier skill score"
    plt.close(fig)


def test_plot_reliability_and_roc_diagrams():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from snowda.viz.esp_streamflow import plot_reliability_diagram, plot_roc_diagram

    probability = xr.DataArray(
        [[[0.0, 1.0], [0.0, 1.0]]],
        dims=("init_month", "return_period", "init_time"),
        coords={
            "init_month": ["02-01"],
            "return_period": [5, 10],
            "init_time": [0, 1],
        },
    )
    observed = xr.DataArray(
        [[[0.0, 1.0], [0.0, 1.0]]],
        dims=("init_month", "return_period", "init_time"),
        coords=probability.coords,
    )
    reliability = reliability_diagram(probability, observed)
    roc = roc_diagram(probability, observed)
    reliability_fig = plot_reliability_diagram(reliability)
    roc_fig = plot_roc_diagram(roc)
    assert len(reliability_fig.axes) == 2
    assert [ax.get_title() for ax in reliability_fig.axes] == ["5-year", "10-year"]
    assert len(roc_fig.axes) == 2
    assert roc_fig.axes[0].get_xlabel() == "False alarm rate"
    assert float(roc["AUC"].sel(init_month="02-01", return_period=5)) == pytest.approx(
        1.0
    )
    plt.close(reliability_fig)
    plt.close(roc_fig)


def test_observational_climatology_matches_calendar_day_and_skips_leap():
    times = pd.date_range("1999-02-27", "2001-03-02", freq="D")
    values = np.arange(len(times), dtype=float)
    obs = _series(times, values)
    target = xr.DataArray(
        pd.to_datetime(["2000-02-28", "2000-02-29", "2000-03-01"]).to_numpy(),
        dims=("lead",),
        coords={"lead": [0, 1, 2]},
        name="valid_time",
    )
    climo = observational_climatology_ensemble(obs, target, exclude_years=[2000])
    assert list(climo["climo_year"].values) == [1999, 2001]
    feb28_1999 = float(obs.sel(time="1999-02-28"))
    assert float(climo.sel(lead=0, climo_year=1999)) == pytest.approx(feb28_1999)
    assert np.isnan(float(climo.sel(lead=1, climo_year=1999)))
    assert np.isnan(float(climo.sel(lead=1, climo_year=2001)))


def test_peak_over_horizon_and_climatology_window():
    fcst = xr.DataArray(
        [[1.0, 5.0, 2.0], [3.0, 4.0, 9.0]],
        dims=("member", "lead"),
        coords={"member": ["a", "b"], "lead": [0, 1, 2]},
    )
    obs = xr.DataArray([1.0, 8.0, 2.0], dims=("lead",), coords={"lead": [0, 1, 2]})
    fcst_peak, obs_peak = peak_over_horizon(fcst, obs)
    np.testing.assert_allclose(fcst_peak.values, [5.0, 9.0])
    assert float(obs_peak) == pytest.approx(8.0)

    full_times = pd.date_range("1999-02-01", "2001-02-10", freq="D")
    full_obs = _series(full_times, np.where(full_times.year == 1999, 3.0, 12.0))
    peaks = climatology_window_peaks(
        full_obs, pd.Timestamp("2000-02-01"), 5, exclude_years=[2000]
    )
    assert float(peaks.sel(climo_year=1999)) == pytest.approx(3.0)
    assert float(peaks.sel(climo_year=2001)) == pytest.approx(12.0)


def test_compute_esp_metrics_perfect_forecast_and_month_aggregation():
    obs_times = pd.date_range("1997-01-01", "2002-12-31", freq="D")
    # Climatology is a large constant; verification years follow the forecast obs.
    obs_values = np.full(obs_times.size, 50.0)
    obs_values[obs_times.year == 2001] = 2.0
    obs_values[obs_times.year == 2002] = 2.0
    obs = _series(obs_times, obs_values)

    feb_2001 = _paired_forecast(
        "2001-02-01",
        {"m1": [1.0, 2.0, 3.0], "m2": [1.0, 2.0, 3.0], "m3": [1.0, 2.0, 3.0]},
        [1.0, 2.0, 3.0],
    )
    feb_2002 = _paired_forecast(
        "2002-02-01",
        {"m1": [1.0, 2.0, 3.0], "m2": [1.0, 2.0, 3.0], "m3": [1.0, 2.0, 3.0]},
        [1.0, 2.0, 3.0],
    )
    mar_2001 = _paired_forecast(
        "2001-03-01",
        {"m1": [4.0, 5.0], "m2": [4.0, 5.0], "m3": [4.0, 5.0]},
        [4.0, 5.0],
    )
    grouped = group_forecasts_by_init_month([feb_2001, mar_2001, feb_2002])
    assert set(grouped) == {"02-01", "03-01"}
    assert len(grouped["02-01"]) == 2

    thresholds = xr.DataArray(
        [10.0, 20.0, 30.0],
        dims=("return_period",),
        coords={"return_period": [5, 10, 20]},
        name="yp",
    )
    metrics = compute_esp_metrics(
        [feb_2001, feb_2002, mar_2001],
        obs,
        thresholds,
        return_periods=(5, 10, 20),
        exclude_verification_year=True,
        crps_method="ecdf",
        fair_brier=False,
    )
    assert list(metrics["init_month"].values) == ["02-01", "03-01"]
    assert int(metrics["n_forecasts"].sel(init_month="02-01")) == 2
    assert int(metrics["n_forecasts"].sel(init_month="03-01")) == 1
    np.testing.assert_allclose(
        metrics["crps"].sel(init_month="02-01").values, 0.0, atol=1e-12
    )
    assert float(metrics["crpss"].sel(init_month="02-01").mean()) == pytest.approx(
        1.0, abs=1e-6
    )
    payload = esp_metrics_to_dict(metrics)
    assert payload["by_init_month"]["02-01"]["n_forecasts"] == 2


def test_compute_esp_metrics_climatology_identical_crpss_is_zero():
    obs_times = pd.date_range("1998-01-01", "2001-12-31", freq="D")
    obs_values = obs_times.year.astype(float) + obs_times.dayofyear / 1000.0
    obs = _series(obs_times, obs_values)

    def _window(year: int) -> list[float]:
        start = pd.Timestamp(year=year, month=2, day=1)
        days = pd.date_range(start, periods=3, freq="D")
        return [float(obs.sel(time=day)) for day in days]

    member_vals = {str(year): _window(year) for year in (1998, 1999, 2000)}
    forecast = _paired_forecast("2001-02-01", member_vals, _window(2001))
    thresholds = xr.DataArray(
        [400.0, 500.0, 600.0],
        dims=("return_period",),
        coords={"return_period": [5, 10, 20]},
        name="yp",
    )
    metrics = compute_esp_metrics(
        [forecast],
        obs,
        thresholds,
        exclude_verification_year=True,
        crps_method="ecdf",
        fair_brier=False,
    )
    assert float(metrics["crpss_mean"].sel(init_month="02-01")) == pytest.approx(
        0.0, abs=1e-6
    )


def _ecdf_crps(ensemble: np.ndarray, obs: float) -> float:
    """Ensemble CRPS for the empirical distribution, Gneiting and Raftery (2007)."""
    x = np.asarray(ensemble, dtype=float)
    mae = float(np.mean(np.abs(x - obs)))
    spread = float(np.mean(np.abs(x[:, None] - x[None, :]))) / 2.0
    return mae - spread


def test_peak_crps_is_one_score_per_forecast_against_peak_climatology():
    obs_times = pd.date_range("1998-01-01", "2002-12-31", freq="D")
    obs = _series(obs_times, np.full(obs_times.size, 10.0))
    imperfect = _paired_forecast(
        "2001-02-01",
        {
            "a": [1.0, 4.0, 1.0],
            "b": [2.0, 2.0, 2.0],
            "c": [0.0, 1.0, 6.0],
            "missing": [np.nan, np.nan, np.nan],
        },
        [1.0, 3.0, 1.0],
    )
    perfect = _paired_forecast(
        "2002-02-01",
        {"a": [5.0, 5.0, 5.0], "b": [5.0, 5.0, 5.0], "c": [5.0, 5.0, 5.0]},
        [5.0, 5.0, 5.0],
    )
    march = _paired_forecast(
        "2001-03-01",
        {"a": [8.0, 1.0], "b": [8.0, 1.0]},
        [8.0, 1.0],
    )
    thresholds = xr.DataArray(
        [100.0],
        dims=("return_period",),
        coords={"return_period": [5]},
        name="yp",
    )
    off = compute_esp_metrics(
        [imperfect, perfect, march],
        obs,
        thresholds,
        return_periods=(5,),
        crps_method="ecdf",
        fair_brier=False,
        score_peak=False,
    )
    assert "peak_crps" not in off

    metrics = compute_esp_metrics(
        [imperfect, perfect, march],
        obs,
        thresholds,
        return_periods=(5,),
        crps_method="ecdf",
        fair_brier=False,
        score_peak=True,
    )
    expected_imperfect = _ecdf_crps(np.array([4.0, 2.0, 6.0]), 3.0)
    expected_ref = _ecdf_crps(np.full(4, 10.0), 3.0)
    feb = metrics["peak_crps"].sel(init_month="02-01")
    finite_feb = feb.where(np.isfinite(feb), drop=True)
    assert finite_feb.sizes["init_time"] == 2
    assert float(feb.sel(init_time="2001-02-01")) == pytest.approx(expected_imperfect)
    assert float(feb.sel(init_time="2002-02-01")) == pytest.approx(0.0, abs=1e-12)
    assert float(
        metrics["peak_crpss"].sel(init_month="02-01", init_time="2001-02-01")
    ) == (pytest.approx(1.0 - expected_imperfect / expected_ref))
    assert float(
        metrics["peak_crpss"].sel(init_month="02-01", init_time="2002-02-01")
    ) == (pytest.approx(1.0, abs=1e-12))
    assert float(metrics["peak_crps_mean"].sel(init_month="02-01")) == pytest.approx(
        expected_imperfect / 2.0
    )
    march_peak = metrics["peak_crps"].sel(init_month="03-01")
    finite_march = march_peak.where(np.isfinite(march_peak), drop=True)
    assert list(pd.to_datetime(finite_march["init_time"].values)) == [
        pd.Timestamp("2001-03-01")
    ]
    assert float(finite_march.values.reshape(-1)[0]) == pytest.approx(0.0, abs=1e-12)

    payload = esp_metrics_to_dict(metrics)
    feb_payload = payload["by_init_month"]["02-01"]
    assert set(feb_payload["peak_crps"]) == {
        "2001-02-01 00:00:00",
        "2002-02-01 00:00:00",
    }
    assert "2001-03-01 00:00:00" not in feb_payload["peak_crps"]
    assert feb_payload["peak_crps"]["2001-02-01 00:00:00"] == pytest.approx(
        expected_imperfect
    )
    assert payload["peak_definition"] == "maximum streamflow over the forecast horizon"


def test_peak_crpss_is_zero_when_forecast_matches_peak_climatology():
    obs_times = pd.date_range("1998-01-01", "2001-12-31", freq="D")
    obs_values = obs_times.year.astype(float) + obs_times.dayofyear / 1000.0
    obs = _series(obs_times, obs_values)

    def _window(year: int) -> list[float]:
        start = pd.Timestamp(year=year, month=2, day=1)
        days = pd.date_range(start, periods=3, freq="D")
        return [float(obs.sel(time=day)) for day in days]

    member_vals = {str(year): _window(year) for year in (1998, 1999, 2000)}
    forecast = _paired_forecast("2001-02-01", member_vals, _window(2001))
    thresholds = xr.DataArray(
        [400.0],
        dims=("return_period",),
        coords={"return_period": [5]},
        name="yp",
    )
    metrics = compute_esp_metrics(
        [forecast],
        obs,
        thresholds,
        return_periods=(5,),
        exclude_verification_year=True,
        crps_method="ecdf",
        fair_brier=False,
        score_peak=True,
    )
    assert float(
        metrics["peak_crpss"].sel(init_month="02-01", init_time="2001-02-01")
    ) == pytest.approx(0.0, abs=1e-6)


def test_peak_binary_scores_exact():
    obs_times = pd.date_range("1998-01-01", "2003-12-31", freq="D")
    obs = _series(obs_times, np.full(obs_times.size, 1.0))
    # Two February inits: one hit, one correct negative.
    hit = _paired_forecast(
        "2001-02-01",
        {"a": [12.0, 1.0], "b": [12.0, 1.0], "c": [8.0, 1.0]},
        [11.0, 1.0],
    )
    miss = _paired_forecast(
        "2002-02-01",
        {"a": [5.0, 1.0], "b": [6.0, 1.0], "c": [7.0, 1.0]},
        [4.0, 1.0],
    )
    false_alarm = _paired_forecast(
        "2001-03-01",
        {"a": [20.0, 1.0], "b": [20.0, 1.0], "c": [20.0, 1.0]},
        [1.0, 1.0],
    )
    thresholds = xr.DataArray(
        [10.0, 50.0, 90.0],
        dims=("return_period",),
        coords={"return_period": [5, 10, 20]},
        name="yp",
    )
    metrics = compute_esp_metrics(
        [hit, miss, false_alarm],
        obs,
        thresholds,
        return_periods=(5,),
        event_probability_threshold=0.5,
        exclude_verification_year=True,
        fair_brier=False,
    )
    feb = metrics.sel(init_month="02-01", return_period=5)
    mar = metrics.sel(init_month="03-01", return_period=5)
    expected_bs = ((2 / 3 - 1) ** 2 + (0.0 - 0.0) ** 2) / 2
    assert float(feb["bs"]) == pytest.approx(expected_bs)
    assert float(feb["pod"]) == pytest.approx(1.0)
    assert float(feb["far"]) == pytest.approx(0.0)
    assert float(feb["event_probability"].sel(init_time="2001-02-01")) == pytest.approx(
        2 / 3
    )
    assert float(feb["event_observed"].sel(init_time="2001-02-01")) == pytest.approx(
        1.0
    )
    assert float(feb["event_probability"].sel(init_time="2002-02-01")) == pytest.approx(
        0.0
    )
    assert float(feb["event_observed"].sel(init_time="2002-02-01")) == pytest.approx(
        0.0
    )
    assert np.isnan(float(mar["pod"]))
    assert float(mar["far"]) == pytest.approx(1.0)


def test_esp_pod_and_far_from_peak_exceedance_table():
    """Peak exceedances become one binary forecast per initialization.

    A forecast event is a yes when the fraction of finite members whose peak
    is at least the return-period flow is at least 0.5. Counts below are for
    the 10 m3 s-1 threshold and were tallied by hand:

    hit, miss, false alarm, correct negative, hit on exact equality,
    excluded NaN observation, miss below the probability threshold,
    hit after dropping NaN members. That is 3 hits, 2 misses, 1 false alarm.
    The March false alarm is a separate sample. A threshold nobody reaches
    has an empty contingency table.
    """
    obs_times = pd.date_range("1998-01-01", "2008-12-31", freq="D")
    obs = _series(obs_times, np.full(obs_times.size, 1.0))
    forecasts = [
        _paired_forecast(
            "2001-02-01",
            {"a": [12.0, 1.0], "b": [12.0, 1.0], "c": [8.0, 1.0]},
            [11.0, 1.0],
        ),
        _paired_forecast(
            "2002-02-01",
            {"a": [9.0, 1.0], "b": [9.0, 1.0], "c": [9.0, 1.0]},
            [15.0, 1.0],
        ),
        _paired_forecast(
            "2003-02-01",
            {"a": [20.0, 1.0], "b": [20.0, 1.0], "c": [1.0, 1.0]},
            [1.0, 1.0],
        ),
        _paired_forecast(
            "2004-02-01",
            {"a": [1.0, 1.0], "b": [1.0, 1.0], "c": [1.0, 1.0]},
            [2.0, 1.0],
        ),
        _paired_forecast(
            "2005-02-01", {"a": [10.0, 1.0], "b": [1.0, 1.0]}, [10.0, 1.0]
        ),
        _paired_forecast(
            "2006-02-01", {"a": [20.0, 1.0], "b": [20.0, 1.0]}, [np.nan, np.nan]
        ),
        _paired_forecast(
            "2007-02-01",
            {"a": [12.0, 1.0], "b": [1.0, 1.0], "c": [1.0, 1.0]},
            [20.0, 1.0],
        ),
        _paired_forecast(
            "2008-02-01",
            {"a": [12.0, 1.0], "b": [np.nan, np.nan], "c": [np.nan, np.nan]},
            [15.0, 1.0],
        ),
        _paired_forecast(
            "2001-03-01",
            {"a": [30.0, 1.0], "b": [30.0, 1.0], "c": [30.0, 1.0]},
            [1.0, 1.0],
        ),
    ]
    thresholds = xr.DataArray(
        [10.0, 1000.0],
        dims=("return_period",),
        coords={"return_period": [5, 20]},
        name="yp",
    )
    metrics = compute_esp_metrics(
        forecasts,
        obs,
        thresholds,
        return_periods=(5, 20),
        event_probability_threshold=0.5,
        exclude_verification_year=True,
        crps_method="ecdf",
        fair_brier=False,
    )
    february_ranks = metrics["rank_histogram"].sel(init_month="02-01")
    assert float(february_ranks.sum()) == pytest.approx(1.0)
    feb = metrics.sel(init_month="02-01", return_period=5)
    hits, misses, false_alarms, correct_negatives = 3, 2, 1, 1
    assert float(feb["pod"]) == pytest.approx(hits / (hits + misses))
    assert float(feb["far"]) == pytest.approx(false_alarms / (hits + false_alarms))
    false_alarm_rate = false_alarms / (false_alarms + correct_negatives)
    assert float(feb["far"]) != pytest.approx(false_alarm_rate)

    mar = metrics.sel(init_month="03-01", return_period=5)
    assert np.isnan(float(mar["pod"]))
    assert float(mar["far"]) == pytest.approx(1.0)

    no_events = metrics.sel(init_month="02-01", return_period=20)
    assert np.isnan(float(no_events["pod"]))
    assert np.isnan(float(no_events["far"]))


def _write_mizuroute(path: Path, times: pd.DatetimeIndex, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {
            "IRFroutedRunoff": xr.DataArray(
                np.asarray(values, dtype=float).reshape(len(times), 1),
                dims=("time", "seg"),
                coords={"time": times, "seg": [0]},
                attrs={"units": "m3 s-1"},
            )
        }
    ).to_netcdf(path)


def test_evaluate_esp_run_writes_outputs(tmp_path: Path):
    run_dir = tmp_path / "esp_run"
    times = pd.date_range("2001-02-01", periods=4, freq="D")
    values = np.array([1.0, 2.0, 3.0, 4.0])
    for year in (1998, 1999, 2000):
        _write_mizuroute(
            run_dir
            / "esp"
            / "001"
            / "2001020100"
            / str(year)
            / "mizuroute"
            / f"mizuroute_{year}.nc",
            times,
            values,
        )
    obs_times = pd.date_range("1998-01-01", "2001-12-31", freq="D")
    obs_values = np.full(obs_times.size, 20.0)
    obs_values[(obs_times >= times.min()) & (obs_times <= times.max())] = np.interp(
        obs_times[(obs_times >= times.min()) & (obs_times <= times.max())].asi8,
        times.asi8,
        values,
    )
    csv_path = tmp_path / "obs.csv"
    pd.DataFrame({"datetime": obs_times, "discharge_cms": obs_values}).to_csv(
        csv_path, index=False
    )
    thresholds_path = tmp_path / "thresholds.csv"
    pd.DataFrame({"T_yrs": [5, 10, 20], "Yp": [50.0, 60.0, 70.0]}).to_csv(
        thresholds_path, index=False
    )

    eval_dir = evaluate_esp_run(
        run_dir,
        csv_path,
        thresholds_path,
        output_dir=tmp_path / "eval",
        resample="1D",
        crps_method="ecdf",
        fair_brier=False,
    )
    assert (eval_dir / "metrics.json").is_file()
    assert (eval_dir / "metrics.nc").is_file()
    assert (eval_dir / "provenance.json").is_file()
    payload = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["by_init_month"]["02-01"]["n_forecasts"] == 1
    assert payload["by_init_month"]["02-01"]["n_members"] == 3
    with xr.open_dataset(eval_dir / "metrics.nc") as ds:
        assert "crps" in ds
        assert "bss" in ds


def test_evaluate_esp_streamflow_from_yaml(tmp_path: Path):
    run_dir = tmp_path / "esp_run"
    times = pd.date_range("2001-03-01", periods=3, freq="D")
    values = np.array([2.0, 2.0, 2.0])
    for year in (1999, 2000):
        _write_mizuroute(
            run_dir
            / "esp"
            / "001"
            / "2001030100"
            / str(year)
            / "mizuroute"
            / f"mizuroute_{year}.nc",
            times,
            values,
        )
    obs_times = pd.date_range("1999-01-01", "2001-12-31", freq="D")
    csv_path = tmp_path / "obs.csv"
    pd.DataFrame(
        {"datetime": obs_times, "discharge_cms": np.full(obs_times.size, 2.0)}
    ).to_csv(csv_path, index=False)
    thresholds_path = tmp_path / "thresholds.csv"
    pd.DataFrame({"T_yrs": [5, 10, 20], "Yp": [8.0, 9.0, 10.0]}).to_csv(
        thresholds_path, index=False
    )
    yaml_path = tmp_path / "eval.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "case_name: test_esp_eval",
                f"output_dir: {tmp_path / 'out'}",
                f"run_dir: {run_dir}",
                f"obs_csv: {csv_path}",
                f"thresholds_csv: {thresholds_path}",
                "resample: 1D",
                "return_periods: [5, 10, 20]",
                "crps_method: ecdf",
                "fair_brier: false",
                "score_peak: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    eval_dir = evaluate_esp_streamflow(yaml_path)
    payload = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "03-01" in payload["by_init_month"]
    march = payload["by_init_month"]["03-01"]
    assert march["peak_crps_mean"] == pytest.approx(0.0, abs=1e-12)
    assert set(march["peak_crps"]) == {"2001-03-01 00:00:00"}
