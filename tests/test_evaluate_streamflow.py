from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from snowda.adapters.scores import compute_deterministic_metrics
from snowda.config.pysummada import StreamflowEvaluationConfig
from snowda.io.streamflow import (
    load_paired_streamflow,
    pair_streamflow,
    read_routed_streamflow,
    read_streamflow_observations,
)
from snowda.workflows.evaluate_streamflow import (
    evaluate_deterministic_run,
    evaluate_streamflow,
    run_evaluation,
)


def _series(values: list[float], start: str = "2000-01-01", freq: str = "D") -> xr.DataArray:
    time = pd.date_range(start, periods=len(values), freq=freq)
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=("time",),
        coords={"time": time},
        name="flow",
        attrs={"units": "m3 s-1"},
    )


def test_perfect_forecast_scores_are_ideal():
    obs = _series([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0])
    metrics = compute_deterministic_metrics(obs, obs)
    assert metrics.attrs["n_pairs"] == 10
    assert metrics.attrs["kge_method"] == "2009"
    assert float(metrics["nse"]) == pytest.approx(1.0)
    assert float(metrics["kge"]) == pytest.approx(1.0)
    assert float(metrics["rho"]) == pytest.approx(1.0)
    assert float(metrics["alpha"]) == pytest.approx(1.0)
    assert float(metrics["beta"]) == pytest.approx(1.0)
    assert float(metrics["rmse"]) == pytest.approx(0.0)
    assert float(metrics["mae"]) == pytest.approx(0.0)
    assert float(metrics["pbias"]) == pytest.approx(0.0)
    assert float(metrics["additive_bias"]) == pytest.approx(0.0)
    assert float(metrics["pearsonr"]) == pytest.approx(1.0)


def test_constant_additive_bias():
    obs = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = obs + 2.0
    metrics = compute_deterministic_metrics(sim, obs)
    assert float(metrics["additive_bias"]) == pytest.approx(2.0)
    assert float(metrics["rmse"]) == pytest.approx(2.0)
    assert float(metrics["mae"]) == pytest.approx(2.0)
    assert float(metrics["pbias"]) == pytest.approx(100.0 * 10.0 / 15.0)
    assert float(metrics["nse"]) < 1.0
    assert float(metrics["beta"]) == pytest.approx((obs.mean() + 2.0) / obs.mean())


def test_nan_pairs_are_dropped_from_n_pairs():
    obs = _series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    sim = obs.copy()
    sim = sim.copy(data=sim.values.copy())
    sim.values[1] = np.nan
    obs = obs.copy(data=obs.values.copy())
    obs.values[4] = np.nan
    metrics = compute_deterministic_metrics(sim, obs)
    assert metrics.attrs["n_pairs"] == 4
    assert float(metrics["nse"]) == pytest.approx(1.0)


def test_too_few_pairs_raises():
    obs = _series([1.0])
    with pytest.raises(ValueError, match="at least 2"):
        compute_deterministic_metrics(obs, obs)


def test_csv_column_autodetection(tmp_path: Path):
    times = pd.date_range("2000-01-01", periods=4, freq="D")
    csv_path = tmp_path / "obs.csv"
    pd.DataFrame({"date": times, "Q": [1.0, 2.0, 3.0, 4.0]}).to_csv(csv_path, index=False)
    obs = read_streamflow_observations(csv_path)
    assert obs.sizes["time"] == 4
    np.testing.assert_allclose(obs.values, [1.0, 2.0, 3.0, 4.0])


def test_blank_csv_rows_are_dropped_before_resample(tmp_path: Path):
    times = pd.date_range("2000-01-01", periods=4, freq="D")
    csv_path = tmp_path / "obs.csv"
    pd.DataFrame(
        {
            "datetime": list(times) + [pd.NaT, pd.NaT],
            "discharge_cms": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan],
        }
    ).to_csv(csv_path, index=False)
    obs = read_streamflow_observations(csv_path)
    assert obs.sizes["time"] == 4
    sim = _series([1.0, 2.0, 3.0, 4.0], start="2000-01-01")
    paired_sim, paired_obs = pair_streamflow(sim, obs, resample="1D")
    np.testing.assert_allclose(paired_sim.values, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(paired_obs.values, [1.0, 2.0, 3.0, 4.0])


def test_trailing_empty_csv_rows_are_not_invalid_timestamps(tmp_path: Path, caplog):
    times = pd.date_range("2000-01-01", periods=3, freq="D")
    csv_path = tmp_path / "obs.csv"
    body = pd.DataFrame({"datetime": times, "discharge_cms": [1.0, 2.0, 3.0]})
    csv_path.write_text(
        body.to_csv(index=False) + ",\n,\n,\n",
        encoding="utf-8-sig",
    )
    with caplog.at_level("WARNING"):
        obs = read_streamflow_observations(csv_path)
    assert obs.sizes["time"] == 3
    assert not any("invalid timestamps" in rec.message for rec in caplog.records)


def test_unparseable_observation_dates_are_dropped_with_warning(tmp_path: Path, caplog):
    csv_path = tmp_path / "obs.csv"
    csv_path.write_text(
        "datetime,discharge_cms\n"
        "2000-01-01,1.0\n"
        "not-a-date,2.0\n"
        "2000-01-03,3.0\n",
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        obs = read_streamflow_observations(csv_path)
    assert obs.sizes["time"] == 2
    np.testing.assert_allclose(obs.values, [1.0, 3.0])
    assert any("invalid timestamps" in rec.message for rec in caplog.records)


def test_nodata_sentinels_are_masked_and_dropped(tmp_path: Path):
    times = pd.date_range("2000-01-01", periods=5, freq="D")
    csv_path = tmp_path / "obs.csv"
    pd.DataFrame(
        {
            "datetime": times,
            "discharge_cms": [1.0, -9999.0, 3.0, -999.0, 5.0],
        }
    ).to_csv(csv_path, index=False)
    sim = _series([1.0, 2.0, 3.0, 4.0, 5.0], start="2000-01-01")
    paired_sim, paired_obs = pair_streamflow(
        sim, read_streamflow_observations(csv_path), resample="1D"
    )
    np.testing.assert_allclose(paired_obs.values, [1.0, 3.0, 5.0])
    np.testing.assert_allclose(paired_sim.values, [1.0, 3.0, 5.0])


def test_disjoint_sim_obs_times_report_spans():
    sim = _series([1.0, 2.0, 3.0], start="2000-09-01")
    obs = _series([1.0, 2.0, 3.0], start="2001-01-01")
    with pytest.raises(ValueError, match="sim time .*2000-09-01.*obs time .*2001-01-01"):
        pair_streamflow(sim, obs, resample="1D")


def test_unsorted_and_duplicate_observation_times_are_cleaned():
    times = pd.to_datetime(["2000-01-03", "2000-01-01", "2000-01-01", "2000-01-02"])
    obs = xr.DataArray(
        np.asarray([4.0, 1.0, 3.0, 2.0], dtype=float),
        dims=("time",),
        coords={"time": times},
        name="obs",
        attrs={"units": "m3 s-1"},
    )
    sim = _series([1.0, 2.0, 4.0], start="2000-01-01")
    paired_sim, paired_obs = pair_streamflow(sim, obs, resample="1D")
    np.testing.assert_allclose(paired_sim.values, [1.0, 2.0, 4.0])
    np.testing.assert_allclose(paired_obs.values, [2.0, 2.0, 4.0])


def test_load_and_pair_hourly_to_daily(tmp_path: Path):
    time = pd.date_range("2000-01-01", periods=48, freq="h")
    values = np.arange(48, dtype=float)
    nc_path = tmp_path / "mizuroute.nc"
    xr.Dataset(
        {
            "IRFroutedRunoff": xr.DataArray(
                values.reshape(48, 1),
                dims=("time", "seg"),
                coords={"time": time, "seg": [101], "reachID": ("seg", [101])},
                attrs={"units": "m3 s-1"},
            )
        }
    ).to_netcdf(nc_path)
    csv_path = tmp_path / "obs.csv"
    pd.DataFrame({"datetime": time, "discharge_cms": values}).to_csv(csv_path, index=False)

    sim, obs = load_paired_streamflow(nc_path, csv_path, resample="1D")
    assert sim.sizes["time"] == 2
    assert obs.sizes["time"] == 2
    metrics = compute_deterministic_metrics(sim, obs)
    assert metrics.attrs["n_pairs"] == 2
    assert float(metrics["nse"]) == pytest.approx(1.0)
    assert float(metrics["kge"]) == pytest.approx(1.0)

    selected = read_routed_streamflow(nc_path, reach_id=101)
    assert selected.sizes["time"] == 48


def test_pair_streamflow_inner_join_and_native_timestep():
    sim = _series([1.0, 2.0, 3.0, 4.0], start="2000-01-02")
    obs = _series([10.0, 1.0, 2.0, 3.0, 4.0, 99.0], start="2000-01-01")
    paired_sim, paired_obs = pair_streamflow(sim, obs, resample=None)
    np.testing.assert_array_equal(
        paired_sim["time"].values, pd.date_range("2000-01-02", periods=4, freq="D")
    )
    np.testing.assert_allclose(paired_sim.values, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(paired_obs.values, [1.0, 2.0, 3.0, 4.0])


def test_equivalent_streamflow_unit_aliases():
    sim = _series([1.0, 2.0, 3.0])
    sim.attrs["units"] = "m3/s"
    obs = _series([1.0, 2.0, 3.0])
    obs.attrs["units"] = "m3 s-1"
    paired_sim, paired_obs = pair_streamflow(sim, obs, resample=None)
    assert paired_sim.attrs["units"] == "m3 s-1"
    assert paired_obs.attrs["units"] == "m3 s-1"


def test_run_evaluation_writes_outputs(tmp_path: Path):
    time = pd.date_range("2000-06-01", periods=10, freq="D")
    values = np.linspace(1.0, 10.0, 10)
    nc_path = tmp_path / "sim.nc"
    xr.Dataset(
        {
            "IRFroutedRunoff": xr.DataArray(
                values.reshape(10, 1),
                dims=("time", "seg"),
                coords={"time": time, "seg": [0]},
                attrs={"units": "m3 s-1"},
            )
        }
    ).to_netcdf(nc_path)
    csv_path = tmp_path / "obs.csv"
    pd.DataFrame({"datetime": time, "discharge_cms": values}).to_csv(csv_path, index=False)

    run_dir = run_evaluation(
        nc_path,
        csv_path,
        output_dir=tmp_path / "eval",
        resample="1D",
    )
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "metrics.nc").is_file()
    assert (run_dir / "paired_streamflow.nc").is_file()
    assert (run_dir / "provenance.json").is_file()
    assert (run_dir / "config.resolved.json").is_file()
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["n_pairs"] == 10
    assert payload["nse"] == pytest.approx(1.0)
    assert payload["kge_method"] == "2009"
    with xr.open_dataset(run_dir / "paired_streamflow.nc") as paired:
        assert paired.sizes["time"] == 10
    with xr.open_dataset(run_dir / "metrics.nc") as metrics_ds:
        assert float(metrics_ds["nse"]) == pytest.approx(1.0)


def test_evaluate_streamflow_from_yaml(tmp_path: Path):
    time = pd.date_range("2001-01-01", periods=6, freq="D")
    values = np.array([2.0, 4.0, 6.0, 8.0, 4.0, 2.0])
    nc_path = tmp_path / "sim.nc"
    xr.Dataset(
        {
            "IRFroutedRunoff": xr.DataArray(
                values.reshape(6, 1),
                dims=("time", "seg"),
                coords={"time": time, "seg": [0]},
            )
        }
    ).to_netcdf(nc_path)
    csv_path = tmp_path / "obs.csv"
    pd.DataFrame({"datetime": time, "discharge_cms": values}).to_csv(csv_path, index=False)
    yaml_path = tmp_path / "eval.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "case_name: test_eval",
                f"output_dir: {tmp_path / 'out'}",
                f"sim_nc: {nc_path}",
                f"obs_csv: {csv_path}",
                "resample: 1D",
                "seg: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = evaluate_streamflow(yaml_path)
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["nse"] == pytest.approx(1.0)
    assert payload["kge"] == pytest.approx(1.0)


def test_evaluate_deterministic_run_writes_into_run_dir(tmp_path: Path):
    time = pd.date_range("2002-01-01", periods=5, freq="D")
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    nc_path = tmp_path / "mizuroute_deterministic.nc"
    xr.Dataset(
        {
            "IRFroutedRunoff": xr.DataArray(
                values.reshape(5, 1),
                dims=("time", "seg"),
                coords={"time": time, "seg": [0]},
                attrs={"units": "m3 s-1"},
            )
        }
    ).to_netcdf(nc_path)
    csv_path = tmp_path / "obs.csv"
    pd.DataFrame({"datetime": time, "discharge_cms": values}).to_csv(csv_path, index=False)
    run_dir = tmp_path / "20260101T000000Z"
    run_dir.mkdir()
    evaluation = StreamflowEvaluationConfig(obs_csv=csv_path, resample="1D", seg=0)
    metrics = evaluate_deterministic_run(nc_path, evaluation, run_dir)
    eval_dir = run_dir / "evaluation"
    assert (eval_dir / "metrics.json").is_file()
    assert (eval_dir / "metrics.nc").is_file()
    assert (eval_dir / "paired_streamflow.nc").is_file()
    assert metrics["nse"] == pytest.approx(1.0)
    assert metrics["n_pairs"] == 5
