"""Tests for SUMMA DA historic, assimilation, and ESP forecast plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from snowda.io.da_traces import load_da_streamflow, load_da_swe
from snowda.io.esp_streamflow import ENSEMBLE_DIM


def _write_series(path: Path, **variables: tuple[np.ndarray, str]) -> None:
    times = pd.date_range("2014-03-01", periods=25, freq="h")
    data = {}
    for name, (values, units) in variables.items():
        data[name] = ("time", values)
    ds = xr.Dataset(data, coords={"time": times})
    for name, (_values, units) in variables.items():
        ds[name].attrs["units"] = units
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _forecast_times() -> pd.DatetimeIndex:
    return pd.date_range("2014-03-02", periods=24, freq="h")


def _write_run(root: Path) -> None:
    swe = np.arange(25, dtype=float)
    runoff = np.full(25, 2.0e-6)
    _write_series(
        root / "summa" / "001_2014_03_02_timestep.nc",
        scalarSWE=(swe, "kg m-2"),
        averageRoutedRunoff=(runoff, "m s-1"),
    )
    _write_series(
        root / "summa" / "002_2014_03_02_timestep.nc",
        scalarSWE=(np.full(25, 10.0), "kg m-2"),
        averageRoutedRunoff=(np.full(25, 4.0e-6), "m s-1"),
    )
    obs = root / "assimilation" / "2014_03_02_swe_obs.csv"
    obs.parent.mkdir(parents=True, exist_ok=True)
    obs.write_text(
        "assimilation_date,station_id,time,swe_model_equivalent,lumped_mean\n"
        "2014-03-02 00:00,s1,2014-03-02 00:00,24.0,24.0\n"
    )
    area_dir = root / "esp" / "001" / "2014030200" / "2010" / "mizuroute"
    area_dir.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"area": ("hru", [1.0e6])}).to_netcdf(area_dir / "topology.nc")
    for year, swe_value, flow in ((2010, 5.0, 3.0), (2011, 7.0, 4.0)):
        mizu = root / "esp" / "001" / "2014030200" / f"{year}" / "mizuroute"
        mizu.mkdir(parents=True, exist_ok=True)
        times = _forecast_times()
        swe_ds = xr.Dataset(
            {"scalarSWE": ("time", np.full(len(times), swe_value))},
            coords={"time": times},
        )
        swe_ds["scalarSWE"].attrs["units"] = "kg m-2"
        swe_ds.to_netcdf(mizu / f"summa_{year}.nc")
        flow_ds = xr.Dataset(
            {"IRFroutedRunoff": (("time", "seg"), np.full((len(times), 1), flow))},
            coords={"time": times},
        )
        flow_ds["IRFroutedRunoff"].attrs["units"] = "m3/s"
        flow_ds.to_netcdf(mizu / f"mizuroute_{year}.nc")


def test_load_da_swe_keeps_update_state_and_forecasts(tmp_path: Path):
    _write_run(tmp_path)
    traces = load_da_swe(tmp_path)

    assert [item.member for item in traces.historic] == ["001", "002"]
    segment = traces.historic[0].segments[0]
    index = pd.DatetimeIndex(segment["time"].values)
    assert index[-1] == pd.Timestamp("2014-03-02")
    assert segment.values[-1] == pytest.approx(24.0)
    assert segment.values[0] == pytest.approx(np.arange(24).mean())

    assert len(traces.states) == 1
    state = traces.states[0]
    assert state.time == pd.Timestamp("2014-03-02")
    assert state.observation == pytest.approx(24.0)
    assert float(state.values.sel({ENSEMBLE_DIM: "001"})) == pytest.approx(24.0)
    assert float(state.values.sel({ENSEMBLE_DIM: "002"})) == pytest.approx(10.0)

    assert len(traces.forecasts) == 1
    forecast = traces.forecasts[0]
    assert forecast.init_time == pd.Timestamp("2014-03-02")
    assert forecast.init_month == "03-02"
    assert set(forecast.values[ENSEMBLE_DIM].values) == {"001_2010", "001_2011"}
    assert float(forecast.values.sel({ENSEMBLE_DIM: "001_2010"}).values[0]) == pytest.approx(
        5.0
    )


def test_load_da_streamflow_converts_specific_discharge(tmp_path: Path):
    _write_run(tmp_path)
    traces = load_da_streamflow(tmp_path)

    assert traces.units == "m3 s-1"
    state = traces.states[0]
    assert state.observation is None
    assert float(state.values.sel({ENSEMBLE_DIM: "001"})) == pytest.approx(2.0)
    assert float(state.values.sel({ENSEMBLE_DIM: "002"})) == pytest.approx(4.0)
    forecast = traces.forecasts[0].values
    assert float(forecast.sel({ENSEMBLE_DIM: "001_2010"}).values[0]) == pytest.approx(3.0)
    assert float(forecast.sel({ENSEMBLE_DIM: "001_2011"}).values[0]) == pytest.approx(4.0)


def test_historic_run_without_esp_uses_external_topology(tmp_path: Path):
    root = tmp_path / "run"
    _write_series(
        root / "summa" / "001_2014_03_02_timestep.nc",
        scalarSWE=(np.arange(25, dtype=float), "kg m-2"),
        averageRoutedRunoff=(np.full(25, 2.0e-6), "m s-1"),
    )
    obs = root / "assimilation" / "2014_03_02_swe_obs.csv"
    obs.parent.mkdir(parents=True, exist_ok=True)
    obs.write_text(
        "assimilation_date,station_id,time,swe_model_equivalent,lumped_mean\n"
        "2014-03-02 00:00,s1,2014-03-02 00:00,24.0,24.0\n"
    )
    topology = tmp_path / "topology.nc"
    xr.Dataset({"area": ("hru", [1.0e6])}).to_netcdf(topology)

    swe = load_da_swe(root)
    assert swe.forecasts == ()
    assert len(swe.historic) == 1

    flow = load_da_streamflow(root, topology_nc=topology)
    assert flow.forecasts == ()
    assert float(flow.states[0].values.sel({ENSEMBLE_DIM: "001"})) == pytest.approx(2.0)


def test_historic_streamflow_reads_mizuroute_output(tmp_path: Path):
    root = tmp_path / "run"
    times = pd.date_range("2014-03-01", periods=25, freq="h")
    routed = root / "mizuroute" / "001_2014_03_02_mizuroute.nc"
    routed.parent.mkdir(parents=True)
    ds = xr.Dataset(
        {"IRFroutedRunoff": (("time", "seg"), np.full((len(times), 1), 6.0))},
        coords={"time": times},
    )
    ds["IRFroutedRunoff"].attrs["units"] = "m3/s"
    ds.to_netcdf(routed)
    obs = root / "assimilation" / "2014_03_02_swe_obs.csv"
    obs.parent.mkdir(parents=True, exist_ok=True)
    obs.write_text(
        "assimilation_date,station_id,time,swe_model_equivalent,lumped_mean\n"
        "2014-03-02 00:00,s1,2014-03-02 00:00,24.0,24.0\n"
    )

    flow = load_da_streamflow(root)
    assert flow.forecasts == ()
    assert flow.historic[0].member == "001"
    assert float(flow.states[0].values.sel({ENSEMBLE_DIM: "001"})) == pytest.approx(6.0)


def test_plot_da_traces_labels_historic_state_and_forecast():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from snowda.io.da_traces import (
        AssimilationState,
        DaForecast,
        DaTraceSet,
        HistoricMemberTrace,
    )
    from snowda.viz.summa_da import plot_da_traces

    historic_time = pd.date_range("2014-03-01", periods=2, freq="D")
    historic = xr.DataArray(
        [1.0, 2.0],
        dims=("time",),
        coords={"time": historic_time},
        name="scalarSWE",
    )
    forecast_time = pd.date_range("2014-03-02", periods=3, freq="D")
    forecast = xr.DataArray(
        [[2.0, 3.0, 4.0], [2.5, 3.5, 4.5]],
        dims=(ENSEMBLE_DIM, "time"),
        coords={ENSEMBLE_DIM: ["a", "b"], "time": forecast_time},
    )
    traces = DaTraceSet(
        historic=(HistoricMemberTrace("001", (historic,)),),
        states=(
            AssimilationState(
                time=pd.Timestamp("2014-03-02"),
                values=xr.DataArray(
                    [2.0],
                    dims=(ENSEMBLE_DIM,),
                    coords={ENSEMBLE_DIM: ["001"]},
                ),
                observation=1.5,
            ),
        ),
        forecasts=(
            DaForecast(
                init_time=pd.Timestamp("2014-03-02"),
                init_month="03-02",
                values=forecast,
            ),
        ),
        units="kg m-2",
        ylabel="SWE (kg m-2)",
    )
    obs = xr.DataArray(
        [1.0, 1.2, 1.4],
        dims=("time",),
        coords={"time": forecast_time},
        name="obs",
    )
    fig, ax = plot_da_traces(traces, obs, title="DA SWE")
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "Historic ensemble" in labels
    assert "State at assimilation" in labels
    assert "SWE observation" in labels
    assert "Assimilation" in labels
    assert "Forecast 2014-03-02" in labels
    assert "Observed" in labels
    assert ax.get_ylabel() == "SWE (kg m-2)"
    assert ax.get_title() == "DA SWE"
    plt.close(fig)
