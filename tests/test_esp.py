from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from snowda.config.pysummada import _load_esp_config
from snowda.workflows.esp import (
    _dropped_leap_days,
    _mizuroute_sim_end,
    _shift_time_to_state_year,
    _trace_years,
    forecast_end_datetime,
    resolve_esp_trigger_dates,
    resolve_forecast_window_end,
)


def test_resolve_esp_trigger_dates_with_frequency_and_month_filter():
    start = datetime(2010, 1, 1, 0, 0)
    end = datetime(2010, 3, 15, 0, 0)
    explicit = [datetime(2010, 2, 10, 0, 0)]
    resolved = resolve_esp_trigger_dates(
        start,
        end,
        explicit,
        frequency_days=30,
        months=[2, 3],
    )
    assert resolved == [
        datetime(2010, 2, 10, 0, 0),
        datetime(2010, 3, 2, 0, 0),
    ]


def test_load_esp_config_deterministic_state(tmp_path: Path):
    esp_cfg = {
        "enabled": True,
        "start_year": 2000,
        "end_year": 2001,
        "forecast_length_days": 30,
        "frequency_days": 30,
        "months": [12],
        "restart_file_filter": "restart",
        "deterministic_state": {
            "path": str(tmp_path / "state.nc"),
            "date": "2010-12-01 00:00",
            "member": "001",
        },
    }
    cfg = {"esp": esp_cfg}
    esp = _load_esp_config(cfg, tmp_path)
    assert esp.deterministic_state is not None
    assert esp.deterministic_state.path == tmp_path / "state.nc"
    assert esp.deterministic_state.date == datetime(2010, 12, 1, 0, 0)
    assert esp.deterministic_state.member == "001"


def test_load_esp_config_deterministic_flag(tmp_path: Path):
    esp_cfg = {
        "enabled": True,
        "start_year": 2000,
        "end_year": 2001,
        "forecast_length_days": 30,
        "frequency_days": 30,
        "months": [12],
        "restart_file_filter": "restart",
        "deterministic": True,
    }
    cfg = {"esp": esp_cfg}
    esp = _load_esp_config(cfg, tmp_path)
    assert esp.deterministic
    assert esp.deterministic_state is None


def test_load_esp_config_mizuroute_resources(tmp_path: Path):
    param = tmp_path / "param.nml.default"
    topology = tmp_path / "topology.nc"
    param.write_text("param", encoding="utf-8")
    topology.write_text("topology", encoding="utf-8")
    mizu_cfg = {
        "executable": str(tmp_path / "route.exe"),
        "control_file": str(tmp_path / "mizuroute.control"),
        "ancil_dir": str(tmp_path / "ancil"),
        "case_name": "case",
        "restart_write": "monthly",
        "param_nml": str(param),
        "topology_nc": str(topology),
        "log_output": True,
    }
    esp_cfg = {
        "enabled": True,
        "start_year": 2000,
        "end_year": 2001,
        "forecast_length_days": 30,
        "frequency_days": 30,
        "months": [12],
        "restart_file_filter": "restart",
        "run_mizuroute": True,
        "mizuroute": mizu_cfg,
    }
    cfg = {"esp": esp_cfg}
    esp = _load_esp_config(cfg, tmp_path)
    assert esp.mizuroute is not None
    assert esp.mizuroute.param_nml == param
    assert esp.mizuroute.topology_nc == topology
    assert esp.mizuroute.log_output is True


def test_load_esp_config_forecast_end(tmp_path: Path):
    esp_cfg = {
        "enabled": True,
        "start_year": 2000,
        "end_year": 2001,
        "forecast_end": "08-01",
        "frequency_days": 30,
        "months": [12],
        "restart_file_filter": "restart",
    }
    esp = _load_esp_config({"esp": esp_cfg}, tmp_path)
    assert esp.forecast_length_days is None
    assert esp.forecast_end == (8, 1)


def test_forecast_end_datetime_same_year():
    start = datetime(2014, 2, 1, 0, 0)
    assert forecast_end_datetime(start, 8, 1) == datetime(2014, 8, 1, 0, 0)


def test_forecast_end_datetime_wraps_to_next_year():
    start = datetime(2014, 9, 1, 0, 0)
    assert forecast_end_datetime(start, 8, 1) == datetime(2015, 8, 1, 0, 0)


def test_forecast_end_datetime_clamps_leap_day():
    start = datetime(2015, 1, 15, 0, 0)
    assert forecast_end_datetime(start, 2, 29) == datetime(2015, 2, 28, 0, 0)


def test_resolve_forecast_window_end_duration():
    start = datetime(2014, 2, 1, 0, 0)
    end = resolve_forecast_window_end(start, 30, None)
    assert end == datetime(2014, 3, 3, 0, 0)


def test_resolve_forecast_window_end_calendar():
    start = datetime(2014, 5, 1, 0, 0)
    end = resolve_forecast_window_end(start, None, (8, 1))
    assert end == datetime(2014, 8, 1, 0, 0)


def test_mizuroute_sim_end_shortens_by_dropped_leap_days():
    base_end = datetime(2014, 6, 1, 0, 0)
    assert _mizuroute_sim_end(base_end, 1) == datetime(2014, 5, 30, 23, 0)
    assert _mizuroute_sim_end(base_end) == datetime(2014, 5, 31, 23, 0)


def test_dropped_leap_days_counts_only_29_february_inside_the_trace():
    # 2016 trace shown on the 2014 calendar: 29 Feb 2016 has no 2014 date.
    assert _dropped_leap_days(datetime(2016, 2, 1), datetime(2016, 4, 1), -2) == 1
    # A leap trace year that starts after February loses nothing.
    assert _dropped_leap_days(datetime(2016, 4, 1), datetime(2016, 8, 1), -2) == 0
    # Leap year onto leap year keeps the day.
    assert _dropped_leap_days(datetime(2016, 2, 1), datetime(2016, 4, 1), 4) == 0
    # The leap day falls in the year after the trace starts.
    assert _dropped_leap_days(datetime(2015, 11, 1), datetime(2016, 4, 1), -1) == 1


def test_shift_time_keeps_a_trace_that_crosses_new_year_in_order(tmp_path):
    times = pd.date_range("1995-11-01", "1996-02-01", freq="MS")
    source = tmp_path / "trace.nc"
    xr.Dataset({"q": ("time", np.arange(times.size, dtype=float))}, {"time": times}).to_netcdf(source)
    shifted = _shift_time_to_state_year(source, 2010, tmp_path / "shifted.nc", trace_year=1995)
    with xr.open_dataset(shifted) as ds:
        assert list(pd.DatetimeIndex(ds["time"].values)) == list(
            pd.date_range("2010-11-01", "2011-02-01", freq="MS")
        )


def test_trace_years_leave_out_the_issue_year_unless_asked():
    esp = SimpleNamespace(start_year=2013, end_year=2016, exclude_issue_year=True)
    assert _trace_years(esp, datetime(2014, 3, 1)) == [2013, 2015, 2016]
    esp.exclude_issue_year = False
    assert _trace_years(esp, datetime(2014, 3, 1)) == [2013, 2014, 2015, 2016]

