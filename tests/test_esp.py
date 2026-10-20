from __future__ import annotations

from datetime import datetime
from pathlib import Path

from snowda.config.pysummada import _load_esp_config
from snowda.workflows.esp import (
    _mizuroute_sim_end,
    discover_restart_dates,
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


def test_discover_restart_dates_filters_by_member(tmp_path: Path):
    files = [
        "001_restart_2011040100_2011_04_01.nc",
        "001_restart_2011050100_2011_05_01.nc",
        "002_restart_2011040100_2011_04_01.nc",
        "notes.txt",
    ]
    for name in files:
        (tmp_path / name).write_text("x", encoding="utf-8")

    discovered = discover_restart_dates(tmp_path, "restart", "001")
    assert discovered.keys() == {"2011040100", "2011050100"}
    assert discovered["2011040100"].name.startswith("001_restart_2011040100")


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


def test_mizuroute_sim_end_leap_adjust_for_duration():
    base_end = datetime(2014, 6, 1, 0, 0)
    leap = _mizuroute_sim_end(base_end, 2016, leap_day_adjust=True)
    non_leap = _mizuroute_sim_end(base_end, 2015, leap_day_adjust=True)
    assert leap == datetime(2014, 5, 30, 23, 0)
    assert non_leap == datetime(2014, 5, 31, 23, 0)


def test_mizuroute_sim_end_skips_leap_adjust_for_calendar():
    base_end = datetime(2014, 8, 1, 0, 0)
    leap = _mizuroute_sim_end(base_end, 2016, leap_day_adjust=False)
    non_leap = _mizuroute_sim_end(base_end, 2015, leap_day_adjust=False)
    assert leap == non_leap == datetime(2014, 7, 31, 23, 0)

