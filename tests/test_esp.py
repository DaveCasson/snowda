from __future__ import annotations

from datetime import datetime
from pathlib import Path

from snowda.config.pysummada import _load_esp_config
from snowda.workflows.esp import discover_restart_dates, resolve_esp_trigger_dates


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
        "output_dir": str(tmp_path / "output"),
        "restart_dir": str(tmp_path / "restart"),
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
