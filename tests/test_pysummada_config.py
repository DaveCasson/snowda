from pathlib import Path

import pytest
import yaml

from snowda.config.pysummada import load_pysummada_config


def _write_base_file_manager(path: Path) -> None:
    content = """outputPath\t'/tmp/output/'
initConditionFile\t'coldState.nc'
"""
    path.write_text(content, encoding="utf-8")


def _build_config(
    tmp_path: Path,
    *,
    run_da: bool | None = None,
    run_mode: str | None = None,
    esp: dict | None = None,
) -> Path:
    fm = tmp_path / "fileManager.txt"
    _write_base_file_manager(fm)
    cfg = {
        "case_name": "test",
        "output_dir": str(tmp_path / "out"),
        "model_run_id": "test_run",
        "base_file_manager": str(fm),
        "forcing_file_base": "/abs/forcing_",
        "num_ens": 1,
        "start_time": "2020-01-01 00:00",
        "end_time": "2020-01-02 00:00",
        "time_delta_days": 1,
        "summa_exe": "/abs/summa",
    }
    if run_da is not None:
        cfg["run_da"] = run_da
    if run_mode is not None:
        cfg["run_mode"] = run_mode
    if esp is not None:
        cfg["esp"] = esp
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_file


def test_run_da_default_true(tmp_path: Path) -> None:
    config_file = _build_config(tmp_path)
    cfg = load_pysummada_config(config_file)
    assert cfg.run_da is True
    assert cfg.run_mode == "da"


def test_run_da_flag_can_disable(tmp_path: Path) -> None:
    config_file = _build_config(tmp_path, run_da=False)
    cfg = load_pysummada_config(config_file)
    assert cfg.run_da is False
    assert cfg.run_mode == "deterministic"


def test_run_mode_deterministic_esp_enables_esp(tmp_path: Path) -> None:
    esp_cfg = {
        "start_year": 2000,
        "end_year": 2001,
        "forecast_length_days": 30,
        "frequency_days": 30,
        "deterministic": True,
    }
    config_file = _build_config(tmp_path, run_mode="deterministic_esp", esp=esp_cfg)
    cfg = load_pysummada_config(config_file)
    assert cfg.run_mode == "deterministic_esp"
    assert cfg.run_da is False
    assert cfg.esp is not None
    assert cfg.esp.enabled is True
    assert cfg.esp.deterministic is True


def test_run_mode_da_esp_rejects_deterministic_settings(tmp_path: Path) -> None:
    esp_cfg = {
        "start_year": 2000,
        "end_year": 2001,
        "forecast_length_days": 30,
        "frequency_days": 30,
        "deterministic": True,
    }
    config_file = _build_config(tmp_path, run_mode="da_esp", esp=esp_cfg)
    with pytest.raises(ValueError, match="da_esp"):
        load_pysummada_config(config_file)
