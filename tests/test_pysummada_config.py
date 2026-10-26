from pathlib import Path

import pytest
import yaml

from snowda.config.pysummada import (
    load_pysummada_config,
    validate_da_config,
    validate_deterministic_config,
    validate_esp_config,
)


def _write_base_file_manager(path: Path) -> None:
    content = """outputPath	'/tmp/output/'
initConditionFile	'coldState.nc'
simStartTime	'2011-01-01 01:00'
simEndTime	'2018-12-31 23:00'
"""
    path.write_text(content, encoding="utf-8")


def _esp_block(**overrides: object) -> dict:
    esp = {
        "start_year": 2000,
        "end_year": 2001,
        "forecast_length_days": 30,
        "frequency_days": 30,
        "n_members": 2,
        "seed": 1,
    }
    esp.update(overrides)
    return esp


def _build_config(
    tmp_path: Path,
    *,
    extra: dict | None = None,
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
    if extra:
        cfg.update(extra)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_file


def test_load_without_variant_keys(tmp_path: Path) -> None:
    config_file = _build_config(tmp_path)
    cfg = load_pysummada_config(config_file)
    assert cfg.esp is None
    assert cfg.mizuroute is None
    assert cfg.pf_scheme == "sis"
    assert cfg.num_workers == 5
    validate_deterministic_config(cfg)


def test_num_workers_comes_from_config(tmp_path: Path) -> None:
    config_file = _build_config(tmp_path, extra={"num_workers": 10})
    cfg = load_pysummada_config(config_file)
    assert cfg.num_workers == 10


def test_num_workers_rejects_zero(tmp_path: Path) -> None:
    config_file = _build_config(tmp_path, extra={"num_workers": 0})
    with pytest.raises(ValueError, match="num_workers"):
        load_pysummada_config(config_file)


def test_pf_scheme_accepts_sir(tmp_path: Path) -> None:
    config_file = _build_config(tmp_path, extra={"pf_scheme": "SIR"})
    cfg = load_pysummada_config(config_file)
    assert cfg.pf_scheme == "sir"


def test_pf_scheme_rejects_unknown(tmp_path: Path) -> None:
    config_file = _build_config(tmp_path, extra={"pf_scheme": "bootstrap"})
    with pytest.raises(ValueError, match="pf_scheme"):
        load_pysummada_config(config_file)


def test_pf_resample_is_rejected(tmp_path: Path) -> None:
    config_file = _build_config(tmp_path, extra={"pf_resample": True})
    with pytest.raises(ValueError, match="pf_scheme"):
        load_pysummada_config(config_file)


def test_run_mode_is_ignored(tmp_path: Path) -> None:
    config_file = _build_config(tmp_path, extra={"run_mode": "da_esp"})
    cfg = load_pysummada_config(config_file)
    assert not hasattr(cfg, "run_mode")
    validate_deterministic_config(cfg)


def test_esp_implicitly_enabled_without_enabled_flag(tmp_path: Path) -> None:
    config_file = _build_config(tmp_path, extra={"esp": _esp_block()})
    cfg = load_pysummada_config(config_file)
    assert cfg.esp is not None
    assert cfg.esp.enabled is True
    validate_esp_config(cfg)


def test_deterministic_rejects_enabled_esp(tmp_path: Path) -> None:
    config_file = _build_config(tmp_path, extra={"esp": _esp_block()})
    cfg = load_pysummada_config(config_file)
    with pytest.raises(ValueError, match="does not run ESP"):
        validate_deterministic_config(cfg)


def test_esp_requires_esp_section(tmp_path: Path) -> None:
    config_file = _build_config(tmp_path)
    cfg = load_pysummada_config(config_file)
    with pytest.raises(ValueError, match="requires an `esp` section"):
        validate_esp_config(cfg)


def test_esp_rejects_fsca_obs(tmp_path: Path) -> None:
    config_file = _build_config(
        tmp_path,
        extra={"esp": _esp_block(), "fsca_obs_nc": "/abs/fsca.nc"},
    )
    cfg = load_pysummada_config(config_file)
    with pytest.raises(ValueError, match="does not run data assimilation"):
        validate_esp_config(cfg)


def test_da_requires_esp_and_fsca_fields(tmp_path: Path) -> None:
    config_file = _build_config(tmp_path, extra={"esp": _esp_block()})
    cfg = load_pysummada_config(config_file)
    with pytest.raises(ValueError, match="fsca_obs_nc"):
        validate_da_config(cfg)


def test_da_rejects_deterministic_esp_settings(tmp_path: Path) -> None:
    config_file = _build_config(
        tmp_path,
        extra={
            "esp": _esp_block(deterministic=True),
            "fsca_obs_nc": "/abs/fsca.nc",
            "summa_output_dir": "/abs/summa_out",
            "fsca_output_dir": str(tmp_path / "fsca"),
        },
    )
    cfg = load_pysummada_config(config_file)
    with pytest.raises(ValueError, match="does not allow"):
        validate_da_config(cfg)


def test_open_loop_esp_requires_n_members_and_seed(tmp_path: Path) -> None:
    block = _esp_block()
    block.pop("n_members")
    block.pop("seed")
    config_file = _build_config(tmp_path, extra={"esp": block})
    cfg = load_pysummada_config(config_file)
    with pytest.raises(ValueError, match="n_members"):
        validate_esp_config(cfg)


def test_open_loop_esp_accepts_balanced_initialization(tmp_path: Path) -> None:
    config_file = _build_config(
        tmp_path,
        extra={
            "esp": _esp_block(
                initialization="balanced_independent",
                n_members=4,
                seed=1,
            )
        },
    )
    cfg = load_pysummada_config(config_file)
    validate_esp_config(cfg)
    assert cfg.esp is not None
    assert cfg.esp.n_members == 4
    assert cfg.esp.seed == 1


def test_da_balanced_initialization_requires_size_and_seed(tmp_path: Path) -> None:
    config_file = _build_config(
        tmp_path,
        extra={
            "esp": _esp_block(
                initialization="balanced_independent", seed=1, n_members=None
            ),
            "fsca_obs_nc": "/abs/fsca.nc",
            "summa_output_dir": "/abs/summa_out",
            "fsca_output_dir": str(tmp_path / "fsca"),
        },
    )
    cfg = load_pysummada_config(config_file)
    with pytest.raises(ValueError, match="n_members"):
        validate_da_config(cfg)


def test_da_balanced_initialization_loads(tmp_path: Path) -> None:
    config_file = _build_config(
        tmp_path,
        extra={
            "esp": _esp_block(
                initialization="balanced_independent",
                n_members=4,
                seed=3,
            ),
            "fsca_obs_nc": "/abs/fsca.nc",
            "summa_output_dir": "/abs/summa_out",
            "fsca_output_dir": str(tmp_path / "fsca"),
        },
    )
    cfg = load_pysummada_config(config_file)
    validate_da_config(cfg)
    assert cfg.esp is not None
    assert cfg.esp.initialization == "balanced_independent"
    assert cfg.esp.n_members == 4
    assert cfg.esp.seed == 3


def test_esp_seed_rejects_negative(tmp_path: Path) -> None:
    config_file = _build_config(
        tmp_path,
        extra={"esp": _esp_block(seed=-1)},
    )
    with pytest.raises(ValueError, match="seed"):
        load_pysummada_config(config_file)


def test_da_accepts_complete_config(tmp_path: Path) -> None:
    config_file = _build_config(
        tmp_path,
        extra={
            "esp": _esp_block(),
            "fsca_obs_nc": "/abs/fsca.nc",
            "summa_output_dir": "/abs/summa_out",
            "fsca_output_dir": str(tmp_path / "fsca"),
        },
    )
    cfg = load_pysummada_config(config_file)
    validate_da_config(cfg)


def test_top_level_mizuroute(tmp_path: Path) -> None:
    mizu = {
        "executable": str(tmp_path / "route.exe"),
        "control_file": str(tmp_path / "mizuroute.control"),
        "ancil_dir": str(tmp_path / "ancil"),
        "case_name": "case",
        "restart_write": "monthly",
        "log_output": True,
    }
    config_file = _build_config(tmp_path, extra={"mizuroute": mizu})
    cfg = load_pysummada_config(config_file)
    assert cfg.mizuroute is not None
    assert cfg.mizuroute.case_name == "case"
    assert cfg.mizuroute.log_output is True
    assert not hasattr(cfg.mizuroute, "output_dir")
    validate_deterministic_config(cfg)


def test_disabled_esp_mizuroute_fallback(tmp_path: Path) -> None:
    config_file = _build_config(
        tmp_path,
        extra={
            "esp": {
                "enabled": False,
                "run_mizuroute": True,
                "mizuroute": {
                    "executable": str(tmp_path / "route.exe"),
                    "control_file": str(tmp_path / "mizuroute.control"),
                    "ancil_dir": str(tmp_path / "ancil"),
                    "case_name": "Ross",
                    "restart_write": "monthly",
                },
            }
        },
    )
    cfg = load_pysummada_config(config_file)
    assert cfg.esp is not None
    assert cfg.esp.enabled is False
    assert cfg.mizuroute is not None
    assert cfg.mizuroute.case_name == "Ross"
    validate_deterministic_config(cfg)


def test_run_pysummada_pf_redirects() -> None:
    from snowda.workflows.run_pysummada_pf import run_pysummada_pf

    with pytest.raises(RuntimeError, match="run_summa_da"):
        run_pysummada_pf("unused.yaml")


def test_ensemble_file_managers_write_yaml_times(tmp_path: Path) -> None:
    from snowda.io.summa_ensemble import EnsembleFileWriter

    config_file = _build_config(tmp_path)
    cfg = load_pysummada_config(config_file)
    writer = EnsembleFileWriter(cfg)
    paths = writer.write_ens_file_managers()
    ens_text = Path(paths[0]).read_text(encoding="utf-8")
    base_text = cfg.base_file_manager.read_text(encoding="utf-8")
    for text in (ens_text, base_text):
        assert "2020-01-01 00:00" in text
        assert "2020-01-02 00:00" in text
        assert "2011-01-01 01:00" not in text
        assert "2018-12-31 23:00" not in text


def test_output_dir_defaults_to_outputs_case_name(tmp_path: Path) -> None:
    from snowda.config import pysummada as pysummada_mod

    fm = tmp_path / "fileManager.txt"
    _write_base_file_manager(fm)
    cfg = {
        "case_name": "ross_summa_deterministic",
        "model_run_id": "test_run",
        "base_file_manager": str(fm),
        "forcing_file_base": "/abs/forcing_",
        "num_ens": 1,
        "start_time": "2020-01-01 00:00",
        "end_time": "2020-01-02 00:00",
        "time_delta_days": 1,
        "summa_exe": "/abs/summa",
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    loaded = load_pysummada_config(config_file)
    repo_root = Path(pysummada_mod.__file__).resolve().parents[3]
    assert loaded.output_dir == repo_root / "outputs" / "ross_summa_deterministic"


def test_redirect_summa_output_updates_file_manager(tmp_path: Path) -> None:
    from snowda.io.summa_ensemble import EnsembleFileWriter
    from snowda.workflows.summa_common import (
        redirect_summa_output,
        summa_run_output_dir,
    )

    from snowda.io.summa_ensemble import update_file_manager_option

    config_file = _build_config(tmp_path)
    cfg = load_pysummada_config(config_file)
    writer = EnsembleFileWriter(cfg)
    paths = writer.write_ens_file_managers()
    shared_states = tmp_path / "shared_states"
    shared_states.mkdir()
    (shared_states / "coldState.nc").write_bytes(b"cold")
    for path in paths:
        update_file_manager_option(path, "statePath", f"'{shared_states}/'")
    run_dir = tmp_path / "20260101T000000Z"
    ens_config = {"001": {"file_manager": paths[0], "manager": {}}}
    summa_dir = redirect_summa_output(paths, ens_config, summa_run_output_dir(run_dir))
    state_dir = run_dir / "states"
    assert summa_dir == run_dir / "summa"
    assert summa_dir.is_dir()
    assert (state_dir / "coldState.nc").read_bytes() == b"cold"
    fm_text = Path(paths[0]).read_text(encoding="utf-8")
    assert str(summa_dir) in fm_text
    assert str(state_dir) in fm_text
    assert "/tmp/output/" not in fm_text
    assert str(shared_states) not in fm_text
    assert ens_config["001"]["manager"]["outputPath"] == str(summa_dir) + "/"
    assert ens_config["001"]["manager"]["statePath"] == str(state_dir) + "/"


def test_mizuroute_yaml_output_dirs_are_ignored(tmp_path: Path) -> None:
    mizu = {
        "executable": str(tmp_path / "route.exe"),
        "control_file": str(tmp_path / "mizuroute.control"),
        "ancil_dir": str(tmp_path / "ancil"),
        "output_dir": str(tmp_path / "legacy_output"),
        "restart_dir": str(tmp_path / "legacy_restart"),
        "case_name": "case",
    }
    config_file = _build_config(tmp_path, extra={"mizuroute": mizu})
    cfg = load_pysummada_config(config_file)
    assert cfg.mizuroute is not None
    assert not hasattr(cfg.mizuroute, "output_dir")
    assert not hasattr(cfg.mizuroute, "restart_dir")


def test_load_streamflow_evaluation(tmp_path: Path) -> None:
    obs = tmp_path / "streamflow.csv"
    obs.write_text("datetime,discharge_cms\n2000-01-01,1.0\n", encoding="utf-8")
    mizu = {
        "executable": "/abs/route.exe",
        "control_file": "/abs/mizuroute.control",
        "ancil_dir": "/abs/ancil",
        "case_name": "Ross",
    }
    config_file = _build_config(
        tmp_path,
        extra={
            "mizuroute": mizu,
            "evaluation": {
                "obs_csv": str(obs),
                "resample": "1D",
                "seg": 0,
            },
        },
    )
    cfg = load_pysummada_config(config_file)
    assert cfg.evaluation is not None
    assert cfg.evaluation.obs_csv == obs.resolve()
    assert cfg.evaluation.variable == "IRFroutedRunoff"
    assert cfg.evaluation.resample == "1D"
    validate_deterministic_config(cfg)


def test_evaluation_requires_mizuroute(tmp_path: Path) -> None:
    obs = tmp_path / "streamflow.csv"
    obs.write_text("datetime,discharge_cms\n2000-01-01,1.0\n", encoding="utf-8")
    config_file = _build_config(
        tmp_path,
        extra={"evaluation": {"obs_csv": str(obs)}},
    )
    cfg = load_pysummada_config(config_file)
    with pytest.raises(ValueError, match="requires top-level `mizuroute`"):
        validate_deterministic_config(cfg)


def test_esp_forecast_end_loads(tmp_path: Path) -> None:
    config_file = _build_config(
        tmp_path,
        extra={"esp": _esp_block(forecast_length_days=None, forecast_end="08-01")},
    )
    cfg = load_pysummada_config(config_file)
    assert cfg.esp is not None
    assert cfg.esp.forecast_length_days is None
    assert cfg.esp.forecast_end == (8, 1)
    validate_esp_config(cfg)


def test_esp_rejects_both_horizon_options(tmp_path: Path) -> None:
    config_file = _build_config(
        tmp_path,
        extra={"esp": _esp_block(forecast_end="08-01")},
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_pysummada_config(config_file)


def test_esp_rejects_neither_horizon_option(tmp_path: Path) -> None:
    config_file = _build_config(
        tmp_path,
        extra={"esp": _esp_block(forecast_length_days=None)},
    )
    with pytest.raises(ValueError, match="either `esp.forecast_length_days` or `esp.forecast_end`"):
        load_pysummada_config(config_file)


@pytest.mark.parametrize("raw", ["13-01", "08/01", "8-1", "02-30"])
def test_esp_rejects_invalid_forecast_end(tmp_path: Path, raw: str) -> None:
    config_file = _build_config(
        tmp_path,
        extra={"esp": _esp_block(forecast_length_days=None, forecast_end=raw)},
    )
    with pytest.raises(ValueError, match="MM-DD"):
        load_pysummada_config(config_file)
