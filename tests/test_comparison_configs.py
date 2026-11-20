"""The Ross comparison configs must stay aligned, or their scores are not comparable."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from snowda.config.run_modes import ASSIMILATION_MODES, FORECAST_MODES, RUN_MODES

CONFIG_DIR = Path(__file__).resolve().parents[1] / "runs" / "ross_comparison"
ENSEMBLE_KEYS = (
    "num_ens",
    "base_file_manager",
    "forcing_file_base",
    "ensemble_forcing_mode",
    "ensemble_forcing_dir",
)
FILTER_KEYS = (
    "n_eff_threshold",
    "tempering_target",
    "pf_seed",
    "season_start_month",
    "assimilation_mode",
    "assimilation",
)


@pytest.fixture(scope="module")
def configs() -> dict[str, dict]:
    loaded = {
        path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(CONFIG_DIR.glob("*.yaml"))
    }
    assert loaded, f"No comparison configs in {CONFIG_DIR}"
    return loaded


def _all_equal(values: list) -> bool:
    return all(value == values[0] for value in values)


def test_each_file_is_named_after_its_run_mode(configs: dict[str, dict]) -> None:
    for name, cfg in configs.items():
        assert cfg["run_mode"] == name
        assert name in RUN_MODES
        assert "pf_scheme" not in cfg, "run_mode sets the scheme"
    assert FORECAST_MODES <= set(configs)


def test_period_and_evaluation_are_shared(configs: dict[str, dict]) -> None:
    for key in ("start_time", "end_time", "evaluation"):
        assert _all_equal([cfg[key] for cfg in configs.values()]), key


def test_ensemble_runs_share_members_and_forcing(configs: dict[str, dict]) -> None:
    ensembles = [cfg for name, cfg in configs.items() if name != "deterministic"]
    for key in ENSEMBLE_KEYS:
        assert _all_equal([cfg[key] for cfg in ensembles]), key


def test_forecast_runs_share_the_esp_block(configs: dict[str, dict]) -> None:
    blocks = [configs[name]["esp"] for name in sorted(FORECAST_MODES)]
    assert _all_equal(blocks)
    assert blocks[0]["n_members"] == configs["esp_sis"]["num_ens"]


def test_assimilating_runs_share_the_filter_settings(configs: dict[str, dict]) -> None:
    runs = [cfg for name, cfg in configs.items() if name in ASSIMILATION_MODES]
    assert len(runs) >= 3
    for key in FILTER_KEYS:
        assert _all_equal([cfg[key] for cfg in runs]), key
    for cfg in runs:
        if "esp" in cfg:
            assert cfg["assimilation"]["dates"] == cfg["esp"]["dates"]
