"""The Ross comparison configs must stay aligned, or their scores are not comparable."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from snowda.config.run_modes import ASSIMILATION_MODES, FORECAST_MODES, RUN_MODES

RUNS = Path(__file__).resolve().parents[1] / "runs"
CONFIG_DIRS = {
    "local": RUNS / "ross_comparison",
    "anvil": RUNS / "ross_comparison_anvil",
}
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


def _load(folder: Path) -> dict[str, dict]:
    loaded = {
        path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(folder.glob("*.yaml"))
    }
    assert loaded, f"No comparison configs in {folder}"
    return loaded


@pytest.fixture(scope="module", params=sorted(CONFIG_DIRS))
def configs(request: pytest.FixtureRequest) -> dict[str, dict]:
    return _load(CONFIG_DIRS[request.param])


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
        # `forcing_file_base` is optional with historic ensemble forcing.
        assert _all_equal([cfg.get(key) for cfg in ensembles]), key


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


def _without_paths(value):
    """Drop path-valued settings, which differ between machines."""
    if isinstance(value, dict):
        return {
            key: _without_paths(item)
            for key, item in value.items()
            if not (isinstance(item, str) and item.startswith(("/", "..")))
        }
    if isinstance(value, list):
        return [_without_paths(item) for item in value]
    return value


def test_anvil_set_matches_the_local_set_apart_from_paths() -> None:
    local = _load(CONFIG_DIRS["local"])
    anvil = _load(CONFIG_DIRS["anvil"])
    assert set(local) == set(anvil)
    for name in local:
        ours = _without_paths(local[name])
        theirs = _without_paths(anvil[name])
        # One CPU per member on Anvil; the laptop has fewer cores.
        ours.pop("num_workers", None)
        theirs.pop("num_workers", None)
        assert ours == theirs, name
        assert "/Users/" not in (CONFIG_DIRS["anvil"] / f"{name}.yaml").read_text()
