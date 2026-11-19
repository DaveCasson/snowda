from pathlib import Path

import pytest

from snowda.config.run_dir import make_run_dir, normalize_run_name, run_name_from_config


@pytest.mark.parametrize("raw", [None, "", "timestamp", "TIMESTAMP", "null"])
def test_timestamp_run_name(raw, tmp_path: Path) -> None:
    assert normalize_run_name(raw) is None
    run_dir = make_run_dir(tmp_path / "out", raw)
    assert run_dir.parent == tmp_path / "out"
    assert run_dir.name.endswith("Z") and "T" in run_dir.name
    assert run_dir.is_dir()


def test_fixed_run_name_reused(tmp_path: Path) -> None:
    first = make_run_dir(tmp_path, "baseline")
    second = make_run_dir(tmp_path, "baseline")
    assert first == second == tmp_path / "baseline"


def test_run_name_rejects_paths() -> None:
    with pytest.raises(ValueError):
        normalize_run_name("a/b")


def test_run_name_from_config_top_level_or_paths() -> None:
    assert run_name_from_config({"run_name": "x"}) == "x"
    assert run_name_from_config({"paths": {"run_name": "y"}}) == "y"
    assert run_name_from_config({}) is None
