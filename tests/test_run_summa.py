from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from snowda.workflows.run_summa import run_summa
from tests.test_pysummada_config import _build_config


def _run_with_freq(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: dict) -> dict:
    config_file = _build_config(tmp_path, extra=extra)
    captured: dict = {}

    def fake_prepare(_cfg, _run_dir):
        return SimpleNamespace(ens_file_managers=[]), SimpleNamespace()

    def fake_apply_run_time(_cfg, _assimilation, _managers, _start, _end):
        return {"001": {"file_manager": "fileManagerEns001.txt"}}

    def fake_redirect(_managers, _ens_config, output_dir):
        return output_dir

    def fake_run_ensemble(_exe, _ens_config, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("snowda.workflows.run_summa.prepare_ensemble", fake_prepare)
    monkeypatch.setattr("snowda.workflows.run_summa.apply_run_time", fake_apply_run_time)
    monkeypatch.setattr("snowda.workflows.run_summa.redirect_summa_output", fake_redirect)
    monkeypatch.setattr("snowda.workflows.run_summa.run_ensemble", fake_run_ensemble)
    run_summa(config_file)
    return captured


def test_deterministic_run_applies_summa_restart_and_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _run_with_freq(
        tmp_path,
        monkeypatch,
        {
            "summa_args": {
                "-m": "/not/used/fileManager.txt",
                "-r": "d",
                "-p": "n",
            }
        },
    )
    assert captured["freq_restart"] == "d"
    assert captured["progress"] == "n"


def test_deterministic_run_defaults_restart_to_end_and_progress_to_monthly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _run_with_freq(tmp_path, monkeypatch, {})
    assert captured["freq_restart"] == "e"
    assert captured["progress"] == "m"
