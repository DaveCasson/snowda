"""mizuRoute subprocess failures keep the program's own output."""

from __future__ import annotations

import subprocess

import pytest

from snowda.adapters import mizuroute


def _completed(returncode: int, stdout: bytes, stderr: bytes) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        args=["route_runoff.exe", "case.control"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_failure_message_includes_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mizuroute.subprocess,
        "run",
        lambda *args, **kwargs: _completed(1, b"FATAL ERROR: not enough slots\n", b""),
    )

    with pytest.raises(RuntimeError, match="FATAL ERROR: not enough slots") as exc_info:
        mizuroute.run_mizuroute("route_runoff.exe", "case.control")

    message = str(exc_info.value)
    assert "exit 1" in message
    assert "stderr:\n(empty)" in message


def test_launch_drops_slurm_step_id(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, dict[str, str] | None] = {}

    def _run(*args, **kwargs):
        seen["env"] = kwargs.get("env")
        return _completed(0, b"SUCCESSFUL EXECUTION\n", b"")

    monkeypatch.setenv("SLURM_JOBID", "999")
    monkeypatch.setenv("SLURM_NODELIST", "a013")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("SLURM_STEPID", "0")
    monkeypatch.setattr(mizuroute.subprocess, "run", _run)

    mizuroute.run_mizuroute("route_runoff.exe", "case.control", log_stdout=False)

    env = seen["env"]
    assert env is not None
    assert "SLURM_STEP_ID" not in env
    assert "SLURM_STEPID" not in env
    assert env["SLURM_NODELIST"] == "a013"
    assert env["SLURM_JOBID"] == "999"


def test_success_returns_completed_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mizuroute.subprocess,
        "run",
        lambda *args, **kwargs: _completed(0, b"SUCCESSFUL EXECUTION\n", b""),
    )

    completed = mizuroute.run_mizuroute("route_runoff.exe", "case.control", log_stdout=False)

    assert completed.returncode == 0
