"""Progress logging for local SUMMA ensemble runs."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from snowda.adapters import pysumma as pysumma_adapter


class _Simulation:
    initialized = 0

    def __init__(self, executable: str, filemanager: str, initialize: bool = True) -> None:
        self.executable = executable
        self.manager_path = filemanager
        self.status = "Uninitialized"
        self.stdout = ""
        self.run_kwargs: dict | None = None

    def initialize(self) -> None:
        type(self).initialized += 1
        self.status = "Initialized"

    def apply_config(self, config: dict) -> None:
        self.config = config

    def start(self, **kwargs) -> None:
        self.run_kwargs = kwargs
        if "002" in str(self.manager_path):
            self.status = "Error"
            self.stdout = "member 2 failed"
            return
        self.status = "Success"
        self.stdout = "member 1"


class _FailingSimulation(_Simulation):
    def start(self, **kwargs) -> None:
        raise RuntimeError("worker died")


class _ProcessSimulation:
    """pysumma-shaped member whose ``monitor`` would open NetCDF."""

    def __init__(self, executable: str, filemanager: str, initialize: bool = True) -> None:
        self.manager_path = filemanager
        self.status = "Uninitialized"
        self.stdout = ""
        self._output = "not-set"
        self.opened = False

    def start(self, **kwargs) -> None:
        self.process = SimpleNamespace(
            communicate=lambda: (
                b"FORTRAN STOP: finished simulation successfully.\n",
                b"",
            )
        )
        self.status = "Running"

    def get_output_files(self) -> list[str]:
        self.opened = True
        return ["member_timestep.nc"]


def test_run_ensemble_logs_each_member(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    _Simulation.initialized = 0
    monkeypatch.setattr(
        pysumma_adapter,
        "_require_pysumma",
        lambda: SimpleNamespace(Simulation=_Simulation),
    )
    snowda_logger = logging.getLogger("snowda")
    propagate = snowda_logger.propagate
    snowda_logger.propagate = True
    try:
        with caplog.at_level(logging.INFO, logger="snowda.snowda.adapters.pysumma"):
            summary = pysumma_adapter.run_ensemble(
                "summa.exe",
                {
                    "001": {"file_manager": "fileManagerEns001.txt"},
                    "002": {"file_manager": "fileManagerEns002.txt"},
                },
                run_suffix="2014_02_01",
                log_stdout=True,
            )
    finally:
        snowda_logger.propagate = propagate

    messages = [record.getMessage() for record in caplog.records]
    assert "SUMMA ensemble starting: 2 members, 2 workers, suffix 2014_02_01" in messages
    assert any("SUMMA member 001 running" in message for message in messages)
    assert any("SUMMA member 001 complete" in message for message in messages)
    assert any("SUMMA member 002 finished with status Error" in message for message in messages)
    assert "SUMMA ensemble finished: 1/2 successful" in messages
    assert summary == {"Success": ["001"], "Error": ["002"], "Other": []}
    assert _Simulation.initialized == 0


def test_decision_edits_still_initialize_pysumma(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    _Simulation.initialized = 0
    monkeypatch.setattr(
        pysumma_adapter,
        "_require_pysumma",
        lambda: SimpleNamespace(Simulation=_Simulation),
    )
    snowda_logger = logging.getLogger("snowda")
    propagate = snowda_logger.propagate
    snowda_logger.propagate = True
    try:
        with caplog.at_level(logging.INFO, logger="snowda.snowda.adapters.pysumma"):
            pysumma_adapter.run_ensemble(
                "summa.exe",
                {
                    "001": {
                        "file_manager": "fileManagerEns001.txt",
                        "decisions": {"snowLayers": "CLM_2010"},
                    }
                },
                write_config=True,
            )
    finally:
        snowda_logger.propagate = propagate

    assert _Simulation.initialized == 1


def test_run_ensemble_logs_member_exception(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    monkeypatch.setattr(
        pysumma_adapter,
        "_require_pysumma",
        lambda: SimpleNamespace(Simulation=_FailingSimulation),
    )
    snowda_logger = logging.getLogger("snowda")
    propagate = snowda_logger.propagate
    snowda_logger.propagate = True
    try:
        with caplog.at_level(logging.INFO, logger="snowda.snowda.adapters.pysumma"):
            with pytest.raises(RuntimeError, match="worker died"):
                pysumma_adapter.run_ensemble(
                    "summa.exe",
                    {"001": {"file_manager": "fileManagerEns001.txt"}},
                )
    finally:
        snowda_logger.propagate = propagate

    messages = [record.getMessage() for record in caplog.records]
    assert any("SUMMA member 001 raised (1/1)" in message for message in messages)


def test_ensemble_workers_do_not_open_output_netcdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_ProcessSimulation] = []

    class _RecordingSimulation(_ProcessSimulation):
        def __init__(self, executable: str, filemanager: str, initialize: bool = True) -> None:
            super().__init__(executable, filemanager, initialize)
            created.append(self)

    monkeypatch.setattr(
        pysumma_adapter,
        "_require_pysumma",
        lambda: SimpleNamespace(Simulation=_RecordingSimulation),
    )
    summary = pysumma_adapter.run_ensemble(
        "summa.exe",
        {
            "001": {"file_manager": "fileManagerEns001.txt"},
            "002": {"file_manager": "fileManagerEns002.txt"},
        },
        num_workers=2,
    )

    assert summary == {"Success": ["001", "002"], "Error": [], "Other": []}
    assert created
    assert all(simulation.opened is False for simulation in created)
    assert all(simulation._output is None for simulation in created)
