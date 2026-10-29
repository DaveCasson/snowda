from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from snowda.core.assimilation import DataAssimilationCore, restart_filename
from snowda.io.summa_ensemble import EnsembleFileWriter
from snowda.workflows.run_summa_esp import open_loop_segment_bounds, run_summa_esp
from tests.test_pysummada_config import _build_config


def test_open_loop_segment_bounds_drop_outside_and_keep_tail() -> None:
    start = datetime(2014, 2, 1)
    end = datetime(2014, 8, 1)
    bounds = open_loop_segment_bounds(
        start,
        end,
        [
            datetime(2013, 3, 1),
            datetime(2014, 3, 1),
            datetime(2014, 4, 1),
            datetime(2014, 5, 1),
            datetime(2015, 3, 1),
        ],
        None,
        None,
    )
    assert bounds == [
        (start, datetime(2014, 3, 1)),
        (datetime(2014, 3, 1), datetime(2014, 4, 1)),
        (datetime(2014, 4, 1), datetime(2014, 5, 1)),
        (datetime(2014, 5, 1), end),
    ]


def test_open_loop_segment_bounds_without_in_window_dates() -> None:
    start = datetime(2010, 2, 1)
    end = datetime(2010, 8, 1)
    bounds = open_loop_segment_bounds(start, end, [datetime(2000, 3, 1)], None, None)
    assert bounds == [(start, end)]


def test_open_loop_segment_bounds_honor_frequency_and_months() -> None:
    start = datetime(2014, 1, 1)
    end = datetime(2014, 4, 1)
    bounds = open_loop_segment_bounds(
        start,
        end,
        [datetime(2014, 2, 10)],
        frequency_days=30,
        months=(2, 3),
    )
    assert bounds == [
        (start, datetime(2014, 2, 10)),
        (datetime(2014, 2, 10), datetime(2014, 3, 2)),
        (datetime(2014, 3, 2), end),
    ]


def _esp_dates() -> dict:
    return {
        "esp": {
            "start_year": 2014,
            "end_year": 2014,
            "forecast_end": "08-01",
            "n_members": 2,
            "seed": 1,
            "dates": [
                "2014-03-01 00:00",
                "2014-04-01 00:00",
                "2014-05-01 00:00",
            ],
        },
        "start_time": "2014-02-01 00:00",
        "end_time": "2014-08-01 00:00",
    }


def _prepare_ensemble(tmp_path: Path):
    shared_states = tmp_path / "shared_states"
    shared_states.mkdir()
    (shared_states / "coldState.nc").write_bytes(b"cold")

    def prepare(cfg, _run_dir=None):
        managers = []
        for label in cfg.ens_label_array:
            path = tmp_path / f"fileManagerEns{label}.txt"
            path.write_text(
                f"statePath\t'{shared_states}/'\n"
                "initConditionFile\t'coldState.nc'\n"
                f"outFilePrefix\t'{label}'\n",
                encoding="utf-8",
            )
            managers.append(str(path))
        writer = EnsembleFileWriter(cfg)
        writer.ens_file_managers = managers
        assimilation = DataAssimilationCore(cfg)
        assimilation.create_ens_config(managers)
        return writer, assimilation

    return prepare


def _write_member_restarts(ens_config: dict, when: datetime) -> None:
    for label, entry in ens_config.items():
        state_dir = Path(entry["manager"]["statePath"])
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / restart_filename(label, when)).write_bytes(f"summa-{label}".encode())


def test_run_summa_esp_writes_da_segment_suffixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = _build_config(tmp_path, extra=_esp_dates())
    calls: list[dict] = []

    def fake_run_ensemble(_exe, ens_config, **kwargs):
        when = datetime.strptime(kwargs["run_suffix"], "%Y_%m_%d")
        _write_member_restarts(ens_config, when)
        calls.append(
            {
                "suffix": kwargs["run_suffix"],
                "freq_restart": kwargs["freq_restart"],
                "init": ens_config["001"]["manager"].get("initConditionFile"),
                "output": ens_config["001"]["manager"]["outputPath"],
                "state": ens_config["001"]["manager"]["statePath"],
            }
        )
        return {}

    def fake_traces(_cfg, _esp, run_dir, _when):
        catalog = json.loads(
            (run_dir / "assimilation" / "post_da_2014030100.json").read_text(
                encoding="utf-8"
            )
        )
        assert catalog["particles"][0]["label"] == "001"
        assert catalog["particles"][0]["donor_label"] == "001"
        assert catalog["particles"][0]["mizuroute_restart"] is None
        assert Path(catalog["particles"][0]["summa_restart"]).read_bytes() == b"summa-001"
        return []

    routed: list[bool] = []

    def fake_route(*_args, **kwargs):
        routed.append(kwargs["require_restart"])

    monkeypatch.setattr(
        "snowda.workflows.run_summa_esp.prepare_ensemble",
        _prepare_ensemble(tmp_path),
    )
    monkeypatch.setattr(
        "snowda.workflows.run_summa_esp.run_ensemble", fake_run_ensemble
    )
    monkeypatch.setattr(
        "snowda.workflows.run_summa_esp.run_balanced_esp_issue", fake_traces
    )
    monkeypatch.setattr(
        "snowda.workflows.run_summa_esp.route_ensemble_segment",
        fake_route,
    )

    run_dir = run_summa_esp(config_file)
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))

    assert [call["suffix"] for call in calls] == [
        "2014_03_01",
        "2014_04_01",
        "2014_05_01",
        "2014_08_01",
    ]
    assert {call["freq_restart"] for call in calls} == {"d"}
    assert calls[0]["init"] is None
    assert calls[1]["init"] == restart_filename("001", datetime(2014, 3, 1))
    assert calls[0]["output"] == str(run_dir / "summa") + "/"
    assert calls[0]["state"] == str(run_dir / "states") + "/"
    assert (run_dir / "summa").is_dir()
    assert (run_dir / "states" / "coldState.nc").read_bytes() == b"cold"
    assert (run_dir / "assimilation" / "post_da_2014030100.json").is_file()
    assert not (run_dir / "assimilation" / "post_da_2014080100.json").exists()
    assert routed == []
    assert provenance["variant"] == "esp"
    assert provenance["esp_n_members"] == 2
    assert provenance["esp_seed"] == 1
    assert provenance["summa_output_dir"] == str(run_dir / "summa")
    assert provenance["state_dir"] == str(run_dir / "states")
    assert "assimilation_mode" not in provenance
    assert "swe_updates" not in provenance
    assert "assimilation_dates" not in provenance


def test_run_summa_esp_routes_open_loop_when_mizuroute_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extra = _esp_dates()
    extra["num_ens"] = 2
    extra["esp"]["run_mizuroute"] = True
    extra["esp"]["mizuroute"] = {
        "executable": str(tmp_path / "route.exe"),
        "control_file": str(tmp_path / "mizuroute.control"),
        "ancil_dir": str(tmp_path / "ancil"),
        "case_name": "Ross",
        "restart_write": "monthly",
    }
    config_file = _build_config(tmp_path, extra=extra)
    routed: list[tuple[datetime, datetime, bool]] = []

    def fake_run_ensemble(_exe, ens_config, **kwargs):
        when = datetime.strptime(kwargs["run_suffix"], "%Y_%m_%d")
        _write_member_restarts(ens_config, when)
        return {}

    def fake_route(
        _mizu, run_dir, labels, _summa_dir, start, end, restarts, **kwargs
    ):
        routed.append((start, end, kwargs["require_restart"]))
        for label in labels:
            route = (
                Path(run_dir)
                / "mizuroute"
                / "restart"
                / label
                / f"Ross.r.{end.strftime('%Y-%m-%d')}-00000.nc"
            )
            route.parent.mkdir(parents=True, exist_ok=True)
            route.write_bytes(f"route-{label}".encode())
            restarts[label] = route

    monkeypatch.setattr(
        "snowda.workflows.run_summa_esp.prepare_ensemble",
        _prepare_ensemble(tmp_path),
    )
    monkeypatch.setattr(
        "snowda.workflows.run_summa_esp.run_ensemble",
        fake_run_ensemble,
    )
    monkeypatch.setattr(
        "snowda.workflows.run_summa_esp.run_balanced_esp_issue",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "snowda.workflows.run_summa_esp.route_ensemble_segment",
        fake_route,
    )

    run_dir = run_summa_esp(config_file)
    assert routed == [
        (datetime(2014, 2, 1), datetime(2014, 3, 1), True),
        (datetime(2014, 3, 1), datetime(2014, 4, 1), True),
        (datetime(2014, 4, 1), datetime(2014, 5, 1), True),
        (datetime(2014, 5, 1), datetime(2014, 8, 1), False),
    ]
    catalog = json.loads(
        (run_dir / "assimilation" / "post_da_2014030100.json").read_text(encoding="utf-8")
    )
    by_label = {item["label"]: item for item in catalog["particles"]}
    assert set(by_label) == {"001", "002"}
    for label, particle in by_label.items():
        assert particle["donor_label"] == label
        assert Path(particle["summa_restart"]).read_bytes() == f"summa-{label}".encode()
        assert Path(particle["mizuroute_restart"]).read_bytes() == f"route-{label}".encode()


def test_run_summa_esp_issues_traces_before_the_next_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extra = _esp_dates()
    extra["esp"]["dates"] = ["2014-03-01 00:00", "2014-04-01 00:00"]
    extra["end_time"] = "2014-05-01 00:00"
    config_file = _build_config(tmp_path, extra=extra)
    events: list[tuple[str, datetime]] = []

    def fake_run_ensemble(_exe, ens_config, **kwargs):
        when = datetime.strptime(kwargs["run_suffix"], "%Y_%m_%d")
        _write_member_restarts(ens_config, when)
        events.append(("ensemble", when))
        return {}

    def fake_issue(_cfg, _esp, _run_dir, when):
        events.append(("esp", when))
        return []

    monkeypatch.setattr(
        "snowda.workflows.run_summa_esp.prepare_ensemble",
        _prepare_ensemble(tmp_path),
    )
    monkeypatch.setattr(
        "snowda.workflows.run_summa_esp.run_ensemble",
        fake_run_ensemble,
    )
    monkeypatch.setattr(
        "snowda.workflows.run_summa_esp.run_balanced_esp_issue",
        fake_issue,
    )
    monkeypatch.setattr(
        "snowda.workflows.run_summa_esp.route_ensemble_segment",
        lambda *_args, **_kwargs: None,
    )

    run_summa_esp(config_file)
    assert events == [
        ("ensemble", datetime(2014, 3, 1)),
        ("esp", datetime(2014, 3, 1)),
        ("ensemble", datetime(2014, 4, 1)),
        ("esp", datetime(2014, 4, 1)),
        ("ensemble", datetime(2014, 5, 1)),
    ]
