from __future__ import annotations

import json
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from snowda.adapters.pysumma import netcdf_lock
from snowda.core.assimilation import DataAssimilationCore, restart_filename
from snowda.core.esp_sampling import (
    ParticleState,
    balanced_indices,
    sample_forecast_members,
    systematic_indices,
    validate_particle_binding,
    write_post_da_catalog,
)
from snowda.core.particle_filter import ParticleFilterCore
from snowda.workflows.esp import run_balanced_esp_traces
from snowda.workflows.run_summa_da import _resample_member_states
from tests.test_pysummada_config import _build_config


def _particle(
    label: str,
    *,
    weight: float = 1.0,
    resampled: bool = False,
    summa: str | None = None,
    route: str | None = None,
) -> ParticleState:
    return ParticleState(
        label=label,
        donor_label=label,
        summa_restart=summa or f"{label}_restart_2014030100_2014_03_01.nc",
        mizuroute_restart=route or f"{label}/Ross.r.2014-03-01-00000.nc",
        weight=weight,
        file_manager=f"fileManagerEns{label}.txt",
        resampled=resampled,
    )


def test_balanced_sample_has_n_pairs_and_floor_ceil_counts() -> None:
    particles = [_particle(label) for label in ("001", "002", "003")]
    years = [2010, 2011, 2012, 2013]
    assignments = sample_forecast_members(
        particles,
        years,
        n_members=7,
        seed=11,
        issue_key="2014030100",
    )
    assert len(assignments) == 7
    assert [item.forecast_id for item in assignments] == [f"m{i:03d}" for i in range(1, 8)]
    state_counts = Counter(item.particle.label for item in assignments)
    weather_counts = Counter(item.trace_year for item in assignments)
    assert set(state_counts) <= {"001", "002", "003"}
    assert set(state_counts.values()) <= {2, 3}
    assert sum(state_counts.values()) == 7
    assert set(weather_counts.values()) <= {1, 2}
    assert sum(weather_counts.values()) == 7
    for item in assignments:
        assert item.particle.summa_restart.startswith(item.particle.label)
        assert item.particle.mizuroute_restart is not None
        assert item.particle.mizuroute_restart.startswith(f"{item.particle.label}/")


def test_same_seed_reproduces_the_mapping() -> None:
    particles = [_particle(label) for label in ("001", "002")]
    years = [2000, 2001, 2002]
    kwargs = {
        "n_members": 5,
        "seed": 19,
        "issue_key": "2014050100",
    }
    first = sample_forecast_members(particles, years, **kwargs)
    second = sample_forecast_members(particles, years, **kwargs)
    assert [(item.particle.label, item.trace_year) for item in first] == [
        (item.particle.label, item.trace_year) for item in second
    ]


def test_state_and_weather_streams_are_not_the_same_seed() -> None:
    seed = 4
    issue_key = 2014030100
    n_pool = 5
    n_draw = 8
    same = balanced_indices(n_pool, n_draw, np.random.default_rng(seed))
    again = balanced_indices(n_pool, n_draw, np.random.default_rng(seed))
    assert np.array_equal(same, again)

    sequence = np.random.SeedSequence([seed, issue_key])
    state_rng, weather_rng = (
        np.random.default_rng(child) for child in sequence.spawn(2)
    )
    state_draw = balanced_indices(n_pool, n_draw, state_rng)
    weather_draw = balanced_indices(n_pool, n_draw, weather_rng)
    assert not np.array_equal(state_draw, weather_draw)

    particles = [_particle(str(index)) for index in range(n_pool)]
    years = list(range(2000, 2000 + n_pool))
    assignments = sample_forecast_members(
        particles,
        years,
        n_members=n_draw,
        seed=seed,
        issue_key=str(issue_key),
    )
    assert [int(item.particle.label) for item in assignments] == state_draw.tolist()
    assert [item.trace_year - 2000 for item in assignments] == weather_draw.tolist()


def test_duplicate_state_weather_pairs_are_kept() -> None:
    particles = [_particle(label) for label in ("001", "002")]
    years = [2000, 2001]
    assignments = sample_forecast_members(
        particles,
        years,
        n_members=8,
        seed=2,
        issue_key="2014030100",
    )
    assert len(assignments) == 8


def test_small_pool_is_sampled_without_replacement() -> None:
    particles = [_particle(str(index)) for index in range(5)]
    assignments = sample_forecast_members(
        particles,
        [1990, 1991, 1992, 1993, 1994],
        n_members=3,
        seed=8,
        issue_key="2014030100",
    )
    labels = [item.particle.label for item in assignments]
    years = [item.trace_year for item in assignments]
    assert len(labels) == len(set(labels)) == 3
    assert len(years) == len(set(years)) == 3


def test_systematic_resampling_respects_posterior_weights() -> None:
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    particles = [
        _particle(label, weight=weight, resampled=False) for label, weight in weights.items()
    ]
    for seed in (1, 7, 21):
        assignments = sample_forecast_members(
            particles,
            [2000, 2001],
            n_members=100,
            seed=seed,
            issue_key="2014030100",
        )
        counts = Counter(item.particle.label for item in assignments)
        for label, weight in weights.items():
            assert abs(counts[label] - 100 * weight) < 1
        assert all(item.weighting == "systematic" for item in assignments)
        assert all(
            item.particle.mizuroute_restart is not None
            and item.particle.mizuroute_restart.startswith(f"{item.particle.label}/")
            for item in assignments
        )


def test_resampled_particles_are_not_weighted_again() -> None:
    particles = [
        _particle("001", weight=0.9, resampled=True),
        _particle("002", weight=0.1, resampled=True),
    ]
    assignments = sample_forecast_members(
        particles,
        [2000, 2001, 2002, 2003],
        n_members=5,
        seed=3,
        issue_key="2014030100",
    )
    counts = Counter(item.particle.label for item in assignments)
    assert set(counts.values()) <= {2, 3}
    assert all(item.weighting == "equal" for item in assignments)


def test_sampling_rejects_empty_pools_and_bad_weights() -> None:
    particles = [_particle("001")]
    with pytest.raises(ValueError, match="empty"):
        sample_forecast_members(particles, [], n_members=1, seed=1, issue_key="2014030100")
    with pytest.raises(ValueError, match="empty"):
        sample_forecast_members([], [2000], n_members=1, seed=1, issue_key="2014030100")
    with pytest.raises(ValueError, match="n_members"):
        sample_forecast_members(particles, [2000], n_members=0, seed=1, issue_key="2014030100")
    with pytest.raises(ValueError, match="non-negative"):
        systematic_indices(np.array([0.2, -0.1]), 4, np.random.default_rng(1))
    with pytest.raises(ValueError, match="finite"):
        systematic_indices(np.array([np.nan, 1.0]), 4, np.random.default_rng(1))
    with pytest.raises(ValueError, match="positive"):
        systematic_indices(np.array([0.0, 0.0]), 4, np.random.default_rng(1))


def _restart_files(tmp_path: Path, when: datetime) -> tuple[dict[str, Path], dict[str, bytes]]:
    state_dir = tmp_path / "run" / "states"
    state_dir.mkdir(parents=True)
    route_root = tmp_path / "run" / "mizuroute" / "restart"
    originals: dict[str, bytes] = {}
    routes: dict[str, Path] = {}
    for label, payload in (("001", b"state-001"), ("002", b"state-002")):
        summa = state_dir / restart_filename(label, when)
        summa.write_bytes(payload)
        route = route_root / label / "Ross.r.2014-03-01-00000.nc"
        route.parent.mkdir(parents=True)
        route.write_bytes(b"route-" + label.encode())
        routes[label] = route
        originals[label] = payload
    return routes, originals


def test_sir_catalog_binds_summa_and_mizuroute_from_the_same_donor(tmp_path: Path) -> None:
    when = datetime(2014, 3, 1)
    routes, summa_bytes = _restart_files(tmp_path, when)
    route_bytes = {label: path.read_bytes() for label, path in routes.items()}
    config_file = _build_config(
        tmp_path,
        extra={
            "pf_scheme": "sir",
            "num_ens": 2,
            "n_eff_threshold": 0.5,
            "mizuroute": {
                "executable": str(tmp_path / "route.exe"),
                "control_file": str(tmp_path / "mizuroute.control"),
                "ancil_dir": str(tmp_path / "ancil"),
                "case_name": "Ross",
                "restart_write": "monthly",
            },
        },
    )
    from snowda.config.pysummada import load_pysummada_config

    cfg = load_pysummada_config(config_file)
    managers = []
    for label in cfg.ens_label_array:
        path = tmp_path / f"fileManagerEns{label}.txt"
        path.write_text("initConditionFile\t'coldState.nc'\n", encoding="utf-8")
        managers.append(str(path))
    weights = pd.DataFrame([[0.8, 0.2]], columns=cfg.ens_label_array)
    copied = _resample_member_states(
        cfg,
        DataAssimilationCore(cfg),
        managers,
        ParticleFilterCore(),
        weights,
        when,
        cfg.ens_label_array,
        routes,
        run_dir=tmp_path / "run",
    )
    assert copied is True
    catalog = json.loads(
        (tmp_path / "run" / "assimilation" / "post_da_2014030100.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(catalog["particles"]) == 2
    for particle in catalog["particles"]:
        donor = particle["donor_label"]
        assert Path(particle["summa_restart"]).name.startswith(f"{donor}_restart_")
        assert Path(particle["summa_restart"]).read_bytes() == summa_bytes[donor]
        assert Path(particle["mizuroute_restart"]).read_bytes() == route_bytes[donor]
        assert particle["resampled"] is True


def test_binding_rejects_disagreement_between_restart_dates(tmp_path: Path) -> None:
    when = datetime(2014, 3, 1)
    summa = tmp_path / "001_restart_2014040100_2014_04_01.nc"
    route = tmp_path / "Ross.r.2014-03-01-00000.nc"
    summa.write_bytes(b"summa")
    route.write_bytes(b"route")
    particle = _particle("001", summa=str(summa), route=str(route))
    with pytest.raises(ValueError, match="does not match|not the forecast issue"):
        validate_particle_binding(particle, when, require_routing=True)


def test_balanced_esp_smoke_copies_bound_restarts_without_mutating_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    when = datetime(2014, 3, 1)
    run_dir = tmp_path / "run"
    state_dir = run_dir / "states"
    state_dir.mkdir(parents=True)
    ancil = tmp_path / "ancil"
    ancil.mkdir()
    (ancil / "param.nml.default").write_text("&params\n/\n", encoding="utf-8")
    control = ancil / "mizuroute.control"
    control.write_text(
        "<fname_state_in>          cold.nc\n"
        "<fname_qsim>              placeholder.nc\n"
        "<case_name>               Ross\n"
        "<sim_start>               2000-01-01 00:00:00\n"
        "<sim_end>                 2000-01-02 00:00:00\n"
        "<ancil_dir>               /\n"
        "<input_dir>               /\n"
        "<output_dir>              /\n"
        "<restart_dir>             /\n"
        "<restart_write>           monthly\n",
        encoding="utf-8",
    )
    base_fm = tmp_path / "fileManager.txt"
    config_file = _build_config(
        tmp_path,
        extra={
            "num_ens": 2,
            "start_time": "2014-02-01 00:00",
            "end_time": "2014-08-01 00:00",
            "esp": {
                "initialization": "balanced_independent",
                "n_members": 4,
                "seed": 7,
                "start_year": 2014,
                "end_year": 2015,
                "forecast_length_days": 10,
                "dates": ["2014-03-01 00:00"],
                "run_mizuroute": True,
                "mizuroute": {
                    "executable": str(tmp_path / "route.exe"),
                    "control_file": str(control),
                    "ancil_dir": str(ancil),
                    "case_name": "Ross",
                    "restart_write": "monthly",
                },
            },
        },
    )
    base_fm.write_text(
        "outputPath\t'/tmp/output/'\n"
        "initConditionFile\t'coldState.nc'\n"
        "forcingPath\t'/forcing/base/'\n"
        "forcingListFile\t'base_forcing.txt'\n"
        "simStartTime\t'2014-02-01 00:00'\n"
        "simEndTime\t'2014-08-01 00:00'\n",
        encoding="utf-8",
    )
    particles: list[ParticleState] = []
    originals: dict[str, tuple[bytes, bytes]] = {}
    for label in ("001", "002"):
        summa = state_dir / restart_filename(label, when)
        route = run_dir / "mizuroute" / "restart" / label / "Ross.r.2014-03-01-00000.nc"
        route.parent.mkdir(parents=True)
        summa_bytes = f"summa-{label}".encode()
        route_bytes = f"route-{label}".encode()
        summa.write_bytes(summa_bytes)
        route.write_bytes(route_bytes)
        manager = tmp_path / f"fileManagerEns{label}.txt"
        manager.write_text(
            f"forcingPath\t'/forcing/{label}/'\n"
            f"forcingListFile\t'forcingFileListEns{label}.txt'\n"
            f"outFilePrefix\t'{label}'\n",
            encoding="utf-8",
        )
        particles.append(
            ParticleState(
                label=label,
                donor_label=label,
                summa_restart=str(summa),
                mizuroute_restart=str(route),
                weight=0.5,
                file_manager=str(manager),
                resampled=False,
            )
        )
        originals[label] = (summa_bytes, route_bytes)
    write_post_da_catalog(run_dir, when, particles)

    calls: list[dict] = []

    def fake_summa(_exe, file_manager, **kwargs):
        manager = kwargs["sim_config"]["manager"]
        output = Path(manager["outputPath"])
        output.mkdir(parents=True, exist_ok=True)
        dataset = xr.Dataset(
            {"scalarSWE": ("time", [1.0, 2.0])},
            coords={"time": pd.to_datetime(["2015-03-01", "2015-03-02"])},
        )
        # Workers run together. This stand-in still writes NetCDF, so it
        # takes the same lock as the time-shift.
        with netcdf_lock:
            dataset.to_netcdf(output / f"{manager['outFilePrefix']}_timestep.nc")
            dataset.close()
        calls.append({"file_manager": file_manager, "manager": manager})

    def fake_route(_executable, control_file, log_stdout=False) -> None:
        from snowda.adapters.mizuroute import read_control_value

        text = Path(control_file).read_text(encoding="utf-8")
        output_dir = Path(read_control_value(control_file, "output_dir").rstrip("/"))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "Ross.h.2014-03-01-00000.nc").write_bytes(b"routed")
        calls.append(
            {
                "control": text,
                "fname_state_in": read_control_value(control_file, "fname_state_in"),
                "restart_dir": read_control_value(control_file, "restart_dir"),
            }
        )

    monkeypatch.setattr("snowda.workflows.esp.run_simulation", fake_summa)
    monkeypatch.setattr("snowda.workflows.esp.run_mizuroute", fake_route)

    from snowda.config.pysummada import load_pysummada_config

    cfg = load_pysummada_config(config_file)
    assert cfg.esp is not None
    records = run_balanced_esp_traces(cfg, cfg.esp, run_dir)

    assert len(records) == 4
    member_map = json.loads((run_dir / "esp" / "member_map.json").read_text(encoding="utf-8"))
    assert member_map["seed"] == 7
    assert member_map["n_members"] == 4
    assert "independently balancing" in member_map["description"]
    mapped = member_map["issues"][0]["members"]
    assert len(mapped) == 4
    assert [record.state_member for record in records] == [
        member["state_label"] for member in mapped
    ]
    assert [record.trace_year for record in records] == [
        member["trace_year"] for member in mapped
    ]
    summa_calls = [call for call in calls if "manager" in call]
    route_calls = [call for call in calls if "fname_state_in" in call]
    assert len(summa_calls) == 4
    assert len(route_calls) == 4
    for member in mapped:
        label = member["state_label"]
        assert Path(member["summa_restart"]).read_bytes() == originals[label][0]
        assert Path(member["mizuroute_restart"]).read_bytes() == originals[label][1]
        assert member["donor_label"] == label
    for call in summa_calls:
        assert call["manager"]["forcingPath"] == "/forcing/base/"
        assert call["manager"]["forcingListFile"] == "base_forcing.txt"
        assert "/forcing/001/" not in call["manager"]["forcingPath"]
        assert Path(call["file_manager"]).name.startswith("fileManagerEns")
        private_summa = Path(call["manager"]["statePath"]) / call["manager"]["initConditionFile"]
        assert "esp_states/forecast" in str(private_summa)
        assert private_summa.is_file()
    for call in route_calls:
        assert "esp_states/forecast" in call["restart_dir"]
        private_route = Path(call["restart_dir"].rstrip("/")) / call["fname_state_in"]
        assert private_route.is_file()

    by_label: dict[str, list[str]] = {}
    for member in mapped:
        by_label.setdefault(member["state_label"], []).append(member["forecast_id"])
    shared_label = next(label for label, ids in by_label.items() if len(ids) >= 2)
    first_id, second_id = by_label[shared_label][:2]
    first_copy = (
        run_dir
        / "esp_states"
        / "forecast"
        / "2014030100"
        / first_id
        / "summa"
        / restart_filename(shared_label, when)
    )
    second_copy = (
        run_dir
        / "esp_states"
        / "forecast"
        / "2014030100"
        / second_id
        / "summa"
        / restart_filename(shared_label, when)
    )
    source = state_dir / restart_filename(shared_label, when)
    first_copy.write_bytes(b"mutated")
    assert source.read_bytes() == originals[shared_label][0]
    assert second_copy.read_bytes() == originals[shared_label][0]
    first_route = (
        run_dir
        / "esp_states"
        / "forecast"
        / "2014030100"
        / first_id
        / "mizuroute"
        / "Ross.r.2014-03-01-00000.nc"
    )
    second_route = (
        run_dir
        / "esp_states"
        / "forecast"
        / "2014030100"
        / second_id
        / "mizuroute"
        / "Ross.r.2014-03-01-00000.nc"
    )
    source_route = run_dir / "mizuroute" / "restart" / shared_label / "Ross.r.2014-03-01-00000.nc"
    first_route.write_bytes(b"mutated-route")
    assert source_route.read_bytes() == originals[shared_label][1]
    assert second_route.read_bytes() == originals[shared_label][1]


def test_balanced_esp_traces_overlap_when_num_workers_exceeds_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    when = datetime(2014, 3, 1)
    run_dir = tmp_path / "run"
    state_dir = run_dir / "states"
    state_dir.mkdir(parents=True)
    ancil = tmp_path / "ancil"
    ancil.mkdir()
    (ancil / "param.nml.default").write_text("&params\n/\n", encoding="utf-8")
    control = ancil / "mizuroute.control"
    control.write_text(
        "<fname_state_in>          cold.nc\n"
        "<case_name>               Ross\n"
        "<output_dir>              /\n"
        "<restart_dir>             /\n",
        encoding="utf-8",
    )
    config_file = _build_config(
        tmp_path,
        extra={
            "num_workers": 2,
            "num_ens": 2,
            "start_time": "2014-02-01 00:00",
            "end_time": "2014-08-01 00:00",
            "esp": {
                "initialization": "balanced_independent",
                "n_members": 2,
                "seed": 3,
                "start_year": 2014,
                "end_year": 2014,
                "forecast_length_days": 5,
                "dates": ["2014-03-01 00:00"],
                "run_mizuroute": True,
                "mizuroute": {
                    "executable": str(tmp_path / "route.exe"),
                    "control_file": str(control),
                    "ancil_dir": str(ancil),
                    "case_name": "Ross",
                    "restart_write": "monthly",
                },
            },
        },
    )
    (tmp_path / "fileManager.txt").write_text(
        "outputPath\t'/tmp/output/'\n"
        "initConditionFile\t'coldState.nc'\n"
        "forcingPath\t'/forcing/base/'\n"
        "forcingListFile\t'base_forcing.txt'\n"
        "simStartTime\t'2014-02-01 00:00'\n"
        "simEndTime\t'2014-08-01 00:00'\n",
        encoding="utf-8",
    )
    particles: list[ParticleState] = []
    for label in ("001", "002"):
        summa = state_dir / restart_filename(label, when)
        route = run_dir / "mizuroute" / "restart" / label / "Ross.r.2014-03-01-00000.nc"
        route.parent.mkdir(parents=True)
        summa.write_bytes(f"summa-{label}".encode())
        route.write_bytes(f"route-{label}".encode())
        manager = tmp_path / f"fileManagerEns{label}.txt"
        manager.write_text(f"outFilePrefix\t'{label}'\n", encoding="utf-8")
        particles.append(
            ParticleState(
                label=label,
                donor_label=label,
                summa_restart=str(summa),
                mizuroute_restart=str(route),
                weight=0.5,
                file_manager=str(manager),
                resampled=False,
            )
        )
    write_post_da_catalog(run_dir, when, particles)

    inflight = {"current": 0, "peak": 0}
    counter = threading.Lock()

    def fake_summa(_exe, _file_manager, **kwargs):
        assert kwargs["open_output"] is False
        manager = kwargs["sim_config"]["manager"]
        output = Path(manager["outputPath"])
        output.mkdir(parents=True, exist_ok=True)
        dataset = xr.Dataset(
            {"scalarSWE": ("time", [1.0])},
            coords={"time": pd.to_datetime(["2014-03-01"])},
        )
        with netcdf_lock:
            dataset.to_netcdf(output / f"{manager['outFilePrefix']}_timestep.nc")
            dataset.close()
        with counter:
            inflight["current"] += 1
            inflight["peak"] = max(inflight["peak"], inflight["current"])
        try:
            time.sleep(0.2)
        finally:
            with counter:
                inflight["current"] -= 1

    def fake_route(_executable, control_file, log_stdout=False) -> None:
        from snowda.adapters.mizuroute import read_control_value

        output_dir = Path(read_control_value(control_file, "output_dir").rstrip("/"))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "Ross.h.2014-03-01-00000.nc").write_bytes(b"routed")

    monkeypatch.setattr("snowda.workflows.esp.run_simulation", fake_summa)
    monkeypatch.setattr("snowda.workflows.esp.run_mizuroute", fake_route)

    from snowda.config.pysummada import load_pysummada_config

    cfg = load_pysummada_config(config_file)
    assert cfg.esp is not None
    assert cfg.num_workers == 2
    records = run_balanced_esp_traces(cfg, cfg.esp, run_dir)

    assert inflight["peak"] >= 2
    member_map = json.loads((run_dir / "esp" / "member_map.json").read_text(encoding="utf-8"))
    mapped = member_map["issues"][0]["members"]
    assert [record.state_member for record in records] == [
        member["state_label"] for member in mapped
    ]
    assert [record.trace_year for record in records] == [
        member["trace_year"] for member in mapped
    ]
