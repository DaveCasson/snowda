from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from snowda.core.assimilation import DataAssimilationCore, restart_filename
from snowda.core.particle_filter import ParticleFilterCore
from snowda.workflows.run_summa_da import _resample_member_states, _run_swe_assimilation
from tests.test_pysummada_config import _build_config


def _routing_config(tmp_path: Path) -> dict:
    return {
        "executable": str(tmp_path / "route.exe"),
        "control_file": str(tmp_path / "mizuroute.control"),
        "ancil_dir": str(tmp_path / "ancil"),
        "case_name": "Ross",
        "restart_write": "monthly",
    }


def _swe_on_issue(path: Path) -> None:
    when = pd.Timestamp("2014-03-01")
    ds = xr.Dataset(
        {
            "swe_model_equivalent": (("time", "station_id"), np.array([[12.0]])),
            "mapping_flag": (("time", "station_id"), np.array([[0]], dtype=np.int8)),
        },
        coords={"time": [when], "station_id": ["A"]},
    )
    ds["swe_model_equivalent"].attrs["units"] = "kg m**-2"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _da_config(tmp_path: Path, swe: Path) -> Path:
    return _build_config(
        tmp_path,
        extra={
            "num_ens": 1,
            "start_time": "2014-02-01 00:00",
            "end_time": "2014-08-01 00:00",
            "assimilation_mode": "swe",
            "assimilation": {
                "dates": ["2014-03-01 00:00"],
                "obs_window_days": 1,
                "swe_nc": str(swe),
                "z_var": 0.1,
                "hru_stations": [{"hru_id": 1, "station_id": "A"}],
            },
            "esp": {
                "initialization": "balanced_independent",
                "start_year": 2014,
                "end_year": 2014,
                "forecast_end": "08-01",
                "n_members": 1,
                "seed": 1,
                "dates": ["2014-03-01 00:00"],
                "run_mizuroute": True,
                "mizuroute": _routing_config(tmp_path),
            },
        },
    )


def test_swe_assimilation_routes_from_esp_mizuroute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    swe = tmp_path / "swe.nc"
    _swe_on_issue(swe)
    config_file = _da_config(tmp_path, swe)
    from snowda.config.pysummada import load_pysummada_config

    cfg = load_pysummada_config(config_file)
    assert cfg.mizuroute is None
    assert cfg.esp is not None and cfg.esp.mizuroute is not None
    manager = tmp_path / "fileManagerEns001.txt"
    manager.write_text(
        "simStartTime\t'2014-02-01 00:00'\nsimEndTime\t'2014-08-01 00:00'\n",
        encoding="utf-8",
    )
    assimilation = DataAssimilationCore(cfg)
    assimilation.create_ens_config([str(manager)])
    run_dir = tmp_path / "run"
    routed: list[tuple[datetime, datetime, bool, str]] = []

    def fake_run_ensemble(_exe, _ens_config, **kwargs):
        when = datetime.strptime(kwargs["run_suffix"], "%Y_%m_%d")
        state_dir = run_dir / "states"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / restart_filename("001", when)).write_bytes(b"summa-001")
        return {}

    def fake_route(mizu, _run_dir, labels, _summa_dir, start, end, restarts, **kwargs):
        routed.append((start, end, kwargs["require_restart"], mizu.case_name))
        for label in labels:
            route = (
                Path(_run_dir)
                / "mizuroute"
                / "restart"
                / label
                / f"Ross.r.{end.strftime('%Y-%m-%d')}-00000.nc"
            )
            route.parent.mkdir(parents=True, exist_ok=True)
            route.write_bytes(f"route-{label}".encode())
            restarts[label] = route

    monkeypatch.setattr("snowda.workflows.run_summa_da.run_ensemble", fake_run_ensemble)
    monkeypatch.setattr(
        "snowda.workflows.run_summa_da.route_ensemble_segment", fake_route
    )
    monkeypatch.setattr(
        "snowda.workflows.run_summa_da.read_member_hru_swe",
        lambda *_args, **_kwargs: {1: 12.0},
    )

    _run_swe_assimilation(
        cfg,
        assimilation,
        [str(manager)],
        ParticleFilterCore(),
        run_dir,
        tmp_path / "summa",
        logging.getLogger("test_run_summa_da"),
    )

    assert routed == [
        (datetime(2014, 2, 1), datetime(2014, 3, 1), True, "Ross"),
        (datetime(2014, 3, 1), datetime(2014, 8, 1), False, "Ross"),
    ]
    catalog = json.loads(
        (run_dir / "assimilation" / "post_da_2014030100.json").read_text(encoding="utf-8")
    )
    particle = catalog["particles"][0]
    assert particle["label"] == "001"
    assert Path(particle["summa_restart"]).read_bytes() == b"summa-001"
    assert Path(particle["mizuroute_restart"]).read_bytes() == b"route-001"


def test_catalog_requires_restart_when_only_esp_mizuroute_is_set(tmp_path: Path) -> None:
    swe = tmp_path / "swe.nc"
    _swe_on_issue(swe)
    config_file = _da_config(tmp_path, swe)
    from snowda.config.pysummada import load_pysummada_config

    cfg = load_pysummada_config(config_file)
    when = datetime(2014, 3, 1)
    state_dir = tmp_path / "run" / "states"
    state_dir.mkdir(parents=True)
    (state_dir / restart_filename("001", when)).write_bytes(b"summa-001")
    manager = tmp_path / "fileManagerEns001.txt"
    manager.write_text("initConditionFile\t'coldState.nc'\n", encoding="utf-8")
    weights = pd.DataFrame([[1.0]], columns=cfg.ens_label_array)
    with pytest.raises(FileNotFoundError, match="mizuRoute restart"):
        _resample_member_states(
            cfg,
            DataAssimilationCore(cfg),
            [str(manager)],
            ParticleFilterCore(),
            weights,
            when,
            cfg.ens_label_array,
            {},
            run_dir=tmp_path / "run",
        )
