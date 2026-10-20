from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from snowda.config.pysummada import (
    load_pysummada_config,
    validate_da_config,
    validate_historic_da_config,
)
from snowda.core.assimilation import (
    DataAssimilationCore,
    assimilation_segments,
    resolve_assimilation_dates,
    restart_filename,
)
from snowda.core.particle_filter import ParticleFilterCore
from snowda.workflows.run_summa_da import _resample_member_states
from snowda.core.swe_assimilation import lumped_swe_by_date
from snowda.io.summa_ensemble import EnsembleFileWriter
from snowda.io.swe_obs import read_lumped_member_swe, read_model_equivalent_swe
from snowda.workflows.summa_common import apply_segment_initial_conditions
from tests.test_pysummada_config import _build_config, _esp_block, _write_base_file_manager


def _mizu_block(tmp_path: Path) -> dict:
    return {
        "executable": str(tmp_path / "route.exe"),
        "control_file": str(tmp_path / "mizuroute.control"),
        "ancil_dir": str(tmp_path / "ancil"),
        "case_name": "Ross",
        "restart_write": "monthly",
    }


def _swe_file(path: Path) -> None:
    times = pd.to_datetime(
        ["2014-02-20", "2014-03-01", "2014-03-06", "2014-03-20"]
    )
    stations = ["A", "B"]
    swe = np.array(
        [
            [10.0, 30.0],
            [np.nan, np.nan],
            [4.0, 8.0],
            [100.0, 100.0],
        ]
    )
    flags = np.array(
        [
            [0, 0],
            [3, 3],
            [0, 1],
            [0, 3],
        ],
        dtype=np.int8,
    )
    ds = xr.Dataset(
        {
            "swe_model_equivalent": (("time", "station_id"), swe),
            "mapping_flag": (("time", "station_id"), flags),
        },
        coords={"time": times, "station_id": stations},
    )
    ds["swe_model_equivalent"].attrs["units"] = "kg m**-2"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _assimilation_block(swe_nc: Path, **overrides: object) -> dict:
    block = {
        "dates": ["2014-03-01 00:00", "2014-03-10 00:00"],
        "obs_window_days": 6,
        "swe_nc": str(swe_nc),
        "z_var": 0.02,
    }
    block.update(overrides)
    return block


def test_resolve_dates_drops_outside_and_filters_months() -> None:
    start = datetime(2014, 1, 1)
    end = datetime(2014, 6, 1)
    resolved = resolve_assimilation_dates(
        start,
        end,
        [
            datetime(2013, 12, 1),
            datetime(2014, 2, 1),
            datetime(2014, 3, 1),
            datetime(2014, 7, 1),
        ],
        frequency_days=None,
        months=(3,),
    )
    assert resolved == [datetime(2014, 3, 1)]


def test_resolve_dates_rejects_start_time() -> None:
    start = datetime(2014, 1, 1)
    with pytest.raises(ValueError, match="after start_time"):
        resolve_assimilation_dates(
            start,
            datetime(2014, 2, 1),
            [start],
            frequency_days=None,
        )


def test_frequency_skips_start_and_keeps_end() -> None:
    start = datetime(2014, 1, 1)
    end = datetime(2014, 1, 11)
    resolved = resolve_assimilation_dates(
        start, end, [], frequency_days=5, months=None
    )
    assert resolved == [datetime(2014, 1, 6), datetime(2014, 1, 11)]


def test_segments_assimilate_only_on_dates() -> None:
    start = datetime(2013, 8, 1)
    end = datetime(2014, 8, 1)
    dates = [datetime(2014, 2, 1), datetime(2014, 5, 1)]
    segments = assimilation_segments(start, end, dates)
    assert segments == [
        (start, datetime(2014, 2, 1), True),
        (datetime(2014, 2, 1), datetime(2014, 5, 1), True),
        (datetime(2014, 5, 1), end, False),
    ]


def test_restart_filename_uses_hour() -> None:
    when = datetime(2014, 3, 1, 0, 0)
    assert restart_filename("016", when) == "016_restart_2014030100_2014_03_01.nc"


def test_lumped_swe_assigns_nearest_and_averages() -> None:
    times = pd.to_datetime(["2014-03-06", "2014-03-20"])
    stations = ["A", "B"]
    swe = np.array(
        [
            [4.0, 8.0],
            [100.0, np.nan],
        ]
    )
    flags = np.array([[0, 1], [0, 3]], dtype=np.int8)
    dates = [datetime(2014, 3, 1), datetime(2014, 3, 10)]
    lumped = lumped_swe_by_date(times, stations, swe, flags, dates, window_days=6)
    # Mar 6 is 5 days from Mar 1 and 4 days from Mar 10.
    assert datetime(2014, 3, 1) not in lumped
    march_10 = lumped[datetime(2014, 3, 10)]
    assert march_10.value == pytest.approx(6.0)
    assert {sample.station_id for sample in march_10.samples} == {"A", "B"}
    # Mar 20 is 10 days from Mar 10, outside a 6-day window. Flag 3 is ignored.
    assert len(lumped) == 1


def test_lumped_swe_includes_zero_flag_and_skips_missing() -> None:
    times = pd.to_datetime(["2014-03-01"])
    swe = np.array([[0.0, 5.0]])
    flags = np.array([[1, 3]], dtype=np.int8)
    lumped = lumped_swe_by_date(
        times,
        ["A", "B"],
        swe,
        flags,
        [datetime(2014, 3, 1)],
        window_days=0,
    )
    assert lumped[datetime(2014, 3, 1)].value == pytest.approx(0.0)
    assert len(lumped[datetime(2014, 3, 1)].samples) == 1


def test_read_lumped_member_swe_from_timestep_file(tmp_path: Path) -> None:
    when = datetime(2014, 3, 1)
    ds = xr.Dataset(
        {
            "scalarSWE": (("time", "hru"), np.array([[12.0]])),
            "hruId": ("hru", [7]),
        },
        coords={"time": [when], "hru": [0]},
    )
    ds["scalarSWE"].attrs["units"] = "kg m-2"
    ds.to_netcdf(tmp_path / "001_2014_03_01_timestep.nc")
    hru_id, value, units = read_lumped_member_swe(
        tmp_path,
        "001",
        variable="scalarSWE",
        when=when,
        file_suffix="timestep.nc",
    )
    assert hru_id == 7
    assert value == pytest.approx(12.0)
    assert units == "kg m-2"


def test_particle_weights_accept_lumped_frames() -> None:
    when = datetime(2014, 3, 1)
    hru_id = 7
    index = pd.MultiIndex.from_tuples([(hru_id, when)], names=["hru", "time"])
    particles = pd.DataFrame({"001": [10.0], "002": [20.0]}, index=index)
    measurements = pd.DataFrame(
        {"hruId": [hru_id], "meas_snw": [10.0]},
        index=index,
    )
    weights = ParticleFilterCore().update_particle_weights(
        particles,
        measurements,
        z_var=0.02,
        end_date_current_run=when,
        hru_index=pd.DataFrame({"hruId": [hru_id]}),
        ens_label_array=["001", "002"],
        obs_column="meas_snw",
    )
    assert list(weights.columns) == ["001", "002"]
    assert weights.iloc[0, 0] > weights.iloc[0, 1]


def test_historic_forcing_lists_use_member_ids(tmp_path: Path) -> None:
    forcing = tmp_path / "ensemble"
    for label in ("001", "016"):
        member = forcing / label
        member.mkdir(parents=True)
        (member / f"ensMember_{label}.nc").write_bytes(b"netcdf")
    fm = tmp_path / "fileManager.txt"
    _write_base_file_manager(fm)
    config_file = _build_config(
        tmp_path,
        extra={
            "num_ens": 2,
            "ensemble_forcing_mode": "historic",
            "ensemble_forcing_dir": str(forcing),
            "ensemble_members": ["001", "016"],
            "forcing_file_base": "/unused",
        },
    )
    cfg = load_pysummada_config(config_file)
    assert cfg.ens_label_array == ["001", "016"]
    writer = EnsembleFileWriter(cfg)
    paths = writer.write_ens_file_managers()
    writer.write_forcing_file_list()
    first = Path(paths[0]).read_text(encoding="utf-8")
    second = Path(paths[1]).read_text(encoding="utf-8")
    assert f"'{forcing / '001'}/'" in first or f"'{(forcing / '001').as_posix()}/'" in first
    assert "forcingFileListEns016.txt" in second
    assert "'016'" in second
    list_016 = (cfg.file_list_dir / "forcingFileListEns016.txt").read_text(encoding="utf-8")
    assert list_016.strip() == "ensMember_016.nc"


def test_historic_member_requires_one_netcdf(tmp_path: Path) -> None:
    forcing = tmp_path / "ensemble" / "001"
    forcing.mkdir(parents=True)
    config_file = _build_config(
        tmp_path,
        extra={
            "ensemble_forcing_mode": "historic",
            "ensemble_forcing_dir": str(tmp_path / "ensemble"),
        },
    )
    cfg = load_pysummada_config(config_file)
    writer = EnsembleFileWriter(cfg)
    with pytest.raises(ValueError, match="one forcing netCDF"):
        writer.write_ens_file_managers()


def test_monthly_forcing_list_unchanged(tmp_path: Path) -> None:
    config_file = _build_config(tmp_path, extra={"num_ens": 2})
    cfg = load_pysummada_config(config_file)
    writer = EnsembleFileWriter(cfg)
    writer.write_forcing_file_list()
    text = (cfg.file_list_dir / "forcingFileListEns001.txt").read_text(encoding="utf-8")
    assert "/abs/forcing_202001.001.nc" in text
    assert "historic" not in text


def test_keep_existing_init_does_not_replace_donor(tmp_path: Path) -> None:
    fm = tmp_path / "fileManagerEns001.txt"
    fm.write_text("initConditionFile\t'016_restart_2014030100_2014_03_01.nc'\n", encoding="utf-8")
    ens_config: dict = {"001": {"manager": {}}}
    apply_segment_initial_conditions(
        ["001"],
        [str(fm)],
        ens_config,
        datetime(2014, 3, 1),
        is_first=False,
        keep_existing=True,
    )
    assert "016_restart_2014030100_2014_03_01.nc" in fm.read_text(encoding="utf-8")
    assert ens_config["001"]["manager"]["initConditionFile"].startswith("016_restart")

    apply_segment_initial_conditions(
        ["001"],
        [str(fm)],
        ens_config,
        datetime(2014, 4, 1),
        is_first=False,
        keep_existing=False,
    )
    written = fm.read_text(encoding="utf-8")
    assert "001_restart_2014040100_2014_04_01.nc" in written
    assert "016_restart" not in written


def test_swe_da_config_does_not_require_fsca(tmp_path: Path) -> None:
    swe = tmp_path / "swe.nc"
    _swe_file(swe)
    forcing = tmp_path / "ensemble" / "001"
    forcing.mkdir(parents=True)
    config_file = _build_config(
        tmp_path,
        extra={
            "start_time": "2014-01-01 00:00",
            "end_time": "2014-04-01 00:00",
            "assimilation_mode": "swe",
            "ensemble_forcing_mode": "historic",
            "ensemble_forcing_dir": str(tmp_path / "ensemble"),
            "summa_output_dir": "/abs/summa_out",
            "esp": _esp_block(),
            "assimilation": _assimilation_block(swe),
        },
    )
    times, _stations, _values, _flags, units = read_model_equivalent_swe(swe)
    assert units == "kg m**-2"
    assert len(times) == 4
    cfg = load_pysummada_config(config_file)
    validate_da_config(cfg)
    assert cfg.assimilation is not None
    assert cfg.assimilation.swe_var == "swe_model_equivalent"


def test_swe_da_rejects_date_at_start(tmp_path: Path) -> None:
    swe = tmp_path / "swe.nc"
    _swe_file(swe)
    config_file = _build_config(
        tmp_path,
        extra={
            "start_time": "2014-03-01 00:00",
            "end_time": "2014-04-01 00:00",
            "assimilation_mode": "swe",
            "summa_output_dir": "/abs/summa_out",
            "esp": _esp_block(),
            "assimilation": _assimilation_block(
                swe, dates=["2014-03-01 00:00"]
            ),
        },
    )
    cfg = load_pysummada_config(config_file)
    with pytest.raises(ValueError, match="after start_time"):
        validate_da_config(cfg)


def test_historic_da_accepts_swe_without_esp(tmp_path: Path) -> None:
    swe = tmp_path / "swe.nc"
    _swe_file(swe)
    forcing = tmp_path / "ensemble" / "001"
    forcing.mkdir(parents=True)
    config_file = _build_config(
        tmp_path,
        extra={
            "start_time": "2014-01-01 00:00",
            "end_time": "2014-04-01 00:00",
            "assimilation_mode": "swe",
            "pf_scheme": "sis",
            "ensemble_forcing_mode": "historic",
            "ensemble_forcing_dir": str(tmp_path / "ensemble"),
            "summa_output_dir": "/abs/summa_out",
            "mizuroute": _mizu_block(tmp_path),
            "assimilation": _assimilation_block(swe),
        },
    )
    cfg = load_pysummada_config(config_file)
    validate_historic_da_config(cfg)
    assert cfg.esp is None
    assert cfg.mizuroute is not None
    assert cfg.pf_scheme == "sis"


def test_historic_da_requires_mizuroute(tmp_path: Path) -> None:
    swe = tmp_path / "swe.nc"
    _swe_file(swe)
    config_file = _build_config(
        tmp_path,
        extra={
            "start_time": "2014-01-01 00:00",
            "end_time": "2014-04-01 00:00",
            "assimilation_mode": "swe",
            "summa_output_dir": "/abs/summa_out",
            "assimilation": _assimilation_block(swe),
        },
    )
    cfg = load_pysummada_config(config_file)
    with pytest.raises(ValueError, match="requires top-level `mizuroute`"):
        validate_historic_da_config(cfg)


def test_historic_da_rejects_esp(tmp_path: Path) -> None:
    swe = tmp_path / "swe.nc"
    _swe_file(swe)
    config_file = _build_config(
        tmp_path,
        extra={
            "start_time": "2014-01-01 00:00",
            "end_time": "2014-04-01 00:00",
            "assimilation_mode": "swe",
            "summa_output_dir": "/abs/summa_out",
            "esp": _esp_block(),
            "assimilation": _assimilation_block(swe),
        },
    )
    cfg = load_pysummada_config(config_file)
    with pytest.raises(ValueError, match="does not run ESP"):
        validate_historic_da_config(cfg)


def test_sis_does_not_replace_restart(tmp_path: Path) -> None:
    config_file = _build_config(tmp_path, extra={"pf_scheme": "sis", "num_ens": 2})
    cfg = load_pysummada_config(config_file)
    managers = []
    for label in cfg.ens_label_array:
        path = tmp_path / f"fileManagerEns{label}.txt"
        path.write_text("initConditionFile\t'coldState.nc'\n", encoding="utf-8")
        managers.append(str(path))
    weights = pd.DataFrame([[0.5, 0.5]], columns=cfg.ens_label_array)
    copied = _resample_member_states(
        cfg,
        DataAssimilationCore(cfg),
        managers,
        ParticleFilterCore(),
        weights,
        datetime(2014, 3, 1),
        cfg.ens_label_array,
    )
    assert copied is False
    for path in managers:
        assert "coldState.nc" in Path(path).read_text(encoding="utf-8")


def test_sir_copies_donor_restart(tmp_path: Path) -> None:
    config_file = _build_config(
        tmp_path,
        extra={"pf_scheme": "sir", "num_ens": 2, "n_eff_threshold": 0.5},
    )
    cfg = load_pysummada_config(config_file)
    managers = []
    for label in cfg.ens_label_array:
        path = tmp_path / f"fileManagerEns{label}.txt"
        path.write_text("initConditionFile\t'coldState.nc'\n", encoding="utf-8")
        managers.append(str(path))
    weights = pd.DataFrame([[0.5, 0.5]], columns=cfg.ens_label_array)
    copied = _resample_member_states(
        cfg,
        DataAssimilationCore(cfg),
        managers,
        ParticleFilterCore(),
        weights,
        datetime(2014, 3, 1),
        cfg.ens_label_array,
    )
    assert copied is True
    for path in managers:
        text = Path(path).read_text(encoding="utf-8")
        assert "coldState.nc" not in text
        assert "_restart_2014030100_2014_03_01.nc" in text


def test_route_segment_uses_member_summa_and_keeps_routing_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from snowda.adapters.mizuroute import read_control_value
    from snowda.config.pysummada import MizuRouteEspConfig
    from snowda.workflows.summa_common import route_ensemble_segment

    ancil = tmp_path / "ancil"
    ancil.mkdir()
    (ancil / "param.nml.default").write_text("&params\n/\n", encoding="utf-8")
    (ancil / "cold.nc").write_bytes(b"cold")
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
    mizu = MizuRouteEspConfig(
        executable=tmp_path / "route.exe",
        control_file=control,
        ancil_dir=ancil,
        case_name="Ross",
        restart_write="monthly",
    )
    summa_dir = tmp_path / "summa"
    summa_dir.mkdir()
    for stamp in ("2014_03_01", "2014_04_01"):
        (summa_dir / f"001_{stamp}_timestep.nc").write_bytes(b"summa")
    seen: list[str] = []

    def fake_run(_executable, control_file, log_stdout=False) -> None:
        text = Path(control_file).read_text(encoding="utf-8")
        seen.append(text)
        output_dir = Path(read_control_value(control_file, "output_dir").rstrip("/"))
        restart_dir = Path(read_control_value(control_file, "restart_dir").rstrip("/"))
        sim_start = read_control_value(control_file, "sim_start")
        start_day = sim_start[:10]
        output_dir.mkdir(parents=True, exist_ok=True)
        restart_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"Ross.h.{start_day}-00000.nc").write_bytes(b"routed")
        end_day = "2014-03-01" if start_day == "2014-02-01" else "2014-04-01"
        (restart_dir / f"Ross.r.{end_day}-00000.nc").write_bytes(b"restart")

    monkeypatch.setattr("snowda.workflows.summa_common.run_mizuroute", fake_run)
    restarts: dict[str, Path | None] = {}
    route_ensemble_segment(
        mizu,
        tmp_path / "run",
        ["001"],
        summa_dir,
        datetime(2014, 2, 1),
        datetime(2014, 3, 1),
        restarts,
        require_restart=True,
    )
    route_ensemble_segment(
        mizu,
        tmp_path / "run",
        ["001"],
        summa_dir,
        datetime(2014, 3, 1),
        datetime(2014, 4, 1),
        restarts,
        require_restart=False,
    )
    assert "001_2014_03_01_timestep.nc" in seen[0]
    assert "cold.nc" in seen[0]
    assert "001_2014_04_01_timestep.nc" in seen[1]
    assert "Ross.r.2014-03-01-00000.nc" in seen[1]
    assert "cold.nc" not in seen[1]
    assert (tmp_path / "run" / "mizuroute" / "001_2014_04_01_mizuroute.nc").is_file()


def test_sir_swaps_mizuroute_restarts_with_summa_donors(tmp_path: Path) -> None:
    from snowda.workflows.summa_common import resample_routing_restarts

    restart_root = tmp_path / "mizuroute" / "restart"
    own = restart_root / "001" / "Ross.r.2014-03-01-00000.nc"
    donor = restart_root / "002" / "Ross.r.2014-03-01-00000.nc"
    own.parent.mkdir(parents=True)
    donor.parent.mkdir(parents=True)
    own.write_bytes(b"route-001")
    donor.write_bytes(b"route-002")
    restarts: dict[str, Path | None] = {"001": own, "002": donor}

    resample_routing_restarts(restarts, ["001", "002"], [1, 0])

    assert restarts["001"] is not None and restarts["001"].read_bytes() == b"route-002"
    assert restarts["002"] is not None and restarts["002"].read_bytes() == b"route-001"
    assert restarts["001"].parent.name == "001"
    assert restarts["002"].parent.name == "002"
    assert not (restart_root / ".resample_stage").exists()
