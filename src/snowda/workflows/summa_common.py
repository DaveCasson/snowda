from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from snowda.adapters.mizuroute import (
    read_control_value,
    run_mizuroute,
    update_control_file,
)
from snowda.adapters.pysumma import get_file_manager_value
from snowda.config.pysummada import MizuRouteEspConfig, PysummadaConfig
from snowda.core.assimilation import DataAssimilationCore, restart_filename
from snowda.io.summa_ensemble import (
    EnsembleFileWriter,
    read_file_manager_option,
    update_file_manager_option,
    write_simulation_times,
)
from snowda.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


def now_utc_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def copy_param_definition(cfg: PysummadaConfig, run_dir: Path) -> bool:
    search_dirs = [
        cfg.file_manager_dir,
        Path(cfg.base_file_manager).parent,
    ]
    for folder in search_dirs:
        candidate = folder / "param.nml.default"
        if candidate.is_file():
            shutil.copy(candidate, run_dir / "param.nml.default")
            return True
    return False


def member_summa_timestep(output_dir: Path, member: str, segment_end: datetime) -> Path:
    """SUMMA timestep file written for one member at a segment end."""
    stamp = segment_end.strftime("%Y_%m_%d")
    path = output_dir / f"{member}_{stamp}_timestep.nc"
    if not path.is_file():
        raise FileNotFoundError(
            f"No SUMMA timestep output for member {member} ending {stamp} in {output_dir}."
        )
    return path


def mizuroute_member_output_name(member: str, segment_end: datetime) -> str:
    return f"{member}_{segment_end.strftime('%Y_%m_%d')}_mizuroute.nc"


def mizuroute_restart_name(case_name: str, when: datetime) -> str:
    return f"{case_name}.r.{when.strftime('%Y-%m-%d')}-00000.nc"


def find_summa_timestep_output(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("*_timestep.nc"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No SUMMA timestep output found in {output_dir}")
    return candidates[-1]


def find_mizuroute_output(output_dir: Path, case_name: str, sim_start: datetime) -> Path:
    canonical = output_dir / f"{case_name}.h.{sim_start.strftime('%Y-%m-%d')}-00000.nc"
    if canonical.exists():
        return canonical
    globbed = sorted(output_dir.glob(f"{case_name}.h.*.nc"), key=lambda p: p.stat().st_mtime)
    if globbed:
        return globbed[-1]
    raise FileNotFoundError(f"Could not find mizuRoute output in {output_dir}")


def locate_resource(
    explicit: Path | None, fallback_dirs: Sequence[Path], filename: str
) -> Path | None:
    if explicit:
        return explicit
    for folder in fallback_dirs:
        candidate = folder / filename
        if candidate.exists():
            return candidate
    return None


def copy_optional_resource(
    source: Path | None,
    fallback_dirs: Sequence[Path],
    destination: Path,
    label: str,
    required: bool = False,
) -> None:
    resource = locate_resource(source, fallback_dirs, destination.name)
    if not resource:
        if required:
            raise FileNotFoundError(
                f"Could not locate required {label} (looked in {fallback_dirs})."
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resource, destination)


def setup_run_directory(
    cfg: PysummadaConfig,
    config_path: str | Path,
    *,
    log_filename: str,
    logger_name: str,
    start_message: str,
) -> tuple[Path, logging.Logger]:
    run_dir = ensure_dir(cfg.output_dir / now_utc_tag())
    copy_param_definition(cfg, run_dir)
    log_file = run_dir / log_filename
    configure_logging(level=logging.INFO, log_file=log_file)
    log = get_logger(logger_name)
    log.info(start_message)
    log.info("Config: %s", str(config_path))
    log.info("Run dir: %s", run_dir)

    config_out = run_dir / "config.resolved.json"
    config_out.write_text(json.dumps(cfg.__dict__, default=str, indent=2), encoding="utf-8")
    log.info("Wrote config snapshot: %s", config_out)
    return run_dir, log


def write_provenance(run_dir: Path, payload: dict[str, Any], log: logging.Logger) -> Path:
    prov = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        **payload,
        "run_dir": str(run_dir),
    }
    prov_out = run_dir / "provenance.json"
    prov_out.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    log.info("Wrote provenance: %s", prov_out)
    return prov_out


def prepare_ensemble(cfg: PysummadaConfig) -> tuple[EnsembleFileWriter, DataAssimilationCore]:
    file_writer = EnsembleFileWriter(cfg)
    file_writer.write_ensemble_files()
    if not file_writer.ens_file_managers:
        raise RuntimeError("Failed to generate ensemble file managers.")
    assimilation = DataAssimilationCore(cfg)
    assimilation.create_ens_config(file_writer.ens_file_managers)
    return file_writer, assimilation


def apply_segment_initial_conditions(
    labels: Sequence[str],
    file_managers: Sequence[str],
    ens_config: dict[str, Any],
    segment_start: datetime,
    *,
    is_first: bool,
    keep_existing: bool,
) -> None:
    """Point each member at the restart that starts this segment.

    The first segment keeps the cold state already written to the file
    managers. After resampling, ``keep_existing`` leaves the donor restart
    on disk and copies that name into the in-memory manager config.
    """
    if is_first:
        return
    if keep_existing:
        for label, file_manager in zip(labels, file_managers):
            name = read_file_manager_option(file_manager, "initConditionFile")
            manager = ens_config.setdefault(label, {}).setdefault("manager", {})
            manager["initConditionFile"] = name
        return
    for label, file_manager in zip(labels, file_managers):
        name = restart_filename(label, segment_start)
        update_file_manager_option(file_manager, "initConditionFile", f"'{name}'")
        manager = ens_config.setdefault(label, {}).setdefault("manager", {})
        manager["initConditionFile"] = name


def apply_run_time(
    cfg: PysummadaConfig,
    assimilation: DataAssimilationCore,
    file_managers: Sequence[str],
    start_time: datetime,
    end_time: datetime,
) -> dict[str, Any]:
    current_run_config = assimilation.update_run_time(start_time, end_time)
    write_simulation_times(
        [*file_managers, cfg.base_file_manager],
        start_time,
        end_time,
    )
    return current_run_config


def summa_run_output_dir(run_dir: Path) -> Path:
    """SUMMA netCDF output for a timestamped run: ``<run_dir>/summa``."""
    return ensure_dir(run_dir / "summa")


def summa_run_state_dir(run_dir: Path) -> Path:
    """SUMMA restart files for a timestamped run: ``<run_dir>/states``."""
    return ensure_dir(run_dir / "states")


def _point_managers_at(
    file_managers: Sequence[str],
    ens_config: dict[str, Any],
    key: str,
    directory: Path,
) -> Path:
    directory = ensure_dir(directory)
    disk_value = f"'{directory}/'"
    runtime_value = str(directory) + "/"
    for file_manager in file_managers:
        update_file_manager_option(file_manager, key, disk_value)
    for entry in ens_config.values():
        if not isinstance(entry, dict):
            continue
        manager = entry.setdefault("manager", {})
        manager[key] = runtime_value
    return directory


def _copy_initial_states(file_managers: Sequence[str], state_dir: Path) -> None:
    """Copy each member's current initial state into the run state directory."""
    copied: set[Path] = set()
    for file_manager in file_managers:
        source_dir = Path(read_file_manager_option(file_manager, "statePath"))
        init_name = read_file_manager_option(file_manager, "initConditionFile")
        source = source_dir / init_name
        destination = state_dir / Path(init_name).name
        if destination in copied:
            continue
        if not source.is_file():
            raise FileNotFoundError(f"SUMMA initial state not found: {source}")
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        log.info("Copied SUMMA initial state %s to %s", source.name, destination)
        copied.add(destination)


def redirect_summa_output(
    file_managers: Sequence[str],
    ens_config: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Point file managers at this run's output and state directories.

    ``outputPath`` becomes ``<run_dir>/summa`` and ``statePath`` becomes
    ``<run_dir>/states``. The initial state named in each file manager is
    copied in before ``statePath`` changes, so the first segment can still
    start from the cold state.
    """
    output_dir = _point_managers_at(file_managers, ens_config, "outputPath", output_dir)
    state_dir = summa_run_state_dir(output_dir.parent)
    _copy_initial_states(file_managers, state_dir)
    _point_managers_at(file_managers, ens_config, "statePath", state_dir)
    log.info("SUMMA statePath: %s", state_dir)
    return output_dir


def mizuroute_run_dirs(run_dir: Path) -> tuple[Path, Path]:
    output_dir = ensure_dir(run_dir / "mizuroute")
    restart_dir = ensure_dir(output_dir / "restart")
    return output_dir, restart_dir


def _seed_mizuroute_initial_state(
    mizu_cfg: MizuRouteEspConfig, restart_dir: Path, sim_start: datetime
) -> str:
    control_parent = Path(mizu_cfg.control_file).parent
    fallback_dirs = [
        mizu_cfg.ancil_dir,
        control_parent,
        control_parent / "restart_files",
    ]
    from_control = read_control_value(mizu_cfg.control_file, "fname_state_in")
    dated_name = f"{mizu_cfg.case_name}.r.{sim_start.strftime('%Y-%m-%d')}-00000.nc"
    for state_name in (from_control, dated_name):
        if not state_name:
            continue
        resource = locate_resource(None, fallback_dirs, state_name)
        if resource:
            copy_optional_resource(
                resource,
                fallback_dirs,
                restart_dir / state_name,
                "mizuRoute initial state file",
                required=True,
            )
            return state_name
    raise FileNotFoundError(
        "Could not locate mizuRoute initial state "
        f"(looked for {from_control!r} and {dated_name!r} in {fallback_dirs})."
    )


def run_mizuroute_after_summa(
    cfg: PysummadaConfig,
    mizu_cfg: MizuRouteEspConfig,
    run_dir: Path,
    sim_start: datetime,
    sim_end: datetime,
    *,
    summa_output_dir: Path | None = None,
) -> Path:
    if summa_output_dir is None:
        summa_output_dir = Path(
            get_file_manager_value(str(cfg.base_file_manager), "outputPath").replace("'", "")
        )
    summa_output = find_summa_timestep_output(summa_output_dir)

    mizu_output_dir, mizu_restart_dir = mizuroute_run_dirs(run_dir)
    state_in = _seed_mizuroute_initial_state(mizu_cfg, mizu_restart_dir, sim_start)
    input_dir = summa_output.parent

    control_parent = Path(mizu_cfg.control_file).parent
    copy_optional_resource(
        mizu_cfg.param_nml,
        [mizu_cfg.ancil_dir, control_parent],
        mizu_output_dir / "param.nml.default",
        "mizuRoute param.nml.default",
        required=True,
    )
    copy_optional_resource(
        mizu_cfg.param_nml,
        [mizu_cfg.ancil_dir, control_parent],
        input_dir / "param.nml.default",
        "mizuRoute param.nml.default",
        required=True,
    )
    copy_optional_resource(
        mizu_cfg.topology_nc,
        [mizu_cfg.ancil_dir, control_parent],
        mizu_output_dir / "topology.nc",
        "mizuRoute topology.nc",
    )

    control_file = mizu_output_dir / "mizuroute_deterministic.control"
    control_settings = {
        "fname_qsim": summa_output.name,
        "fname_state_in": state_in,
        "case_name": mizu_cfg.case_name,
        "sim_start": sim_start.strftime("%Y-%m-%d %H:%M:%S"),
        "sim_end": (sim_end - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "ancil_dir": str(mizu_cfg.ancil_dir) + "/",
        "input_dir": str(input_dir) + "/",
        "output_dir": str(mizu_output_dir) + "/",
        "restart_dir": str(mizu_restart_dir) + "/",
        "restart_write": mizu_cfg.restart_write,
    }
    update_control_file(mizu_cfg.control_file, control_file, control_settings)
    run_mizuroute(mizu_cfg.executable, control_file, log_stdout=mizu_cfg.log_output)
    routed_generated = find_mizuroute_output(mizu_output_dir, mizu_cfg.case_name, sim_start)
    mizuroute_output = mizu_output_dir / "mizuroute_deterministic.nc"
    routed_generated.rename(mizuroute_output)
    return mizuroute_output


def route_ensemble_segment(
    mizu_cfg: MizuRouteEspConfig,
    run_dir: Path,
    members: Sequence[str],
    summa_output_dir: Path,
    sim_start: datetime,
    sim_end: datetime,
    restarts: dict[str, Path | None],
    *,
    require_restart: bool,
    num_workers: int = 5,
) -> None:
    """Route each member's SUMMA file for one segment.

    Members in this segment run together. Each one is its own
    ``route_runoff.exe`` process, and the threads only wait on those
    processes. The next segment still waits, because it reads these restarts.

    ``restarts`` is that particle's mizuRoute restart. After SIR it is the
    donor particle's restart, already copied into this member's restart directory.
    """
    if not members:
        return
    control_parent = Path(mizu_cfg.control_file).parent
    copy_optional_resource(
        mizu_cfg.param_nml,
        [mizu_cfg.ancil_dir, control_parent],
        summa_output_dir / "param.nml.default",
        "mizuRoute param.nml.default",
        required=True,
    )
    jobs = [
        (
            member,
            member_summa_timestep(summa_output_dir, member, sim_end),
            restarts.get(member),
        )
        for member in members
    ]
    workers = max(1, min(num_workers, len(jobs)))
    log.info(
        "mizuRoute segment starting: %s members, %s workers, %s to %s",
        len(jobs),
        workers,
        sim_start,
        sim_end,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _route_summa_member,
                mizu_cfg,
                run_dir,
                member,
                summa_output,
                sim_start,
                sim_end,
                restart_in,
            ): member
            for member, summa_output, restart_in in jobs
        }
        for future in as_completed(futures):
            member = futures[future]
            try:
                restart = future.result()
            except Exception:
                log.exception("mizuRoute member %s failed", member)
                raise
            if require_restart and restart is None:
                raise FileNotFoundError(
                    "mizuRoute did not write a restart for member "
                    f"{member} at {sim_end}. Set `mizuroute.restart_write` so a "
                    "restart exists at each assimilation segment end."
                )
            if restart is not None:
                restarts[member] = restart


def _route_summa_member(
    mizu_cfg: MizuRouteEspConfig,
    run_dir: Path,
    member: str,
    summa_output: Path,
    sim_start: datetime,
    sim_end: datetime,
    restart_in: Path | None,
) -> Path | None:
    stamp = sim_end.strftime("%Y%m%d%H")
    output_dir = ensure_dir(run_dir / "mizuroute" / "segments" / member / stamp)
    restart_dir = ensure_dir(run_dir / "mizuroute" / "restart" / member)
    control_parent = Path(mizu_cfg.control_file).parent
    copy_optional_resource(
        mizu_cfg.param_nml,
        [mizu_cfg.ancil_dir, control_parent],
        output_dir / "param.nml.default",
        "mizuRoute param.nml.default",
        required=True,
    )
    copy_optional_resource(
        mizu_cfg.topology_nc,
        [mizu_cfg.ancil_dir, control_parent],
        output_dir / "topology.nc",
        "mizuRoute topology.nc",
    )
    if restart_in is None:
        state_in = _seed_mizuroute_initial_state(mizu_cfg, restart_dir, sim_start)
    else:
        state_in = restart_in.name
    control_file = output_dir / "mizuroute_segment.control"
    control_settings = {
        "fname_qsim": summa_output.name,
        "fname_state_in": state_in,
        "case_name": mizu_cfg.case_name,
        "sim_start": sim_start.strftime("%Y-%m-%d %H:%M:%S"),
        "sim_end": (sim_end - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "ancil_dir": str(mizu_cfg.ancil_dir) + "/",
        "input_dir": str(summa_output.parent) + "/",
        "output_dir": str(output_dir) + "/",
        "restart_dir": str(restart_dir) + "/",
        "restart_write": mizu_cfg.restart_write,
    }
    update_control_file(mizu_cfg.control_file, control_file, control_settings)
    run_mizuroute(mizu_cfg.executable, control_file, log_stdout=mizu_cfg.log_output)
    routed_generated = find_mizuroute_output(output_dir, mizu_cfg.case_name, sim_start)
    flat_dir = ensure_dir(run_dir / "mizuroute")
    flat_output = flat_dir / mizuroute_member_output_name(member, sim_end)
    if flat_output.exists():
        flat_output.unlink()
    routed_generated.rename(flat_output)
    restart_path = restart_dir / mizuroute_restart_name(mizu_cfg.case_name, sim_end)
    if restart_path.is_file():
        return restart_path
    return None


def resample_routing_restarts(
    restarts: dict[str, Path | None],
    member_labels: Sequence[str],
    resample_indices: Sequence[int],
) -> None:
    """Give each member the donor particle's mizuRoute restart.

    Files are staged before copying so a swap does not overwrite a restart
    another member still needs. The copy lives in the receiving member's
    restart directory, which is where the next segment looks up ``fname_state_in``.
    """
    if not restarts:
        return
    assignments: list[tuple[str, Path]] = []
    for member_index, donor_index in enumerate(resample_indices):
        member = member_labels[member_index]
        donor = member_labels[int(donor_index)]
        source = restarts.get(donor)
        if source is None or not source.is_file():
            raise FileNotFoundError(
                f"No mizuRoute restart for donor member {donor} to assign to {member}."
            )
        assignments.append((member, source))
    stage_root = assignments[0][1].parent.parent / ".resample_stage"
    stage_root.mkdir(parents=True, exist_ok=True)
    staged: dict[Path, Path] = {}
    try:
        for _member, source in assignments:
            if source in staged:
                continue
            staged_path = stage_root / f"{source.parent.name}_{source.name}"
            shutil.copy2(source, staged_path)
            staged[source] = staged_path
        for member, source in assignments:
            destination = source.parent.parent / member / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged[source], destination)
            restarts[member] = destination
            log.info("Assigned %s mizuRoute restart from %s", member, source)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
