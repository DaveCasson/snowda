from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

from snowda.utils.logging import get_logger

log = get_logger(__name__)


def _require_pysumma() -> Any:
    try:
        import pysumma as ps  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "pysumma is required for SUMMA integration. Install from "
            "https://github.com/CH-Earth/pysumma"
        ) from e
    return ps


def get_file_manager_value(file_manager_path: str, key: str) -> str:
    """Read a value from a SUMMA file manager using pysumma."""
    ps = _require_pysumma()
    fm = ps.FileManager(file_manager_path)
    return str(fm.get_value(key))


_PYSUMMA_FILE_EDITS = (
    "decisions",
    "parameters",
    "output_control",
    "attributes",
    "trial_parameters",
)


def _config_rewrites_files(config: Dict[str, Any], *, write_config: bool, write_manager: bool) -> bool:
    if write_config or write_manager:
        return True
    return any(config.get(key) for key in _PYSUMMA_FILE_EDITS)


def _release_loaded_datasets(simulation: Any) -> None:
    """Drop datasets pysumma already closed.

    ``FileManager`` stores attributes, trial parameters, and the initial
    state after a ``with xr.open_dataset`` block. Garbage collection then
    closes those NetCDF ids again and prints ``NetCDF: Not a valid ID``.
    """
    manager = getattr(simulation, "manager", None)
    if manager is None:
        return
    for name in ("_local_attrs", "_trial_params", "_init_cond"):
        dataset = getattr(manager, name, None)
        if dataset is None:
            continue
        try:
            dataset.close()
        except Exception:
            pass
        setattr(manager, name, None)


def _prepare_simulation(
    ps: Any,
    summa_exe: str,
    config: Dict[str, Any],
    *,
    write_config: bool,
    write_manager: bool,
) -> Any:
    file_manager = config.get("file_manager")
    if not file_manager:
        raise ValueError("Ensemble member config is missing file_manager.")
    simulation = ps.Simulation(summa_exe, file_manager, False)
    # Member runs read the file manager already written to disk. Loading the
    # shared NetCDF settings in this process makes xarray close them twice.
    if _config_rewrites_files(config, write_config=write_config, write_manager=write_manager):
        simulation.initialize()
        simulation.apply_config(config)
        _release_loaded_datasets(simulation)
    return simulation


_SUMMA_SUCCESS = "FORTRAN STOP: finished simulation successfully."


def _finish_simulation(simulation: Any) -> None:
    """Wait for ``summa.exe`` and record its status without opening NetCDF.

    pysumma's ``Simulation.monitor`` opens every new output file on the
    calling thread. Ensemble workers finish together, so those opens overlap.
    The HDF5 library behind netCDF4 is not thread-safe, and that overlap
    segfaults the process. snowda reads the output files later, one member
    at a time.
    """
    process = getattr(simulation, "process", None)
    if process is None:
        if getattr(simulation, "status", None) in {"Success", "Error"}:
            return
        raise RuntimeError("SUMMA simulation has no process to wait on.")
    stdout, stderr = process.communicate()
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", "ignore")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "ignore")
    simulation.stdout = "" if stdout is None else stdout
    simulation.stderr = "" if stderr is None else stderr
    simulation.status = (
        "Success" if _SUMMA_SUCCESS in simulation.stdout else "Error"
    )
    simulation._output = None


def _run_prepared_simulation(name: str, simulation: Any, run_kwargs: Dict[str, Any]) -> None:
    log.info("SUMMA member %s running", name)
    # ``start`` launches the process. ``run`` would also open the NetCDF
    # output on this worker thread.
    simulation.start(**run_kwargs)
    _finish_simulation(simulation)


def _ensemble_summary(simulations: Dict[str, Any]) -> Dict[str, list[str]]:
    success: list[str] = []
    error: list[str] = []
    other: list[str] = []
    for name, simulation in simulations.items():
        status = getattr(simulation, "status", "Unknown")
        if status == "Success":
            success.append(name)
        elif status == "Error":
            error.append(name)
        else:
            other.append(name)
    return {"Success": success, "Error": error, "Other": other}


def run_ensemble(
    summa_exe: str,
    ens_config: Dict[str, Dict[str, str]],
    *,
    num_workers: int = 5,
    run_option: str = "local",
    run_suffix: Optional[str] = None,
    freq_restart: str = "d",
    write_config: bool = False,
    log_stdout: bool = False,
) -> Dict[str, Any]:
    """Run SUMMA members in a thread pool and return summary metadata.

    Each member is a separate ``summa.exe`` process. Threads only wait on
    those processes and do not open the NetCDF output. pysumma's dask
    ensemble is not used: its workers exit when spawned from a Jupyter
    kernel, and the gather then waits forever.
    """
    ps = _require_pysumma()
    total = len(ens_config)
    workers = max(1, min(num_workers, total or 1))
    log.info(
        "SUMMA ensemble starting: %s members, %s workers, suffix %s",
        total,
        workers,
        run_suffix,
    )
    write_manager = False
    simulations = {
        name: _prepare_simulation(
            ps,
            summa_exe,
            config,
            write_config=write_config,
            write_manager=write_manager,
        )
        for name, config in ens_config.items()
    }
    run_kwargs: Dict[str, Any] = {
        "run_option": run_option,
        "run_suffix": run_suffix,
        "freq_restart": freq_restart,
        "write_config": write_config,
        "write_manager": write_manager,
    }
    finished = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_prepared_simulation, name, simulation, run_kwargs): name
            for name, simulation in simulations.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception:
                finished += 1
                log.exception("SUMMA member %s raised (%s/%s)", name, finished, total)
                raise
            finished += 1
            status = getattr(simulations[name], "status", "Unknown")
            if status != "Success":
                errors += 1
                log.warning(
                    "SUMMA member %s finished with status %s (%s/%s)",
                    name,
                    status,
                    finished,
                    total,
                )
            else:
                log.info("SUMMA member %s complete (%s/%s)", name, finished, total)
    log.info("SUMMA ensemble finished: %s/%s successful", total - errors, total)
    summary = _ensemble_summary(simulations)
    if log_stdout:
        log.info("SUMMA ensemble summary: %s", summary)
        log.info("SUMMA stdout (001): %s", simulations["001"].stdout)
    return summary


def run_simulation(
    summa_exe: str,
    file_manager: str,
    *,
    run_option: str | None = None,
    run_suffix: str | None = None,
    freq_restart: str | None = None,
    write_config: bool | None = None,
    update_manager: bool | None = None,
    log_stdout: bool = True,
    sim_start_time: str | None = None,
    sim_end_time: str | None = None,
    sim_config: Dict[str, Any] | None = None,
    write_configuration_name: str | None = None,
) -> Any:
    """Run a single SUMMA simulation via pysumma."""
    ps = _require_pysumma()
    sim = ps.Simulation(summa_exe, file_manager)
    if sim_config:
        sim.apply_config(sim_config)
    if write_configuration_name:
        sim.run_suffix = write_configuration_name
        sim._write_configuration(name=write_configuration_name)
    if sim_start_time:
        sim.manager.set_option("simStartTime", sim_start_time)
    if sim_end_time:
        sim.manager.set_option("simEndTime", sim_end_time)
    if sim_start_time or sim_end_time:
        sim.manager.write()
    run_kwargs: Dict[str, Any] = {}
    if run_option is not None:
        run_kwargs["run_option"] = run_option
    if run_suffix is not None:
        run_kwargs["run_suffix"] = run_suffix
    if freq_restart is not None:
        run_kwargs["freq_restart"] = freq_restart
    if write_config is not None:
        run_kwargs["write_config"] = write_config
    if update_manager is not None:
        run_kwargs["write_manager"] = update_manager
    sim.run(**run_kwargs)
    if log_stdout:
        log.info("SUMMA stdout: %s", sim.stdout)
    return sim
