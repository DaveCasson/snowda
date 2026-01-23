from __future__ import annotations

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


def run_ensemble(
    summa_exe: str,
    ens_config: Dict[str, Dict[str, str]],
    *,
    num_workers: int = 5,
    run_option: str = "local",
    run_suffix: Optional[str] = None,
    freq_restart: str = "d",
    write_config: bool = False,
    update_manager: bool = True,
    log_stdout: bool = False,
) -> Dict[str, Any]:
    """Run a SUMMA ensemble via pysumma and return summary metadata."""
    ps = _require_pysumma()
    ensemble = ps.Ensemble(summa_exe, ens_config, num_workers=num_workers)
    ensemble.run(
        run_option=run_option,
        run_suffix=run_suffix,
        freq_restart=freq_restart,
        write_config=write_config,
        update_manager=update_manager,
    )
    summary = ensemble.summary()
    if log_stdout:
        log.info("SUMMA ensemble summary: %s", summary)
        log.info("SUMMA stdout (001): %s", ensemble.simulations["001"].stdout)
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
