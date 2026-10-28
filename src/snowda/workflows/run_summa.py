from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from snowda.adapters.pysumma import run_ensemble, run_simulation
from snowda.config.loader import as_config_dict, config_label
from snowda.config.pysummada import load_pysummada_config, validate_deterministic_config
from snowda.utils.logging import configure_logging, get_logger
from snowda.workflows.evaluate_streamflow import evaluate_deterministic_run
from snowda.workflows.summa_common import (
    apply_run_time,
    prepare_ensemble,
    redirect_summa_output,
    run_mizuroute_after_summa,
    setup_run_directory,
    summa_run_output_dir,
    write_provenance,
)


def run_summa(config: str | Path | Mapping[str, Any]) -> Path:
    raw = as_config_dict(config)
    label = config_label(config)
    if raw.get("file_manager") and not raw.get("base_file_manager"):
        return _run_summa_flat(raw, label)
    return _run_summa_deterministic(raw, label)


def _run_summa_deterministic(config: Mapping[str, Any], label: str) -> Path:
    cfg = load_pysummada_config(config)
    validate_deterministic_config(cfg)
    run_dir, log = setup_run_directory(
        cfg,
        label,
        log_filename="summa_deterministic.log",
        logger_name=__name__,
        start_message="snowda SUMMA deterministic run starting",
    )

    file_writer, assimilation = prepare_ensemble(cfg)
    log.info("Starting SUMMA run %s to %s", cfg.start_time, cfg.end_time)
    current_run_config = apply_run_time(
        cfg,
        assimilation,
        file_writer.ens_file_managers or [],
        cfg.start_time,
        cfg.end_time,
    )
    summa_output_dir = redirect_summa_output(
        file_writer.ens_file_managers or [],
        current_run_config,
        summa_run_output_dir(run_dir),
    )
    log.info("SUMMA outputPath: %s", summa_output_dir)
    run_ensemble(
        str(cfg.summa_exe),
        current_run_config,
        num_workers=cfg.num_workers,
        run_suffix=cfg.end_time.strftime("%Y_%m_%d"),
        log_stdout=True,
    )

    mizuroute_output: str | None = None
    if cfg.mizuroute:
        log.info("Deterministic SUMMA run complete; starting mizuRoute.")
        mizuroute_output = str(
            run_mizuroute_after_summa(
                cfg,
                cfg.mizuroute,
                run_dir,
                cfg.start_time,
                cfg.end_time,
                summa_output_dir=summa_output_dir,
            )
        )

    evaluation_metrics: dict[str, Any] | None = None
    if cfg.evaluation is not None:
        if not mizuroute_output:
            raise ValueError(
                "`evaluation` is configured but no mizuRoute output was produced."
            )
        log.info("Evaluating routed streamflow against observations.")
        evaluation_metrics = evaluate_deterministic_run(
            mizuroute_output, cfg.evaluation, run_dir
        )

    write_provenance(
        run_dir,
        {
            "config_path": label,
            "model_result_output_dir": str(cfg.model_result_output_dir),
            "variant": "deterministic",
            "summa_output_dir": str(summa_output_dir),
            "mizuroute_output": mizuroute_output,
            "evaluation": evaluation_metrics,
        },
        log,
    )
    log.info("snowda SUMMA deterministic run complete")
    return run_dir


def _now_utc_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _normalize_path(raw: str, base_dir: Path | None, *, allow_relative: bool) -> str:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        if not allow_relative:
            raise ValueError(f"Path must be absolute (got: {raw})")
        if base_dir is None:
            raise ValueError(
                f"Relative path {raw!r} needs `_meta.config_dir`. "
                "Load the YAML with snowda.config.loader.load_config() before editing it."
            )
        p = (base_dir / p).resolve()
    return str(p)


def _normalize_flat_config(cfg: Dict[str, Any], base_dir: Path | None) -> Dict[str, Any]:
    if "summa" in cfg and "paths" in cfg:
        return cfg

    paths = dict(cfg.get("paths", {}))
    if "output_dir" in cfg and "output_dir" not in paths:
        paths["output_dir"] = _normalize_path(
            str(cfg.get("output_dir")), base_dir, allow_relative=True
        )
    if "log_file" in cfg and "log_file" not in paths:
        paths["log_file"] = _normalize_path(
            str(cfg.get("log_file")), base_dir, allow_relative=True
        )

    summa = dict(cfg.get("summa", {}))
    if "summa_exe" in cfg and "summa_exe" not in summa:
        summa["summa_exe"] = _normalize_path(
            str(cfg.get("summa_exe")), base_dir, allow_relative=False
        )
    if "file_manager" in cfg and "file_manager" not in summa:
        summa["file_manager"] = _normalize_path(
            str(cfg.get("file_manager")), base_dir, allow_relative=False
        )

    run_options = dict(cfg.get("run_options", {}))
    for key in (
        "run_option",
        "run_suffix",
        "freq_restart",
        "write_config",
        "update_manager",
        "sim_start_time",
        "sim_end_time",
    ):
        if key in cfg and key not in run_options:
            run_options[key] = cfg.get(key)

    cfg["paths"] = paths
    cfg["summa"] = summa
    cfg["run_options"] = run_options
    return cfg


def _run_summa_flat(config: Mapping[str, Any], label: str) -> Path:
    cfg: Dict[str, Any] = dict(config)
    meta = cfg.get("_meta") or {}
    config_dir = meta.get("config_dir") if isinstance(meta, Mapping) else None
    base_dir = Path(str(config_dir)) if config_dir else None
    cfg = _normalize_flat_config(cfg, base_dir)

    paths = cfg.get("paths", {})
    output_dir = paths.get("output_dir")
    if not output_dir:
        raise ValueError("Config must define `output_dir` (or `paths.output_dir`).")

    run_dir = _ensure_dir(Path(output_dir) / _now_utc_tag())

    log_file_cfg = paths.get("log_file")
    log_file = None
    if isinstance(log_file_cfg, str) and log_file_cfg:
        log_file = run_dir / Path(log_file_cfg).name

    configure_logging(level=logging.INFO, log_file=log_file)
    log = get_logger(__name__)
    log.warning(
        "Flat `file_manager` SUMMA configs are deprecated. "
        "Prefer pysummada YAML examples such as "
        "runs/examples/ross_summa_deterministic_era5.yaml."
    )
    log.info("snowda SUMMA run starting")
    log.info("Config: %s", label)
    log.info("Run dir: %s", run_dir)

    config_out = run_dir / "config.resolved.yaml"
    try:
        import yaml

        with config_out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        log.info("Wrote config snapshot: %s", config_out)
    except Exception as e:
        log.warning("Failed to write config snapshot (%s): %s", config_out, e)

    summa_cfg = cfg.get("summa", {})
    summa_exe = summa_cfg.get("summa_exe")
    file_manager = summa_cfg.get("file_manager")
    if not summa_exe or not file_manager:
        raise ValueError("Config must define `summa_exe` and `file_manager`.")
    # Guard against nested .pysumma configurations causing repeated settingsPath
    fm_path = Path(str(file_manager)).resolve()
    if ".pysumma" in fm_path.parts:
        auto_cleanup = cfg.get("auto_cleanup_pysumma", True)
        if not auto_cleanup:
            raise ValueError(
                "file_manager points inside a .pysumma config directory. "
                "Set auto_cleanup_pysumma: true to auto-clean, or "
                "use the base fileManager.txt outside .pysumma."
            )
        idx = fm_path.parts.index(".pysumma")
        pysumma_dir = Path(*fm_path.parts[: idx + 1])
        base_settings_dir = Path(*fm_path.parts[:idx])
        base_file_manager = base_settings_dir / fm_path.name
        if not base_file_manager.exists():
            alt = base_settings_dir / "fileManager.txt"
            if alt.exists():
                base_file_manager = alt
        if pysumma_dir.exists():
            shutil.rmtree(pysumma_dir)
            log.warning("Removed nested .pysumma directory: %s", pysumma_dir)
        if not base_file_manager.exists():
            raise FileNotFoundError(
                f"Base file manager not found after cleanup: {base_file_manager}"
            )
        file_manager = str(base_file_manager)

    run_opts = cfg.get("run_options", {})
    sim_start_time = cfg.get("sim_start_time")
    sim_end_time = cfg.get("sim_end_time")
    sim_config = cfg.get("sim_config") or {}
    log.info("sim_start_time=%s sim_end_time=%s", sim_start_time, sim_end_time)
    if not sim_config:
        manager_cfg = {}
        forcing_dir = cfg.get("summa_forcing_dir")
        output_dir_cfg = cfg.get("output_dir")
        forcing_list_file = cfg.get("forcing_list_file")
        tm_zone_info = cfg.get("tmZoneInfo")
        if forcing_dir:
            manager_cfg["forcingPath"] = str(forcing_dir) + "/"
        if output_dir_cfg:
            manager_cfg["outputPath"] = str(output_dir_cfg) + "/"
        if forcing_list_file:
            manager_cfg["forcingListFile"] = str(forcing_list_file)
        if tm_zone_info:
            manager_cfg["tmZoneInfo"] = str(tm_zone_info)
        if sim_start_time:
            manager_cfg["simStartTime"] = sim_start_time
        if sim_end_time:
            manager_cfg["simEndTime"] = sim_end_time
        if manager_cfg:
            sim_config = {"manager": manager_cfg}

    write_configuration_name = (
        run_opts.get("write_configuration_name") or cfg.get("write_configuration_name")
    )
    def _validate_summa_time(value: str, key: str) -> str:
        try:
            # SUMMA expects 'YYYY-MM-DD HH:MM'
            datetime.strptime(value, "%Y-%m-%d %H:%M")
        except ValueError as exc:
            raise ValueError(
                f"{key} must be in 'YYYY-MM-DD HH:MM' format (got: {value})"
            ) from exc
        return value

    if sim_start_time:
        sim_start_time = _validate_summa_time(str(sim_start_time), "sim_start_time")
    if sim_end_time:
        sim_end_time = _validate_summa_time(str(sim_end_time), "sim_end_time")

    if write_configuration_name and ".pysumma" in str(file_manager):
        log.warning(
            "Skipping write_configuration_name because file_manager is already "
            "inside a .pysumma config directory: %s",
            file_manager,
        )
        write_configuration_name = None
    # Times are applied inside pysumma.run_simulation to ensure the same
    # in-memory manager is used for the run and any config writes.
    if cfg.get("debug_pause"):
        # Force a debugger stop before the SUMMA subprocess starts.
        breakpoint()
    if cfg.get("debug_skip_run"):
        log.info("Debug skip run enabled; returning before SUMMA execution.")
        return run_dir
    sim = run_simulation(
        str(summa_exe),
        str(file_manager),
        run_option=run_opts.get("run_option"),
        run_suffix=run_opts.get("run_suffix"),
        freq_restart=run_opts.get("freq_restart"),
        write_config=run_opts.get("write_config"),
        update_manager=run_opts.get("update_manager"),
        log_stdout=True,
        sim_start_time=sim_start_time,
        sim_end_time=sim_end_time,
        sim_config=sim_config,
        write_configuration_name=write_configuration_name,
    )
    if cfg.get("show_summa_stdout"):
        print(getattr(sim, "stdout", ""))

    prov = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": label,
        "run_dir": str(run_dir),
        "summa_stdout": getattr(sim, "stdout", None),
        "sim_start_time": sim_start_time,
        "sim_end_time": sim_end_time,
        "show_summa_stdout": bool(cfg.get("show_summa_stdout")),
        "write_configuration_name": write_configuration_name,
        "sim_config_applied": bool(sim_config),
    }
    prov_out = run_dir / "provenance.json"
    prov_out.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    log.info("Wrote provenance: %s", prov_out)

    log.info("snowda SUMMA run complete")
    return run_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a deterministic SUMMA simulation from a YAML config.")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    run_summa(args.config)


if __name__ == "__main__":
    main()
