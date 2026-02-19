from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Sequence
import shutil

import glob

import pandas as pd
import xarray as xr

from snowda.adapters.mizuroute import run_mizuroute, update_control_file
from snowda.adapters.pysumma import get_file_manager_value, run_ensemble
from snowda.config.pysummada import load_pysummada_config
from snowda.core.assimilation import DataAssimilationCore
from snowda.core.particle_filter import ParticleFilterCore
from snowda.io.fsca_obs import read_fsca_observations
from snowda.io.summa_ensemble import EnsembleFileWriter, SummaEnsembleReader, update_file_manager_option
from snowda.io.summa_fsca import calc_summa_fsca, prepare_summa_fsca_input
from snowda.utils.logging import configure_logging, get_logger
from snowda.workflows.esp import run_summa_esp


def _now_utc_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _copy_param_definition(cfg, run_dir: Path) -> bool:
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


def _find_summa_timestep_output(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("*_timestep.nc"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No SUMMA timestep output found in {output_dir}")
    return candidates[-1]


def _find_mizuroute_output(output_dir: Path, case_name: str, sim_start: datetime) -> Path:
    canonical = output_dir / f"{case_name}.h.{sim_start.strftime('%Y-%m-%d')}-00000.nc"
    if canonical.exists():
        return canonical
    globbed = sorted(output_dir.glob(f"{case_name}.h.*.nc"), key=lambda p: p.stat().st_mtime)
    if globbed:
        return globbed[-1]
    raise FileNotFoundError(f"Could not find mizuRoute output in {output_dir}")


def _locate_resource(
    explicit: Path | None, fallback_dirs: Sequence[Path], filename: str
) -> Path | None:
    if explicit:
        return explicit
    for folder in fallback_dirs:
        candidate = folder / filename
        if candidate.exists():
            return candidate
    return None


def _copy_optional_resource(
    source: Path | None,
    fallback_dirs: Sequence[Path],
    destination: Path,
    label: str,
    required: bool = False,
) -> None:
    resource = _locate_resource(source, fallback_dirs, destination.name)
    if not resource:
        if required:
            raise FileNotFoundError(
                f"Could not locate required {label} (looked in {fallback_dirs})."
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resource, destination)


def _run_mizuroute_after_summa(
    cfg, mizu_cfg, run_dir: Path, sim_start: datetime, sim_end: datetime
) -> Path:
    output_path = Path(
        get_file_manager_value(str(cfg.base_file_manager), "outputPath").replace("'", "")
    )
    summa_output = _find_summa_timestep_output(output_path)

    mizu_output_dir = Path(mizu_cfg.output_dir)
    mizu_output_dir.mkdir(parents=True, exist_ok=True)

    control_parent = Path(mizu_cfg.control_file).parent
    _copy_optional_resource(
        mizu_cfg.param_nml,
        [mizu_cfg.ancil_dir, control_parent],
        mizu_output_dir / "param.nml.default",
        "mizuRoute param.nml.default",
        required=True,
    )
    _copy_optional_resource(
        mizu_cfg.topology_nc,
        [mizu_cfg.ancil_dir, control_parent],
        mizu_output_dir / "topology.nc",
        "mizuRoute topology.nc",
    )

    control_file = run_dir / "mizuroute" / "mizuroute_deterministic.control"
    control_settings = {
        "fname_qsim": summa_output.name,
        "fname_state_in": f"{mizu_cfg.case_name}.r.{sim_start.strftime('%Y-%m-%d')}-00000.nc",
        "case_name": mizu_cfg.case_name,
        "sim_start": sim_start.strftime("%Y-%m-%d %H:%M:%S"),
        "sim_end": (sim_end - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "ancil_dir": str(mizu_cfg.ancil_dir) + "/",
        "input_dir": str(summa_output.parent) + "/",
        "output_dir": str(mizu_output_dir) + "/",
        "restart_dir": str(mizu_cfg.restart_dir) + "/",
        "restart_write": mizu_cfg.restart_write,
    }
    update_control_file(mizu_cfg.control_file, control_file, control_settings)
    run_mizuroute(mizu_cfg.executable, control_file, log_stdout=mizu_cfg.log_output)
    routed_generated = _find_mizuroute_output(mizu_output_dir, mizu_cfg.case_name, sim_start)
    mizuroute_output = mizu_output_dir / "mizuroute_deterministic.nc"
    routed_generated.rename(mizuroute_output)
    return mizuroute_output


def run_pysummada_pf(config_path: str | Path) -> Path:
    cfg = load_pysummada_config(config_path)

    run_dir = _ensure_dir(cfg.output_dir / _now_utc_tag())
    _copy_param_definition(cfg, run_dir)
    log_file = run_dir / "pysummada_pf.log"
    configure_logging(level=logging.INFO, log_file=log_file)
    log = get_logger(__name__)

    log.info("snowda pysummada PF run starting")
    log.info("Config: %s", str(config_path))
    log.info("Run dir: %s", run_dir)

    config_out = run_dir / "config.resolved.json"
    config_out.write_text(json.dumps(cfg.__dict__, default=str, indent=2), encoding="utf-8")
    log.info("Wrote config snapshot: %s", config_out)

    run_mode = getattr(cfg, "run_mode", "da")
    deterministic_esp = run_mode == "deterministic_esp"
    run_da = bool(cfg.run_da)
    esp_records: list[Any] = []
    conversions_log_run: list[Any] = []

    assimilation = DataAssimilationCore(cfg)
    file_writer = EnsembleFileWriter(cfg)
    file_writer.write_ensemble_files()
    if not file_writer.ens_file_managers:
        raise RuntimeError("Failed to generate ensemble file managers.")
    assimilation.create_ens_config(file_writer.ens_file_managers)
    if run_da:
        run_period = assimilation.set_run_period()
    else:
        run_period = [cfg.start_time]

    if deterministic_esp:
        assimilation_mode = "deterministic"
        log.info("Deterministic ESP run requested; skipping ensemble data assimilation.")
    elif not run_da:
        assimilation_mode = "disabled"
        log.info("Data assimilation disabled; running SUMMA without PF loop.")
    else:
        assimilation_mode = getattr(cfg, "assimilation_mode", "fsca_only")
        if assimilation_mode not in {"fsca_only"}:
            raise ValueError(
                f"Unsupported assimilation_mode '{assimilation_mode}'. "
                "Use 'fsca_only' for now."
            )

    fsca_obs_nc = getattr(cfg, "fsca_obs_nc", None)
    fsca_obs_var = getattr(cfg, "fsca_obs_var", "fsca")
    summa_output_dir = getattr(cfg, "summa_output_dir", None)
    summa_output_suffix = getattr(cfg, "summa_output_suffix", "day.nc")
    fsca_output_dir = getattr(cfg, "fsca_output_dir", None)
    fsca_model = getattr(cfg, "fsca_model", "ssnowd")
    fsca_cv_raw = getattr(cfg, "fsca_cv", None)
    fsca_cv = float(fsca_cv_raw) if fsca_cv_raw is not None else 0.5
    helbig_settings = getattr(cfg, "helbig_settings", None)
    snow_depth_var = getattr(cfg, "snow_depth_var", "scalarSnowDepth")

    particle_filter = ParticleFilterCore() if run_da else None
    reader = SummaEnsembleReader(cfg) if run_da else None
    perform_assimilation = run_da and not deterministic_esp

    last_end_time = None
    for idx, start_time in enumerate(run_period):
        if run_da:
            end_time_current_run = start_time + cfg.time_delta
        else:
            end_time_current_run = cfg.end_time
        last_end_time = end_time_current_run
        log.info(
            "Starting ensemble run %s to %s", start_time, end_time_current_run
        )
        current_run_config = assimilation.update_run_time(start_time, end_time_current_run)
        if idx != 0 and run_da:
            current_run_config = assimilation.update_warm_state(start_time)

        run_ensemble(
            str(cfg.summa_exe),
            current_run_config,
            run_suffix=end_time_current_run.strftime("%Y_%m_%d"),
            log_stdout=True,
        )

        if not perform_assimilation:
            continue

        if fsca_obs_nc and summa_output_dir and fsca_output_dir:
            obs_df = read_fsca_observations(
                fsca_obs_nc,
                variable=str(fsca_obs_var),
                start_time=start_time,
                end_time=end_time_current_run,
            )
            if obs_df.empty:
                log.info("No fSCA observations for %s", end_time_current_run)
                continue

            fsca_ens_dir = Path(fsca_output_dir) / end_time_current_run.strftime("%Y_%m_%d")
            fsca_ens_dir.mkdir(parents=True, exist_ok=True)

            modeled_cols = {}
            conversions_log = []
            hru_list = obs_df.index.unique(level="hru").tolist()
            hru_index = reader.get_hru_index(hru_list)

            for ens_label in cfg.ens_label_array:
                summa_files = glob.glob(
                    str(Path(summa_output_dir) / f"{ens_label}*{summa_output_suffix}")
                )
                if not summa_files:
                    log.warning("No SUMMA outputs found for %s", ens_label)
                    continue

                fsca_input_path = fsca_ens_dir / f"{ens_label}_fsca_input.nc"
                fsca_output_path = fsca_ens_dir / f"{ens_label}_fsca_output.nc"
                fsca_input_path, conversions = prepare_summa_fsca_input(
                    summa_files, fsca_input_path
                )
                conversions_log.extend(conversions)
                calc_summa_fsca(
                    fsca_input_path,
                    fsca_cv,
                    fsca_output_path,
                    fsca_model=str(fsca_model),
                    helbig_settings=helbig_settings,
                    snow_depth_var=str(snow_depth_var),
                )

                with xr.open_dataset(fsca_output_path) as ds:
                    if "fsca" not in ds:
                        log.warning("fSCA variable not found in %s", fsca_output_path)
                        continue
                    ds_t = ds["fsca"].sel(time=end_time_current_run).to_dataframe()
                    ds_t.index.names = ["hru", "time"]
                modeled_cols[ens_label] = ds_t["fsca"]

            if not modeled_cols:
                log.info("No modeled fSCA data found for %s", end_time_current_run)
                continue

            modeled_df = pd.concat(modeled_cols, axis=1)
            weights_fsca = particle_filter.update_particle_weights_fsca(
                modeled_df,
                obs_df,
                z_var=1.0,
                end_date_current_run=end_time_current_run,
                hru_index=hru_index,
                ens_label_array=list(modeled_cols.keys()),
            )
            resample_indices = particle_filter.resample_indices(
                weights_fsca,
                num_ens=len(modeled_cols),
                n_eff_threshold=cfg.n_eff_threshold,
            )
            if resample_indices is not None:
                warm_states = assimilation.create_warm_state_list(end_time_current_run)
                for i, resampled_idx in enumerate(resample_indices):
                    fm_path = file_writer.ens_file_managers[i]
                    update_file_manager_option(
                        fm_path, "initConditionFile", f"'{warm_states[resampled_idx]}'"
                    )
            weights_out = fsca_ens_dir / "fsca_weights.csv"
            weights_fsca.to_csv(weights_out)
            log.info("Wrote fSCA weights: %s", weights_out)
            conversions_log_run.extend(conversions_log)

    mizuroute_output: str | None = None
    if (
        run_mode == "deterministic"
        and cfg.esp
        and cfg.esp.run_mizuroute
        and cfg.esp.mizuroute
        and last_end_time
    ):
        log.info("Deterministic SUMMA run complete; starting mizuRoute.")
        mizuroute_output = str(
            _run_mizuroute_after_summa(cfg, cfg.esp.mizuroute, run_dir, cfg.start_time, last_end_time)
        )

    if cfg.esp and cfg.esp.enabled:
        log.info("ESP is enabled; starting ESP traces.")
        esp_records = run_summa_esp(cfg, cfg.esp, run_dir)
        log.info("ESP complete. Generated %s traces.", len(esp_records))

    prov = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": str(Path(config_path).resolve()),
        "run_dir": str(run_dir),
        "model_result_output_dir": str(cfg.model_result_output_dir),
        "run_mode": run_mode,
        "assimilation_mode": assimilation_mode,
        "unit_conversions": conversions_log_run,
        "esp_enabled": bool(cfg.esp and cfg.esp.enabled),
        "esp_trace_count": len(esp_records),
        "esp_traces": [record.__dict__ for record in esp_records],
        "mizuroute_output": mizuroute_output,
    }
    prov_out = run_dir / "provenance.json"
    prov_out.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    log.info("Wrote provenance: %s", prov_out)

    log.info("snowda pysummada PF run complete")
    return run_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a pysummada-style SUMMA particle filter workflow."
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    run_pysummada_pf(args.config)


if __name__ == "__main__":
    main()
