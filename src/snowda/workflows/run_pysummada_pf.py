from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import glob

import pandas as pd
import xarray as xr

from snowda.adapters.pysumma import run_ensemble
from snowda.config.pysummada import load_pysummada_config
from snowda.core.assimilation import DataAssimilationCore
from snowda.core.particle_filter import ParticleFilterCore
from snowda.io.fsca_obs import read_fsca_observations
from snowda.io.summa_ensemble import EnsembleFileWriter, SummaEnsembleReader, update_file_manager_option
from snowda.io.summa_fsca import calc_summa_fsca, prepare_summa_fsca_input
from snowda.utils.logging import configure_logging, get_logger


def _now_utc_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_pysummada_pf(config_path: str | Path) -> Path:
    cfg = load_pysummada_config(config_path)

    run_dir = _ensure_dir(cfg.output_dir / _now_utc_tag())
    log_file = run_dir / "pysummada_pf.log"
    configure_logging(level=logging.INFO, log_file=log_file)
    log = get_logger(__name__)

    log.info("snowda pysummada PF run starting")
    log.info("Config: %s", str(config_path))
    log.info("Run dir: %s", run_dir)

    config_out = run_dir / "config.resolved.json"
    config_out.write_text(json.dumps(cfg.__dict__, default=str, indent=2), encoding="utf-8")
    log.info("Wrote config snapshot: %s", config_out)

    file_writer = EnsembleFileWriter(cfg)
    file_writer.write_ensemble_files()
    if not file_writer.ens_file_managers:
        raise RuntimeError("Failed to generate ensemble file managers.")

    assimilation = DataAssimilationCore(cfg)
    ens_config = assimilation.create_ens_config(file_writer.ens_file_managers)
    run_period = assimilation.set_run_period()

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
    fsca_cv = float(getattr(cfg, "fsca_cv", 0.5))
    helbig_settings = getattr(cfg, "helbig_settings", None)
    snow_depth_var = getattr(cfg, "snow_depth_var", "scalarSnowDepth")

    particle_filter = ParticleFilterCore()
    reader = SummaEnsembleReader(cfg)

    conversions_log_run = []

    for idx, start_time in enumerate(run_period):
        end_time_current_run = start_time + cfg.time_delta
        log.info(
            "Starting ensemble run %s to %s", start_time, end_time_current_run
        )
        ens_config = assimilation.update_run_time(start_time, end_time_current_run)
        if idx != 0:
            ens_config = assimilation.update_warm_state(start_time)

        run_ensemble(
            str(cfg.summa_exe),
            ens_config,
            run_suffix=end_time_current_run.strftime("%Y_%m_%d"),
            log_stdout=True,
        )

        if fsca_obs_nc and summa_output_dir and fsca_output_dir:
            obs_df = read_fsca_observations(
                fsca_obs_nc, variable=str(fsca_obs_var), start_time=start_time, end_time=end_time_current_run
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

    prov = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": str(Path(config_path).resolve()),
        "run_dir": str(run_dir),
        "model_result_output_dir": str(cfg.model_result_output_dir),
        "assimilation_mode": assimilation_mode,
        "unit_conversions": conversions_log_run,
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
