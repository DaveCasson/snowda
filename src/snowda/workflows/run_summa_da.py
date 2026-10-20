from __future__ import annotations

import glob
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from snowda.adapters.pysumma import run_ensemble
from snowda.config.pysummada import (
    AssimilationConfig,
    PysummadaConfig,
    load_pysummada_config,
    validate_da_config,
)
from snowda.core.assimilation import (
    DataAssimilationCore,
    assimilation_segments,
    resolve_assimilation_dates,
)
from snowda.core.particle_filter import ParticleFilterCore
from snowda.core.swe_assimilation import LumpedSwe, lumped_swe_by_date
from snowda.io.fsca_obs import read_fsca_observations
from snowda.io.summa_ensemble import SummaEnsembleReader, update_file_manager_option
from snowda.io.summa_fsca import calc_summa_fsca, prepare_summa_fsca_input
from snowda.io.swe_obs import read_lumped_member_swe, read_model_equivalent_swe
from snowda.workflows.esp import run_esp_traces
from snowda.workflows.summa_common import (
    apply_run_time,
    apply_segment_initial_conditions,
    prepare_ensemble,
    redirect_summa_output,
    resample_routing_restarts,
    route_ensemble_segment,
    setup_run_directory,
    summa_run_output_dir,
    write_provenance,
)

SUMMA_TIMESTEP_SUFFIX = "timestep.nc"


def _freq_restart(cfg: PysummadaConfig) -> str:
    return cfg.summa_args.get("-r") or cfg.summa_args.get("freq_restart") or "d"


def _write_swe_record(run_dir: Path, observation: LumpedSwe) -> Path:
    out_dir = run_dir / "assimilation"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "assimilation_date": [observation.date.strftime("%Y-%m-%d %H:%M")]
            * len(observation.samples),
            "station_id": [sample.station_id for sample in observation.samples],
            "time": [sample.time.strftime("%Y-%m-%d %H:%M") for sample in observation.samples],
            "swe_model_equivalent": [sample.value for sample in observation.samples],
            "lumped_mean": [observation.value] * len(observation.samples),
        }
    )
    path = out_dir / f"{observation.date.strftime('%Y_%m_%d')}_swe_obs.csv"
    frame.to_csv(path, index=False)
    return path


def _run_swe_assimilation(
    cfg: PysummadaConfig,
    assimilation: DataAssimilationCore,
    ens_file_managers: list[str],
    particle_filter: ParticleFilterCore,
    run_dir: Path,
    summa_output_dir: Path,
    run_log: Any,
) -> list[dict[str, Any]]:
    schedule = cfg.assimilation
    if schedule is None:
        raise ValueError("SWE assimilation requires an `assimilation` section.")
    dates = resolve_assimilation_dates(
        cfg.start_time,
        cfg.end_time,
        schedule.dates,
        schedule.frequency_days,
        schedule.months,
    )
    segments = assimilation_segments(cfg.start_time, cfg.end_time, dates)
    times, station_ids, values, flags, _units = read_model_equivalent_swe(
        schedule.swe_nc, variable=schedule.swe_var
    )
    observations = lumped_swe_by_date(
        times,
        station_ids,
        values,
        flags,
        dates,
        schedule.obs_window_days,
    )
    labels = cfg.ens_label_array
    updates: list[dict[str, Any]] = []
    previous_resampled = False
    routing_restarts: dict[str, Path | None] = {}
    for index, (segment_start, segment_end, do_assimilate) in enumerate(segments):
        run_log.info("Starting ensemble run %s to %s", segment_start, segment_end)
        current_run_config = apply_run_time(
            cfg,
            assimilation,
            ens_file_managers,
            segment_start,
            segment_end,
        )
        apply_segment_initial_conditions(
            labels,
            ens_file_managers,
            current_run_config,
            segment_start,
            is_first=index == 0,
            keep_existing=previous_resampled,
        )
        run_ensemble(
            str(cfg.summa_exe),
            current_run_config,
            run_suffix=segment_end.strftime("%Y_%m_%d"),
            freq_restart=_freq_restart(cfg),
            log_stdout=True,
        )
        if cfg.mizuroute is not None:
            route_ensemble_segment(
                cfg.mizuroute,
                run_dir,
                labels,
                summa_output_dir,
                segment_start,
                segment_end,
                routing_restarts,
                require_restart=index < len(segments) - 1,
            )
        previous_resampled = False
        if not do_assimilate:
            continue
        observation = observations.get(segment_end)
        if observation is None:
            run_log.info("No usable model-equivalent SWE for %s", segment_end)
            continue
        record_path = _write_swe_record(run_dir, observation)
        resampled = _update_swe_particles(
            cfg,
            schedule,
            assimilation,
            ens_file_managers,
            particle_filter,
            observation,
            summa_output_dir,
            run_dir,
            run_log,
            routing_restarts,
        )
        previous_resampled = resampled
        updates.append(
            {
                "date": segment_end.isoformat(),
                "lumped_swe": observation.value,
                "n_samples": len(observation.samples),
                "record": str(record_path),
                "resampled": resampled,
            }
        )
    return updates


def _update_swe_particles(
    cfg: PysummadaConfig,
    schedule: AssimilationConfig,
    assimilation: DataAssimilationCore,
    ens_file_managers: list[str],
    particle_filter: ParticleFilterCore,
    observation: LumpedSwe,
    summa_output_dir: Path,
    run_dir: Path,
    run_log: Any,
    routing_restarts: dict[str, Path | None],
) -> bool:
    output_dir = summa_output_dir
    suffix = cfg.summa_output_suffix or SUMMA_TIMESTEP_SUFFIX
    member_values: dict[str, float] = {}
    hru_id: int | None = None
    for label in cfg.ens_label_array:
        member_hru, value, _units = read_lumped_member_swe(
            output_dir,
            label,
            variable=schedule.summa_swe_var,
            when=observation.date,
            file_suffix=suffix,
        )
        if hru_id is None:
            hru_id = member_hru
        elif member_hru != hru_id:
            raise ValueError(
                f"Member {label} hruId {member_hru} does not match {hru_id}."
            )
        member_values[label] = value
    if hru_id is None:
        raise ValueError("No ensemble members were configured.")
    index = pd.MultiIndex.from_tuples([(hru_id, observation.date)], names=["hru", "time"])
    particles = pd.DataFrame(member_values, index=index)
    measurements = pd.DataFrame(
        {"hruId": [hru_id], "meas_snw": [observation.value]},
        index=index,
    )
    hru_index = pd.DataFrame({"hruId": [hru_id]})
    weights = particle_filter.update_particle_weights(
        particles,
        measurements,
        z_var=schedule.z_var,
        end_date_current_run=observation.date,
        hru_index=hru_index,
        ens_label_array=list(member_values),
        obs_column="meas_snw",
    )
    weight_dir = run_dir / "assimilation"
    weight_dir.mkdir(parents=True, exist_ok=True)
    weight_path = weight_dir / f"{observation.date.strftime('%Y_%m_%d')}_swe_weights.csv"
    weights.to_csv(weight_path)
    run_log.info("Wrote SWE weights: %s", weight_path)
    return _resample_member_states(
        cfg,
        assimilation,
        ens_file_managers,
        particle_filter,
        weights,
        observation.date,
        list(member_values),
        routing_restarts,
    )


def _resample_member_states(
    cfg: PysummadaConfig,
    assimilation: DataAssimilationCore,
    ens_file_managers: list[str],
    particle_filter: ParticleFilterCore,
    weights: pd.DataFrame,
    when: datetime,
    member_labels: list[str],
    routing_restarts: dict[str, Path | None] | None = None,
) -> bool:
    """Copy donor SUMMA and mizuRoute restarts under SIR.

    SIS leaves each particle's states in place. A particle is the member's
    SUMMA restart together with that same member's mizuRoute restart.
    """
    if cfg.pf_scheme != "sir":
        return False
    resample_indices = particle_filter.resample_indices(
        weights,
        num_ens=len(member_labels),
        n_eff_threshold=cfg.n_eff_threshold,
    )
    if resample_indices is None:
        return False
    managers = dict(zip(cfg.ens_label_array, ens_file_managers))
    warm_states = dict(
        zip(cfg.ens_label_array, assimilation.create_warm_state_list(when))
    )
    for member_index, donor_index in enumerate(resample_indices):
        member = member_labels[member_index]
        donor = member_labels[int(donor_index)]
        update_file_manager_option(
            managers[member],
            "initConditionFile",
            f"'{warm_states[donor]}'",
        )
    resample_routing_restarts(routing_restarts or {}, member_labels, resample_indices)
    return True


def _run_fsca_assimilation(
    cfg: PysummadaConfig,
    assimilation: DataAssimilationCore,
    ens_file_managers: list[str],
    particle_filter: ParticleFilterCore,
    reader: SummaEnsembleReader,
    conversions_log_run: list[Any],
    summa_output_dir: Path,
    run_log: Any,
) -> None:
    run_period = assimilation.set_run_period()
    fsca_obs_nc = cfg.fsca_obs_nc
    fsca_obs_var = cfg.fsca_obs_var or "fsca"
    summa_output_suffix = cfg.summa_output_suffix or SUMMA_TIMESTEP_SUFFIX
    fsca_output_dir = cfg.fsca_output_dir
    fsca_model = cfg.fsca_model or "ssnowd"
    fsca_cv = float(cfg.fsca_cv) if cfg.fsca_cv is not None else 0.5
    helbig_settings = cfg.helbig_settings
    snow_depth_var = cfg.snow_depth_var or "scalarSnowDepth"
    labels = cfg.ens_label_array
    previous_resampled = False
    routing_restarts: dict[str, Path | None] = {}

    for idx, start_time in enumerate(run_period):
        end_time_current_run = start_time + cfg.time_delta
        run_log.info("Starting ensemble run %s to %s", start_time, end_time_current_run)
        current_run_config = apply_run_time(
            cfg,
            assimilation,
            ens_file_managers,
            start_time,
            end_time_current_run,
        )
        apply_segment_initial_conditions(
            labels,
            ens_file_managers,
            current_run_config,
            start_time,
            is_first=idx == 0,
            keep_existing=previous_resampled,
        )

        run_ensemble(
            str(cfg.summa_exe),
            current_run_config,
            run_suffix=end_time_current_run.strftime("%Y_%m_%d"),
            log_stdout=True,
        )
        if cfg.mizuroute is not None:
            route_ensemble_segment(
                cfg.mizuroute,
                run_dir,
                labels,
                summa_output_dir,
                start_time,
                end_time_current_run,
                routing_restarts,
                require_restart=idx < len(run_period) - 1,
            )
        previous_resampled = False

        if not (fsca_obs_nc and summa_output_dir and fsca_output_dir):
            continue

        obs_df = read_fsca_observations(
            fsca_obs_nc,
            variable=str(fsca_obs_var),
            start_time=start_time,
            end_time=end_time_current_run,
        )
        if obs_df.empty:
            run_log.info("No fSCA observations for %s", end_time_current_run)
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
                run_log.warning("No SUMMA outputs found for %s", ens_label)
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
                    run_log.warning("fSCA variable not found in %s", fsca_output_path)
                    continue
                ds_t = ds["fsca"].sel(time=end_time_current_run).to_dataframe()
                ds_t.index.names = ["hru", "time"]
            modeled_cols[ens_label] = ds_t["fsca"]

        if not modeled_cols:
            run_log.info("No modeled fSCA data found for %s", end_time_current_run)
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
        previous_resampled = _resample_member_states(
            cfg,
            assimilation,
            ens_file_managers,
            particle_filter,
            weights_fsca,
            end_time_current_run,
            list(modeled_cols.keys()),
            routing_restarts,
        )
        weights_out = fsca_ens_dir / "fsca_weights.csv"
        weights_fsca.to_csv(weights_out)
        run_log.info("Wrote fSCA weights: %s", weights_out)
        conversions_log_run.extend(conversions_log)


@dataclass
class AssimilationRun:
    run_dir: Path
    log: Any
    swe_updates: list[dict[str, Any]]
    unit_conversions: list[Any]
    assimilation_mode: str


def run_summa_assimilation(
    cfg: PysummadaConfig,
    config_path: str | Path,
    *,
    log_filename: str,
    logger_name: str,
    start_message: str,
) -> AssimilationRun:
    """Carry the ensemble through the run, updating weights at assimilation dates."""
    run_dir, log = setup_run_directory(
        cfg,
        config_path,
        log_filename=log_filename,
        logger_name=logger_name,
        start_message=start_message,
    )

    file_writer, assimilation = prepare_ensemble(cfg)
    if file_writer.ens_file_managers is None:
        raise RuntimeError("Failed to generate ensemble file managers.")
    ens_file_managers = file_writer.ens_file_managers
    summa_output_dir = redirect_summa_output(
        ens_file_managers,
        assimilation.ens_config,
        summa_run_output_dir(run_dir),
    )
    log.info("SUMMA outputPath: %s", summa_output_dir)
    particle_filter = ParticleFilterCore()
    reader = SummaEnsembleReader(cfg)
    conversions_log_run: list[Any] = []
    swe_updates: list[dict[str, Any]] = []
    assimilation_mode = cfg.assimilation_mode or "fsca_only"

    if assimilation_mode == "swe":
        swe_updates = _run_swe_assimilation(
            cfg,
            assimilation,
            ens_file_managers,
            particle_filter,
            run_dir,
            summa_output_dir,
            log,
        )
    else:
        _run_fsca_assimilation(
            cfg,
            assimilation,
            ens_file_managers,
            particle_filter,
            reader,
            conversions_log_run,
            summa_output_dir,
            log,
        )

    log.info("Assimilation complete.")
    return AssimilationRun(
        run_dir=run_dir,
        log=log,
        swe_updates=swe_updates,
        unit_conversions=conversions_log_run,
        assimilation_mode=assimilation_mode,
    )


def assimilation_provenance(
    cfg: PysummadaConfig,
    config_path: str | Path,
    result: AssimilationRun,
    *,
    variant: str,
) -> dict[str, Any]:
    return {
        "config_path": str(Path(config_path).resolve()),
        "model_result_output_dir": str(cfg.model_result_output_dir),
        "variant": variant,
        "pf_scheme": cfg.pf_scheme,
        "assimilation_mode": result.assimilation_mode,
        "assimilation_dates": [item["date"] for item in result.swe_updates],
        "swe_updates": result.swe_updates,
        "unit_conversions": result.unit_conversions,
    }


def run_summa_da(config_path: str | Path) -> Path:
    cfg = load_pysummada_config(config_path)
    validate_da_config(cfg)
    if cfg.esp is None:
        raise ValueError("SUMMA DA requires an `esp` section.")

    result = run_summa_assimilation(
        cfg,
        config_path,
        log_filename="summa_da.log",
        logger_name=__name__,
        start_message="snowda SUMMA DA run starting",
    )
    result.log.info("Starting incremental ESP sequencing + traces.")
    esp_records = run_esp_traces(cfg, cfg.esp, result.run_dir)
    result.log.info("ESP complete. Generated %s traces.", len(esp_records))

    provenance = assimilation_provenance(cfg, config_path, result, variant="da")
    provenance["esp_trace_count"] = len(esp_records)
    provenance["esp_traces"] = [record.__dict__ for record in esp_records]
    write_provenance(result.run_dir, provenance, result.log)
    result.log.info("snowda SUMMA DA run complete")
    return result.run_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SUMMA data assimilation followed by ESP traces."
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    run_summa_da(args.config)


if __name__ == "__main__":
    main()
