from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from snowda.adapters.pysumma import run_ensemble
from snowda.config.loader import config_label
from snowda.config.pysummada import (
    PysummadaConfig,
    load_pysummada_config,
    validate_esp_config,
)
from snowda.core.assimilation import assimilation_segments, restart_filename
from snowda.core.esp_sampling import (
    ESP_SAMPLING_DESCRIPTION,
    ParticleState,
    write_post_da_catalog,
)
from snowda.workflows.esp import (
    balanced_trigger_dates,
    resolve_esp_trigger_dates,
    run_balanced_esp_issue,
)
from snowda.workflows.summa_common import (
    apply_run_time,
    apply_segment_initial_conditions,
    prepare_ensemble,
    redirect_summa_output,
    route_ensemble_segment,
    setup_run_directory,
    summa_run_output_dir,
    summa_run_state_dir,
    write_provenance,
)


def open_loop_segment_bounds(
    start_time: datetime,
    end_time: datetime,
    explicit_dates: Sequence[datetime],
    frequency_days: int | None,
    months: Sequence[int] | None = None,
) -> list[tuple[datetime, datetime]]:
    """Return ``(segment_start, segment_end)`` for the open-loop historic ensemble.

    Dates outside ``(start_time, end_time]`` are dropped. The last segment
    ends at ``end_time`` when that is after the last in-window date. With no
    in-window date, the run is one segment from start to end.
    """
    candidates = resolve_esp_trigger_dates(
        start_time,
        end_time,
        explicit_dates,
        frequency_days,
        months,
    )
    dates = [date for date in candidates if start_time < date <= end_time]
    if not dates:
        return [(start_time, end_time)]
    return [
        (left, right)
        for left, right, _assimilate in assimilation_segments(
            start_time, end_time, dates
        )
    ]


def _freq_restart(cfg: PysummadaConfig) -> str:
    return cfg.summa_args.get("-r") or cfg.summa_args.get("freq_restart") or "d"


def _progress(cfg: PysummadaConfig) -> str | None:
    return cfg.summa_args.get("-p") or cfg.summa_args.get("progress") or None


def _issue_dates(cfg: PysummadaConfig) -> set[datetime]:
    if cfg.esp is None:
        return set()
    return {
        date
        for date in resolve_esp_trigger_dates(
            cfg.start_time,
            cfg.end_time,
            list(cfg.esp.dates),
            cfg.esp.frequency_days,
            cfg.esp.months,
        )
        if cfg.start_time < date <= cfg.end_time
    }


def _record_historic_particles(
    run_dir: Path,
    when: datetime,
    labels: Sequence[str],
    file_managers: Sequence[str],
    routing_restarts: dict[str, Path | None],
    *,
    require_routing: bool,
) -> None:
    """Save each historic member as one SUMMA restart bound to its mizuRoute restart."""
    state_dir = run_dir / "states"
    particles: list[ParticleState] = []
    for label, file_manager in zip(labels, file_managers):
        summa_restart = state_dir / restart_filename(label, when)
        route = routing_restarts.get(label)
        if require_routing and (route is None or not route.is_file()):
            raise FileNotFoundError(
                f"Historic member {label} has no mizuRoute restart at {when}. "
                "SUMMA and mizuRoute restarts are stored as one particle."
            )
        particles.append(
            ParticleState(
                label=label,
                donor_label=label,
                summa_restart=str(summa_restart),
                mizuroute_restart=str(route) if route is not None else None,
                weight=1.0,
                file_manager=str(file_manager),
                resampled=False,
            )
        )
    write_post_da_catalog(run_dir, when, particles)


def run_open_loop_historic(
    cfg: PysummadaConfig,
    run_dir: Path,
    log: logging.Logger,
    on_issue: Callable[[datetime], None] | None = None,
) -> Path:
    """Write historic ensemble segments to ``<run_dir>/summa`` and ``<run_dir>/states``.

    Segment ends match a DA run that uses the same ESP dates. Members continue
    from their own restarts. Each in-window issue date writes a particle catalog
    binding that member's SUMMA restart to its mizuRoute restart, then calls
    ``on_issue`` before the next segment starts.
    """
    if cfg.esp is None:
        raise ValueError("ESP workflow requires an `esp` section.")
    file_writer, assimilation = prepare_ensemble(cfg, run_dir)
    ens_file_managers = file_writer.ens_file_managers or []
    summa_output_dir = redirect_summa_output(
        ens_file_managers,
        assimilation.ens_config,
        summa_run_output_dir(run_dir),
    )
    log.info("SUMMA outputPath: %s", summa_output_dir)
    segments = open_loop_segment_bounds(
        cfg.start_time,
        cfg.end_time,
        list(cfg.esp.dates),
        cfg.esp.frequency_days,
        cfg.esp.months,
    )
    issue_dates = _issue_dates(cfg)
    labels = cfg.ens_label_array
    mizu = cfg.mizuroute or cfg.esp.mizuroute
    routing_restarts: dict[str, Path | None] = {}
    for index, (segment_start, segment_end) in enumerate(segments):
        log.info("Starting open-loop ensemble %s to %s", segment_start, segment_end)
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
            keep_existing=False,
        )
        run_ensemble(
            str(cfg.summa_exe),
            current_run_config,
            num_workers=cfg.num_workers,
            run_suffix=segment_end.strftime("%Y_%m_%d"),
            freq_restart=_freq_restart(cfg),
            progress=_progress(cfg),
            log_stdout=True,
        )
        if mizu is not None:
            route_ensemble_segment(
                mizu,
                run_dir,
                labels,
                summa_output_dir,
                segment_start,
                segment_end,
                routing_restarts,
                require_restart=index < len(segments) - 1 or segment_end in issue_dates,
                num_workers=cfg.num_workers,
            )
        if segment_end in issue_dates:
            _record_historic_particles(
                run_dir,
                segment_end,
                labels,
                ens_file_managers,
                routing_restarts,
                require_routing=mizu is not None,
            )
            if on_issue is not None:
                on_issue(segment_end)
    return summa_output_dir


def run_summa_esp(config: str | Path | Mapping[str, Any]) -> Path:
    cfg = load_pysummada_config(config)
    label = config_label(config)
    validate_esp_config(cfg)
    if cfg.esp is None:
        raise ValueError("ESP workflow requires an `esp` section.")

    run_dir, log = setup_run_directory(
        cfg,
        label,
        log_filename="summa_esp.log",
        logger_name=__name__,
        start_message="snowda SUMMA ESP run starting",
    )

    triggers = balanced_trigger_dates(cfg, cfg.esp)
    issued: set[str] = set()
    esp_records = []

    def _issue(when: datetime) -> None:
        key = when.strftime("%Y%m%d%H")
        if key in issued:
            return
        esp_records.extend(run_balanced_esp_issue(cfg, cfg.esp, run_dir, when))
        issued.add(key)

    summa_output_dir = run_open_loop_historic(cfg, run_dir, log, on_issue=_issue)
    for trigger in triggers:
        key = trigger.strftime("%Y%m%d%H")
        if key in issued:
            continue
        esp_records.extend(run_balanced_esp_issue(cfg, cfg.esp, run_dir, trigger))
        issued.add(key)
    log.info("ESP complete. Generated %s traces.", len(esp_records))

    write_provenance(
        run_dir,
        {
            "config_path": label,
            "model_result_output_dir": str(cfg.model_result_output_dir),
            "variant": "esp",
            "summa_output_dir": str(summa_output_dir),
            "state_dir": str(summa_run_state_dir(run_dir)),
            "esp_trace_count": len(esp_records),
            "esp_traces": [record.__dict__ for record in esp_records],
            "esp_sampling": ESP_SAMPLING_DESCRIPTION,
            "esp_member_map": str(run_dir / "esp" / "member_map.json"),
            "esp_n_members": cfg.esp.n_members,
            "esp_seed": cfg.esp.seed,
        },
        log,
    )
    log.info("snowda SUMMA ESP run complete")
    return run_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SUMMA Ensemble Streamflow Prediction (no assimilation)."
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    run_summa_esp(args.config)


if __name__ == "__main__":
    main()
