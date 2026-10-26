from __future__ import annotations

from pathlib import Path

from snowda.adapters.pysumma import run_ensemble
from snowda.config.pysummada import (
    PysummadaConfig,
    load_pysummada_config,
    validate_historic_config,
)
from snowda.workflows.summa_common import (
    apply_run_time,
    prepare_ensemble,
    redirect_summa_output,
    route_ensemble_segment,
    setup_run_directory,
    summa_run_output_dir,
    write_provenance,
)


def _freq_restart(cfg: PysummadaConfig) -> str:
    return cfg.summa_args.get("-r") or cfg.summa_args.get("freq_restart") or "e"


def run_summa_historic(config_path: str | Path) -> Path:
    """Run the historic ensemble once, without assimilation or ESP."""
    cfg = load_pysummada_config(config_path)
    validate_historic_config(cfg)
    run_dir, log = setup_run_directory(
        cfg,
        config_path,
        log_filename="summa_historic.log",
        logger_name=__name__,
        start_message="snowda SUMMA historic run starting",
    )

    file_writer, assimilation = prepare_ensemble(cfg)
    log.info("Starting historic ensemble %s to %s", cfg.start_time, cfg.end_time)
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
        freq_restart=_freq_restart(cfg),
        log_stdout=True,
    )

    routed = False
    if cfg.mizuroute is not None:
        log.info("Historic SUMMA run complete; routing each ensemble member.")
        route_ensemble_segment(
            cfg.mizuroute,
            run_dir,
            cfg.ens_label_array,
            summa_output_dir,
            cfg.start_time,
            cfg.end_time,
            {},
            require_restart=False,
            num_workers=cfg.num_workers,
        )
        routed = True

    write_provenance(
        run_dir,
        {
            "config_path": str(Path(config_path).resolve()),
            "model_result_output_dir": str(cfg.model_result_output_dir),
            "variant": "historic",
            "ensemble_forcing_mode": cfg.ensemble_forcing_mode,
            "summa_output_dir": str(summa_output_dir),
            "routed": routed,
            "mizuroute_dir": str(run_dir / "mizuroute") if routed else None,
        },
        log,
    )
    log.info("snowda SUMMA historic run complete")
    return run_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a historic SUMMA ensemble without assimilation or ESP."
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    run_summa_historic(args.config)


if __name__ == "__main__":
    main()
