from __future__ import annotations

from pathlib import Path

from snowda.config.pysummada import load_pysummada_config, validate_esp_config
from snowda.workflows.esp import run_esp_traces
from snowda.workflows.summa_common import setup_run_directory, write_provenance


def run_summa_esp(config_path: str | Path) -> Path:
    cfg = load_pysummada_config(config_path)
    validate_esp_config(cfg)
    if cfg.esp is None:
        raise ValueError("ESP workflow requires an `esp` section.")

    run_dir, log = setup_run_directory(
        cfg,
        config_path,
        log_filename="summa_esp.log",
        logger_name=__name__,
        start_message="snowda SUMMA ESP run starting",
    )

    log.info("Starting incremental ESP sequencing + traces.")
    esp_records = run_esp_traces(cfg, cfg.esp, run_dir)
    log.info("ESP complete. Generated %s traces.", len(esp_records))

    write_provenance(
        run_dir,
        {
            "config_path": str(Path(config_path).resolve()),
            "model_result_output_dir": str(cfg.model_result_output_dir),
            "variant": "esp",
            "esp_trace_count": len(esp_records),
            "esp_traces": [record.__dict__ for record in esp_records],
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
