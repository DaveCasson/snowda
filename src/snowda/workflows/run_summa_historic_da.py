from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from snowda.config.loader import config_label
from snowda.config.pysummada import load_pysummada_config, validate_historic_da_config
from snowda.workflows.run_summa_da import (
    assimilation_provenance,
    run_summa_assimilation,
)
from snowda.workflows.summa_common import write_provenance


def run_summa_historic_da(config: str | Path | Mapping[str, Any]) -> Path:
    """Carry the historic ensemble through assimilation without ESP traces."""
    cfg = load_pysummada_config(config)
    label = config_label(config)
    validate_historic_da_config(cfg)
    result = run_summa_assimilation(
        cfg,
        label,
        log_filename="summa_historic_da.log",
        logger_name=__name__,
        start_message="snowda SUMMA historic DA run starting",
    )
    write_provenance(
        result.run_dir,
        assimilation_provenance(cfg, label, result, variant="historic_da"),
        result.log,
    )
    result.log.info("snowda SUMMA historic DA run complete")
    return result.run_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run historic SUMMA data assimilation without ESP traces."
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    run_summa_historic_da(args.config)


if __name__ == "__main__":
    main()
