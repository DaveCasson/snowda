"""Score a finished run again: ``python -m snowda.workflows.evaluate_run <run_dir>``.

Every workflow runs this evaluation when it finishes. Use this entry point to
re-score an existing run, for example after changing the ``evaluation``
section, or after an automatic evaluation failed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from snowda.config.pysummada import load_pysummada_config
from snowda.config.run_modes import RUN_MODES
from snowda.evaluation.run import EVALUATION_DIRNAME, evaluate_run
from snowda.utils.logging import configure_logging


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a finished snowda run directory."
    )
    parser.add_argument("run_dir", type=str, help="Run directory to score")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config with the `evaluation` section "
        "(default: the config named in the run's provenance.json)",
    )
    parser.add_argument(
        "--run-mode",
        choices=RUN_MODES,
        default=None,
        help="Override the run mode recorded in provenance.json",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    log_dir = run_dir / EVALUATION_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(level=logging.INFO, log_file=log_dir / "evaluate_run.log")
    cfg = load_pysummada_config(args.config) if args.config else None
    evaluate_run(run_dir, cfg, run_mode=args.run_mode)


if __name__ == "__main__":
    main()
