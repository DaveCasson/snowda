from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from snowda.config.loader import load_config
from snowda.utils.logging import configure_logging, get_logger


def _git_commit_hash(repo_root: Path) -> Optional[str]:
    """Return current git commit hash if available, otherwise None."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _now_utc_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_case(config_path: str | Path) -> Path:
    """
    Run a single snowda case described by a YAML config.

    Returns:
        Path to the created run directory.
    """
    cfg: Dict[str, Any] = load_config(config_path)

    paths = cfg.get("paths", {})
    output_dir = paths.get("output_dir")
    if not output_dir:
        raise ValueError("Config must define `paths.output_dir`.")

    # Create a unique run folder inside output_dir (keeps repeated runs clean)
    run_dir = _ensure_dir(Path(output_dir) / _now_utc_tag())

    # Configure logging once. If config provided a log file, put it inside run_dir.
    log_file_cfg = paths.get("log_file")
    log_file = None
    if isinstance(log_file_cfg, str) and log_file_cfg:
        # If user provided "logs/run.log", write it relative to run_dir
        log_file = run_dir / Path(log_file_cfg).name

    configure_logging(level=logging.INFO, log_file=log_file)
    log = get_logger(__name__)

    log.info("snowda run starting")
    log.info("Config: %s", cfg.get("_meta", {}).get("config_path", str(config_path)))
    log.info("Run dir: %s", run_dir)

    # Save resolved config snapshot used for this run
    config_out = run_dir / "config.resolved.yaml"
    try:
        import yaml  # local import to keep dependencies explicit

        with config_out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        log.info("Wrote config snapshot: %s", config_out)
    except Exception as e:
        log.warning("Failed to write config snapshot (%s): %s", config_out, e)

    # Save minimal provenance
    repo_root = Path(__file__).resolve().parents[3]  # snowda/src/snowda/workflows/run_case.py -> repo root
    prov = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit_hash(repo_root),
        "config_path": cfg.get("_meta", {}).get("config_path", str(config_path)),
        "run_dir": str(run_dir),
    }
    prov_out = run_dir / "provenance.json"
    prov_out.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    log.info("Wrote provenance: %s", prov_out)

    # ---- Pipeline stubs (fill these in later) ----
    log.info("STEP: load forcing/inputs (TODO)")
    log.info("STEP: load observations (TODO)")
    log.info("STEP: run model simulation (TODO)")
    log.info("STEP: run data assimilation (TODO)")
    log.info("STEP: write outputs (TODO)")
    log.info("STEP: generate figures (TODO)")
    # ---------------------------------------------

    log.info("snowda run complete")
    return run_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a snowda case from a YAML config.")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    run_case(args.config)


if __name__ == "__main__":
    main()