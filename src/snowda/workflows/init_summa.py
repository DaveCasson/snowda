from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from snowda.config.loader import load_config
from snowda.utils.logging import configure_logging, get_logger


def _now_utc_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _require_keys(section: Dict[str, Any], keys: list[str], section_name: str) -> None:
    missing = [key for key in keys if key not in section or section[key] in ("", None)]
    if missing:
        raise ValueError(f"Config is missing required {section_name} fields: {missing}")


def _normalize_path(raw: str, base_dir: Path, *, allow_relative: bool) -> str:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        if not allow_relative:
            raise ValueError(f"Path must be absolute (got: {raw})")
        p = (base_dir / p).resolve()
    return str(p)


def _normalize_flat_config(cfg: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    flat_keys = {
        "case_name",
        "watershed",
        "description",
        "output_dir",
        "log_file",
        "summa_forcing_dir",
        "summa_settings_dir",
        "summa_exe",
        "file_manager",
        "output_file_name",
        "fsca_output_name",
        "initialize_only",
    }
    if not any(key in cfg for key in flat_keys):
        return cfg

    case = dict(cfg.get("case", {}))
    if "case_name" in cfg and "name" not in case:
        case["name"] = cfg.get("case_name")
    if "watershed" in cfg and "watershed" not in case:
        case["watershed"] = cfg.get("watershed")
    if "description" in cfg and "description" not in case:
        case["description"] = cfg.get("description")

    paths = dict(cfg.get("paths", {}))
    if "output_dir" in cfg and "output_dir" not in paths:
        paths["output_dir"] = _normalize_path(
            str(cfg.get("output_dir")), base_dir, allow_relative=True
        )
    if "log_file" in cfg and "log_file" not in paths:
        paths["log_file"] = _normalize_path(
            str(cfg.get("log_file")), base_dir, allow_relative=True
        )
    if "summa_forcing_dir" in cfg and "summa_forcing_dir" not in paths:
        paths["summa_forcing_dir"] = _normalize_path(
            str(cfg.get("summa_forcing_dir")), base_dir, allow_relative=False
        )
    if "summa_settings_dir" in cfg and "summa_settings_dir" not in paths:
        paths["summa_settings_dir"] = _normalize_path(
            str(cfg.get("summa_settings_dir")), base_dir, allow_relative=False
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
    if "output_file_name" in cfg and "output_file_name" not in summa:
        summa["output_file_name"] = cfg.get("output_file_name")
    if "fsca_output_name" in cfg and "fsca_output_name" not in summa:
        summa["fsca_output_name"] = cfg.get("fsca_output_name")

    run_options = dict(cfg.get("run_options", {}))
    if "initialize_only" in cfg and "initialize_only" not in run_options:
        run_options["initialize_only"] = cfg.get("initialize_only")

    cfg["case"] = case
    cfg["paths"] = paths
    cfg["summa"] = summa
    cfg["run_options"] = run_options
    return cfg


def init_summa_run(config_path: str | Path) -> Path:
    """
    Initialize (but do not run) a SUMMA model case.

    Creates a timestamped run directory, writes a resolved config snapshot,
    and generates a summa_run_plan.json describing the inputs for a future run.
    """
    cfg: Dict[str, Any] = load_config(config_path)
    base_dir = Path(cfg.get("_meta", {}).get("config_dir", Path(config_path).resolve().parent))
    cfg = _normalize_flat_config(cfg, base_dir)

    paths = cfg.get("paths", {})
    output_dir = paths.get("output_dir")
    if not output_dir:
        raise ValueError("Config must define `paths.output_dir`.")

    run_dir = _ensure_dir(Path(output_dir) / _now_utc_tag())

    log_file_cfg = paths.get("log_file")
    log_file = None
    if isinstance(log_file_cfg, str) and log_file_cfg:
        log_file = run_dir / Path(log_file_cfg).name

    configure_logging(level=logging.INFO, log_file=log_file)
    log = get_logger(__name__)

    log.info("snowda SUMMA init starting")
    log.info("Config: %s", cfg.get("_meta", {}).get("config_path", str(config_path)))
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
    _require_keys(summa_cfg, ["summa_exe", "file_manager"], "summa")

    plan = {
        "initialized_utc": datetime.utcnow().isoformat() + "Z",
        "run_dir": str(run_dir),
        "case": cfg.get("case", {}),
        "paths": paths,
        "summa": {
            "summa_exe": summa_cfg.get("summa_exe"),
            "file_manager": summa_cfg.get("file_manager"),
            "output_file_name": summa_cfg.get("output_file_name"),
            "fsca_output_name": summa_cfg.get("fsca_output_name"),
        },
        "run_options": cfg.get("run_options", {}),
        "notes": "Initialized only; no model execution performed.",
    }

    plan_out = run_dir / "summa_run_plan.json"
    plan_out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    log.info("Wrote SUMMA run plan: %s", plan_out)

    log.info("snowda SUMMA init complete")
    return run_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Initialize a SUMMA model run from a YAML config (no execution)."
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    init_summa_run(args.config)


if __name__ == "__main__":
    main()
