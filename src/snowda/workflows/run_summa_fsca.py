from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import xarray as xr

from snowda.adapters.pysumma import run_simulation
from snowda.config.loader import as_config_dict, config_label
from snowda.io.fsca_obs import read_fsca_observations
from snowda.io.summa_fsca import calc_summa_fsca, prepare_summa_fsca_input
from snowda.utils.logging import configure_logging, get_logger


def _now_utc_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _normalize_path(raw: str, base_dir: Path | None, *, allow_relative: bool) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        if not allow_relative:
            raise ValueError(f"Path must be absolute (got: {raw})")
        if base_dir is None:
            raise ValueError(
                f"Relative path {raw!r} needs `_meta.config_dir`. "
                "Load the YAML with snowda.config.loader.load_config() before editing it."
            )
        p = (base_dir / p).resolve()
    return p


def _load_summa_output_files(cfg: Dict[str, Any], base_dir: Path | None) -> List[Path]:
    files = cfg.get("summa_output_files")
    if files:
        return [
            _normalize_path(str(f), base_dir, allow_relative=False)
            for f in list(files)
        ]
    txt = cfg.get("summa_output_file_list_txt")
    if not txt:
        raise ValueError(
            "Config must define `summa_output_files` or `summa_output_file_list_txt`."
        )
    txt_path = _normalize_path(str(txt), base_dir, allow_relative=False)
    return [Path(line.strip()) for line in txt_path.read_text().splitlines() if line.strip()]


def _ensure_summa_outputs(
    cfg: Dict[str, Any],
    base_dir: Path | None,
    output_files: List[Path],
) -> List[Path]:
    missing = [path for path in output_files if not path.exists()]
    if not missing:
        return output_files

    if not cfg.get("summa_run_if_missing"):
        raise FileNotFoundError(
            f"Missing SUMMA outputs: {missing}. Set `summa_run_if_missing: true` "
            "to re-run SUMMA."
        )

    summa_exe = cfg.get("summa_exe")
    file_manager = cfg.get("file_manager")
    if not summa_exe or not file_manager:
        raise ValueError(
            "To re-run SUMMA, config must define `summa_exe` and `file_manager`."
        )

    sim_start_time = cfg.get("sim_start_time")
    sim_end_time = cfg.get("sim_end_time")
    sim_config = cfg.get("summa_sim_config") or {}
    write_configuration_name = cfg.get("summa_write_configuration_name")

    run_simulation(
        str(summa_exe),
        str(file_manager),
        sim_start_time=str(sim_start_time) if sim_start_time else None,
        sim_end_time=str(sim_end_time) if sim_end_time else None,
        sim_config=sim_config if sim_config else None,
        write_configuration_name=write_configuration_name,
        log_stdout=True,
    )

    # Re-check after the SUMMA run. If a list file is provided, reload it.
    if cfg.get("summa_output_file_list_txt"):
        output_files = _load_summa_output_files(cfg, base_dir)

    missing_after = [path for path in output_files if not path.exists()]
    if missing_after:
        raise FileNotFoundError(
            f"SUMMA run complete but outputs still missing: {missing_after}"
        )
    return output_files


def run_summa_fsca(config: str | Path | Mapping[str, Any]) -> Path:
    cfg: Dict[str, Any] = as_config_dict(config)
    label = config_label(config)
    meta = cfg.get("_meta") or {}
    config_dir = meta.get("config_dir") if isinstance(meta, Mapping) else None
    base_dir = Path(str(config_dir)) if config_dir else None

    output_dir = cfg.get("output_dir")
    if not output_dir:
        raise ValueError("Config must define `output_dir`.")
    output_dir = _normalize_path(str(output_dir), base_dir, allow_relative=True)

    run_dir = _ensure_dir(output_dir / _now_utc_tag())

    log_file_cfg = cfg.get("log_file")
    log_file = None
    if isinstance(log_file_cfg, str) and log_file_cfg:
        log_file = run_dir / Path(log_file_cfg).name

    configure_logging(level=logging.INFO, log_file=log_file)
    log = get_logger(__name__)
    log.info("snowda fSCA run starting")
    log.info("Config: %s", label)
    log.info("Run dir: %s", run_dir)

    config_out = run_dir / "config.resolved.yaml"
    try:
        import yaml

        with config_out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        log.info("Wrote config snapshot: %s", config_out)
    except Exception as e:
        log.warning("Failed to write config snapshot (%s): %s", config_out, e)

    summa_output_files = _load_summa_output_files(cfg, base_dir)
    summa_output_files = _ensure_summa_outputs(cfg, base_dir, summa_output_files)
    fsca_input_name = cfg.get("fsca_input_nc", "summa_fsca_input.nc")
    fsca_output_name = cfg.get("fsca_output_nc", "summa_fsca_output.nc")
    fsca_model = cfg.get("fsca_model", "ssnowd")
    fsca_cv = float(cfg.get("fsca_cv", 0.5))
    helbig_settings = cfg.get("helbig_settings")
    snow_depth_var = cfg.get("snow_depth_var", "scalarSnowDepth")
    if fsca_model == "helbig" and not helbig_settings:
        raise ValueError("helbig_settings are required when fsca_model is 'helbig'.")

    fsca_input_path = run_dir / str(fsca_input_name)
    fsca_output_path = run_dir / str(fsca_output_name)

    fsca_input_path, conversions = prepare_summa_fsca_input(
        summa_output_files, fsca_input_path
    )
    fsca_out = calc_summa_fsca(
        fsca_input_path,
        fsca_cv,
        fsca_output_path,
        fsca_model=str(fsca_model),
        helbig_settings=helbig_settings,
        snow_depth_var=str(snow_depth_var),
    )

    obs_nc = cfg.get("fsca_obs_nc")
    obs_var = cfg.get("fsca_obs_var", "fsca")
    if obs_nc:
        obs_df = read_fsca_observations(obs_nc, variable=str(obs_var))
        with xr.open_dataset(fsca_out) as modeled_ds:
            modeled_df = modeled_ds["fsca"].to_dataframe()
            modeled_df.index.names = ["hru", "time"]
        merged = modeled_df.join(obs_df, how="inner")
        merged["fsca_error"] = merged["fsca"] - merged["fsca_obs"]
        errors_ds = merged.to_xarray()
        error_out = run_dir / "fsca_obs_comparison.nc"
        errors_ds.to_netcdf(error_out)
        log.info("Wrote fSCA observation comparison: %s", error_out)

    prov = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": label,
        "run_dir": str(run_dir),
        "fsca_input": str(fsca_input_path),
        "fsca_output": str(fsca_output_path),
        "unit_conversions": conversions,
    }
    prov_out = run_dir / "provenance.json"
    prov_out.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    log.info("Wrote provenance: %s", prov_out)

    log.info("snowda fSCA run complete")
    return run_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run SUMMA fSCA workflow from a YAML config.")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    run_summa_fsca(args.config)


if __name__ == "__main__":
    main()
