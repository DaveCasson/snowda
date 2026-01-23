from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from snowda.utils.logging import get_logger

log = get_logger(__name__)


def log_subprocess_output(pipe: Iterable[bytes]) -> None:
    """Log streaming output from a subprocess pipe."""
    for raw in pipe:
        line = raw.decode("utf-8", errors="replace").rstrip()
        if line:
            log.info("SUMMA: %s", line)


def read_summa_workflow_control_setting(workflow_control_file: str | Path, setting: str) -> str:
    """Read a single setting from a SUMMAworkflow control file."""
    with Path(workflow_control_file).open("r", encoding="utf-8") as contents:
        for line in contents:
            if setting in line:
                break
        else:
            raise KeyError(f"Setting '{setting}' not found in {workflow_control_file}")

    substring = line.split("|", 1)[1]
    substring = substring.split("#", 1)[0]
    return substring.strip()


def read_summa_workflow_control_file(workflow_control_file: str | Path) -> Dict[str, str]:
    """Read all settings from a SUMMAworkflow control file."""
    control_options: Dict[str, str] = {}
    with Path(workflow_control_file).open("r", encoding="utf-8") as f:
        for line in f:
            if "#" in line:
                line, _comment = line.split("#", 1)
            if "|" not in line:
                continue
            option, value = line.split("|", 1)
            control_options[option.strip()] = value.strip()
    return control_options


def output_netcdf(xarray_ds, output_nc: str | Path) -> Path:
    """Write a compressed netCDF from an xarray dataset."""
    encoding = {key: {"zlib": True, "complevel": 3} for key in xarray_ds.keys()}
    output_path = Path(output_nc)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xarray_ds.to_netcdf(output_path, engine="netcdf4", encoding=encoding)
    return output_path


def calculate_ensemble_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic ensemble statistics across columns."""
    stats = df.copy()
    stats["mean"] = df.mean(axis=1)
    stats["max"] = df.max(axis=1)
    stats["min"] = df.min(axis=1)
    stats["p75"] = df.quantile(q=0.75, axis=1)
    stats["p25"] = df.quantile(q=0.25, axis=1)
    return stats
