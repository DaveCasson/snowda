"""Run-directory naming shared by all workflows.

Every workflow writes into ``<output_dir>/<run folder>``. The run folder is set
by the optional ``run_name`` config key:

- omitted, empty, ``null`` or ``timestamp``: a UTC timestamp such as
  ``20261119T153000Z`` (a fresh folder per run).
- any other string: that fixed folder name. Re-running reuses the folder and
  overwrites files with the same names.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from snowda.utils.logging import get_logger

log = get_logger(__name__)

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
TIMESTAMP_RUN_NAME = "timestamp"


def now_utc_tag() -> str:
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)


def utc_timestamp() -> str:
    """Current UTC time for provenance records, such as ``2026-11-19T15:30:00.123456Z``."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def git_commit_hash(repo_root: Path) -> Optional[str]:
    """Current git commit hash, or ``None`` outside a repository."""
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


def normalize_run_name(raw: Any) -> Optional[str]:
    """Return a fixed run folder name, or ``None`` to use a timestamp."""
    if raw is None:
        return None
    name = str(raw).strip()
    if not name or name.lower() in {TIMESTAMP_RUN_NAME, "null", "none"}:
        return None
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(
            f"`run_name` must be a single folder name, not a path (got {raw!r})."
        )
    return name


def run_name_from_config(cfg: Mapping[str, Any]) -> Optional[str]:
    """Read ``run_name`` from the top level or ``paths:`` of a config dict."""
    raw = cfg.get("run_name")
    if raw is None:
        paths = cfg.get("paths")
        if isinstance(paths, Mapping):
            raw = paths.get("run_name")
    return normalize_run_name(raw)


def make_run_dir(output_dir: str | Path, run_name: Any = None) -> Path:
    """Create and return ``<output_dir>/<run_name or UTC timestamp>``."""
    name = normalize_run_name(run_name)
    run_dir = Path(output_dir) / (name or now_utc_tag())
    if name and run_dir.exists():
        log.warning("Run dir %s already exists; files will be overwritten.", run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
