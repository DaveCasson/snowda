# src/snowda/config/loader.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


OUTPUT_LIKE_TOKENS = ("output", "cache", "log")


def _is_output_like(key: str) -> bool:
    k = key.lower()
    return any(tok in k for tok in OUTPUT_LIKE_TOKENS)


def _expand(p: str) -> str:
    # Expand environment variables and ~
    return str(Path(p).expanduser())


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML config. Path rules:
      - Relative paths are resolved relative to the config file's directory.
      - Under `paths:`, any key NOT containing 'output', 'cache', or 'log' must be an absolute path.
        (This enforces absolute input data sources while letting outputs be relative.)
    """
    config_path = Path(config_path).expanduser().resolve()
    base_dir = config_path.parent

    with config_path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    cfg.setdefault("paths", {})

    if not isinstance(cfg["paths"], dict):
        raise TypeError("Config field `paths` must be a mapping (YAML dictionary).")

    # Normalize + enforce path policy for cfg["paths"]
    normalized_paths: Dict[str, Any] = {}
    for key, value in cfg["paths"].items():
        if not isinstance(value, str):
            normalized_paths[key] = value
            continue

        raw = _expand(value)
        p = Path(raw)

        if _is_output_like(key):
            # Allow relative; resolve relative to config file directory
            if not p.is_absolute():
                p = (base_dir / p).resolve()
        else:
            # Input-like paths must be absolute
            if not p.is_absolute():
                raise ValueError(
                    f"`paths.{key}` must be an absolute path (got: {value}). "
                    f"Use a full path like /... or ${'{'}ENV_VAR{'}'}/..."
                )
            p = p.resolve()

        normalized_paths[key] = str(p)

    cfg["paths"] = normalized_paths

    # Useful metadata (won't break YAML structure if you re-dump cfg)
    cfg.setdefault("_meta", {})
    cfg["_meta"]["config_path"] = str(config_path)
    cfg["_meta"]["config_dir"] = str(base_dir)

    return cfg