# src/snowda/config/loader.py

from __future__ import annotations

from collections.abc import Mapping
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


def _normalize_paths(
    paths: Mapping[str, Any], base_dir: Path | None
) -> Dict[str, Any]:
    """Apply the `paths:` policy. Relative output paths need ``base_dir``."""
    normalized_paths: Dict[str, Any] = {}
    for key, value in paths.items():
        if not isinstance(value, str):
            normalized_paths[key] = value
            continue

        raw = _expand(value)
        p = Path(raw)

        if _is_output_like(key):
            # Allow relative; resolve relative to the config directory.
            if not p.is_absolute():
                if base_dir is None:
                    raise ValueError(
                        f"`paths.{key}` is relative ({value!r}) but the in-memory "
                        "config has no `_meta.config_dir`. Load the YAML with "
                        "snowda.config.loader.load_config() before editing it."
                    )
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

    return normalized_paths


def _config_dir(cfg: Mapping[str, Any]) -> Path | None:
    meta = cfg.get("_meta") or {}
    if isinstance(meta, Mapping) and meta.get("config_dir"):
        return Path(str(meta["config_dir"]))
    return None


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

    cfg["paths"] = _normalize_paths(cfg["paths"], base_dir)

    # Useful metadata (won't break YAML structure if you re-dump cfg)
    cfg.setdefault("_meta", {})
    cfg["_meta"]["config_path"] = str(config_path)
    cfg["_meta"]["config_dir"] = str(base_dir)

    return cfg


def as_config_dict(source: str | Path | Mapping[str, Any]) -> Dict[str, Any]:
    """Return a config dict from a YAML path or an already-loaded mapping.

    A mapping is shallow-copied. ``paths`` entries are normalized again so an
    edit made after ``load_config`` is still resolved. The caller's mapping is
    left unchanged. Relative output paths need ``_meta.config_dir``, which
    ``load_config`` records.
    """
    if isinstance(source, (str, Path)):
        return load_config(source)
    if not isinstance(source, Mapping):
        raise TypeError(
            "Config must be a path or a mapping, "
            f"got {type(source).__name__}."
        )

    cfg = dict(source)
    paths = cfg.get("paths", {})
    if not isinstance(paths, dict):
        raise TypeError("Config field `paths` must be a mapping (YAML dictionary).")
    cfg["paths"] = _normalize_paths(dict(paths), _config_dir(cfg))
    return cfg


def config_label(source: str | Path | Mapping[str, Any]) -> str:
    """Path recorded in logs and provenance. Not resolved again for a dict."""
    if isinstance(source, (str, Path)):
        return str(Path(source).expanduser().resolve())
    if not isinstance(source, Mapping):
        raise TypeError(
            "Config must be a path or a mapping, "
            f"got {type(source).__name__}."
        )
    meta = source.get("_meta") or {}
    if isinstance(meta, Mapping) and meta.get("config_path"):
        return str(meta["config_path"])
    return "<in-memory>"