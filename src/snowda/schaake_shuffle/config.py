from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _coerce_to_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(v) for v in value)
    return (str(value),)


@dataclass(frozen=True)
class SchaakeShuffleConfig:
    enabled: bool
    window_days: int
    seed: int | None
    archive_source: Path | None
    archive_variable: str
    target_variables: tuple[str, ...]
    apply_same_permutation_to: tuple[str, ...]

    @classmethod
    def from_config(
        cls, raw: dict[str, Any] | None, base_dir: Path, default_variable: str
    ) -> "SchaakeShuffleConfig":
        raw = raw or {}
        enabled = bool(raw.get("use_schaake_shuffle", raw.get("use", False)))
        window_days = int(raw.get("schaake_window_days", 15))
        seed = raw.get("schaake_seed")
        if seed is not None:
            seed = int(seed)
        archive_source_raw = raw.get("schaake_archive_source")
        archive_source = (
            None
            if archive_source_raw is None
            else Path(archive_source_raw).expanduser()
        )
        if archive_source and not archive_source.is_absolute():
            archive_source = (base_dir / archive_source).resolve()
        archive_variable = str(raw.get("schaake_archive_variable", default_variable))
        target_variables = _coerce_to_tuple(
            raw.get("schaake_target_variables", [default_variable])
        )
        if not target_variables:
            target_variables = (default_variable,)
        apply_same = _coerce_to_tuple(raw.get("schaake_apply_same_permutation_to", []))

        if enabled and archive_source is None:
            raise ValueError("Schaake shuffle enabled but no archive source provided.")

        return cls(
            enabled=enabled,
            window_days=window_days,
            seed=seed,
            archive_source=archive_source,
            archive_variable=archive_variable,
            target_variables=target_variables,
            apply_same_permutation_to=apply_same,
        )
