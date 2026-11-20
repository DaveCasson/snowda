"""Run any snowda case from its ``run_mode``: ``python -m snowda.workflows.run_case <config>``.

``run_mode`` in the config picks the workflow (see ``snowda.config.run_modes``).
For the assimilating modes it also fixes the particle-filter scheme, so
``esp_sis`` and ``esp_sir`` configs can differ in that one line only. Every
workflow evaluates the run when it finishes, if the config has an
``evaluation`` section.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from snowda.config.loader import as_config_dict
from snowda.config.run_modes import (
    DETERMINISTIC,
    ESP_OPEN_LOOP,
    ESP_SIR,
    ESP_SIS,
    HISTORIC_ENSEMBLE,
    HISTORIC_SIR,
    HISTORIC_SIS,
    PF_SCHEME_BY_MODE,
    RUN_MODES,
    parse_run_mode,
)


def _workflow(run_mode: str) -> Callable[[Mapping[str, Any]], Path]:
    # Imported here so asking for one mode does not load every workflow.
    if run_mode == DETERMINISTIC:
        from snowda.workflows.run_summa import run_summa

        return run_summa
    if run_mode == HISTORIC_ENSEMBLE:
        from snowda.workflows.run_summa_historic import run_summa_historic

        return run_summa_historic
    if run_mode in (HISTORIC_SIS, HISTORIC_SIR):
        from snowda.workflows.run_summa_historic_da import run_summa_historic_da

        return run_summa_historic_da
    if run_mode == ESP_OPEN_LOOP:
        from snowda.workflows.run_summa_esp import run_summa_esp

        return run_summa_esp
    if run_mode in (ESP_SIS, ESP_SIR):
        from snowda.workflows.run_summa_da import run_summa_da

        return run_summa_da
    raise ValueError(f"Unsupported run_mode {run_mode!r}.")


def resolve_case_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Config dict with ``run_mode`` checked and ``pf_scheme`` set from it.

    ``pf_scheme`` may be left out of an assimilating config. When it is given
    it must agree with ``run_mode``.
    """
    cfg = as_config_dict(config)
    run_mode = parse_run_mode(cfg.get("run_mode"))
    if run_mode is None:
        raise ValueError(
            "Config must define `run_mode`. Use one of: " + ", ".join(RUN_MODES) + "."
        )
    cfg["run_mode"] = run_mode
    scheme = PF_SCHEME_BY_MODE.get(run_mode)
    if scheme is not None:
        configured = cfg.get("pf_scheme")
        if configured not in (None, "") and str(configured).strip().lower() != scheme:
            raise ValueError(
                f"run_mode {run_mode!r} uses pf_scheme {scheme!r}, but the config "
                f"sets pf_scheme {configured!r}. Remove `pf_scheme` or change `run_mode`."
            )
        cfg["pf_scheme"] = scheme
    return cfg


def run_case(config: str | Path | Mapping[str, Any]) -> Path:
    """Run the workflow named by the config's ``run_mode``. Returns the run directory."""
    cfg = resolve_case_config(config)
    return _workflow(cfg["run_mode"])(cfg)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a snowda case; `run_mode` in the YAML picks the workflow."
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    run_case(args.config)


if __name__ == "__main__":
    main()
