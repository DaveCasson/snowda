"""Named run modes shared by the workflows, the dispatcher and evaluation.

A run mode says what a run did, so evaluation knows what to score:

==================  ==========================================  ============
``run_mode``        workflow                                    scored as
==================  ==========================================  ============
deterministic       ``run_summa``                               historic
historic_ensemble   ``run_summa_historic``                      historic
historic_sis        ``run_summa_historic_da``, ``pf_scheme: sis``  historic
historic_sir        ``run_summa_historic_da``, ``pf_scheme: sir``  historic
esp_open_loop       ``run_summa_esp`` (no assimilation)         both
esp_sis             ``run_summa_da``, ``pf_scheme: sis``        both
esp_sir             ``run_summa_da``, ``pf_scheme: sir``        both
==================  ==========================================  ============

``run_mode`` in a config is optional when a workflow is called directly. When
it is set, the workflow refuses a config written for another mode.
"""

from __future__ import annotations

from typing import Any, Optional

DETERMINISTIC = "deterministic"
HISTORIC_ENSEMBLE = "historic_ensemble"
HISTORIC_SIS = "historic_sis"
HISTORIC_SIR = "historic_sir"
ESP_OPEN_LOOP = "esp_open_loop"
ESP_SIS = "esp_sis"
ESP_SIR = "esp_sir"

RUN_MODES = (
    DETERMINISTIC,
    HISTORIC_ENSEMBLE,
    HISTORIC_SIS,
    HISTORIC_SIR,
    ESP_OPEN_LOOP,
    ESP_SIS,
    ESP_SIR,
)
FORECAST_MODES = frozenset({ESP_OPEN_LOOP, ESP_SIS, ESP_SIR})
ASSIMILATION_MODES = frozenset({HISTORIC_SIS, HISTORIC_SIR, ESP_SIS, ESP_SIR})

# Particle-filter scheme each assimilating mode runs with.
PF_SCHEME_BY_MODE = {
    HISTORIC_SIS: "sis",
    HISTORIC_SIR: "sir",
    ESP_SIS: "sis",
    ESP_SIR: "sir",
}

# ``variant`` written to provenance.json by runs made before ``run_mode`` existed.
_LEGACY_VARIANTS = {
    "deterministic": DETERMINISTIC,
    "historic": HISTORIC_ENSEMBLE,
    "esp": ESP_OPEN_LOOP,
}


def parse_run_mode(raw: Any) -> Optional[str]:
    """Validated ``run_mode`` from a config, or ``None`` when it is not set."""
    if raw in (None, ""):
        return None
    mode = str(raw).strip().lower()
    if mode not in RUN_MODES:
        raise ValueError(
            f"Unsupported run_mode {raw!r}. Use one of: {', '.join(RUN_MODES)}."
        )
    return mode


def historic_da_mode(pf_scheme: str) -> str:
    return f"historic_{pf_scheme}"


def esp_da_mode(pf_scheme: str) -> str:
    return f"esp_{pf_scheme}"


def check_run_mode(configured: Optional[str], actual: str) -> str:
    """Return ``actual``, refusing a config whose ``run_mode`` names another mode."""
    if configured is not None and configured != actual:
        raise ValueError(
            f"Config sets run_mode {configured!r}, but this workflow and its "
            f"settings run {actual!r}. Run it with `snowda.workflows.run_case`, "
            "or correct `run_mode` / `pf_scheme`."
        )
    return actual


def run_mode_from_provenance(provenance: dict[str, Any]) -> Optional[str]:
    """Run mode of a finished run, including runs older than ``run_mode``."""
    mode = provenance.get("run_mode")
    if mode in RUN_MODES:
        return str(mode)
    variant = provenance.get("variant")
    if variant in _LEGACY_VARIANTS:
        return _LEGACY_VARIANTS[variant]
    scheme = provenance.get("pf_scheme")
    if scheme in {"sis", "sir"}:
        if variant == "historic_da":
            return historic_da_mode(scheme)
        if variant == "da":
            return esp_da_mode(scheme)
    return None
