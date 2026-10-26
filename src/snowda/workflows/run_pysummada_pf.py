from __future__ import annotations

from pathlib import Path

_REDIRECT = (
    "run_pysummada_pf has been replaced by these workflows:\n"
    "  python -m snowda.workflows.run_summa <yaml>             # deterministic SUMMA\n"
    "  python -m snowda.workflows.run_summa_historic <yaml>    # historic ensemble, no DA, no ESP\n"
    "  python -m snowda.workflows.run_summa_esp <yaml>         # ESP, no assimilation\n"
    "  python -m snowda.workflows.run_summa_historic_da <yaml> # historic DA, no ESP\n"
    "  python -m snowda.workflows.run_summa_da <yaml>          # DA then ESP"
)


def run_pysummada_pf(config_path: str | Path | None = None) -> Path:
    raise RuntimeError(_REDIRECT)


def main() -> None:
    raise SystemExit(_REDIRECT)


if __name__ == "__main__":
    main()
