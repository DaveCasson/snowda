"""Save small headless figures next to the laugh-test report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from laugh_tests._report import FIGURES

OUTPUT = Path(__file__).resolve().parent / "output"


def save_figure(fig: plt.Figure, name: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append(name)
    return path
