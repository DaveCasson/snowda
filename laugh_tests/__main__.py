"""Run the laugh tests and print the expected-versus-computed report."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    # Importing snowda needs src/ on the path even if pytest's pythonpath
    # option is not picked up. The repo root makes `laugh_tests` importable.
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    args = [
        str(Path(__file__).resolve().parent),
        "-ra",
        f"--rootdir={repo}",
        *sys.argv[1:],
    ]
    return int(pytest.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
