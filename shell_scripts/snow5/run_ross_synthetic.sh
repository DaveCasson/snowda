#!/usr/bin/env bash
# Ross snow5 synthetic experiment: setup, four historic modes, figures.
# Usage: shell_scripts/snow5/run_ross_synthetic.sh [python]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${1:-python}"
CFG="$REPO/runs/snow5/ross_synthetic"
LOG="$REPO/outputs/snow5/ross_synthetic"
mkdir -p "$LOG"
cd "$REPO"

"$PY" -m snowda.workflows.snow5_synthetic_setup "$CFG/ross_snow5_synthetic_setup.yaml"

"$PY" -m snowda.workflows.run_snow5          "$CFG/ross_snow5_historic_mean.yaml"     > "$LOG/mean.out" 2>&1 &
"$PY" -m snowda.workflows.run_snow5_ensemble "$CFG/ross_snow5_historic_ensemble.yaml" > "$LOG/ens.out"  2>&1 &
"$PY" -m snowda.workflows.run_snow5_da       "$CFG/ross_snow5_sir.yaml"               > "$LOG/sir.out"  2>&1 &
"$PY" -m snowda.workflows.run_snow5_da       "$CFG/ross_snow5_sis.yaml"               > "$LOG/sis.out"  2>&1 &
wait

"$PY" scripts/snow5/plot_ross_synthetic.py
