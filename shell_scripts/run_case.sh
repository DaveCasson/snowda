#!/bin/bash
#SBATCH --account=ees250064
#SBATCH -J snowda
#SBATCH -o /home/x-dcasson/GitRepos/snowda/shell_scripts/logs/%x-%j.out
#SBATCH -e /home/x-dcasson/GitRepos/snowda/shell_scripts/logs/%x-%j.err
#SBATCH -N 1
#SBATCH --time=1-00:00:00

# Run any snowda case on Anvil. `run_mode` in the YAML picks the workflow, so
# this one script replaces the per-workflow ones for configs that set it.
#
# Running this script submits one job per config. The model starts only
# inside the job.
#   ./run_case.sh ../runs/ross_comparison_anvil/esp_sis.yaml
#   ./run_case.sh ../runs/ross_comparison_anvil/*.yaml      # all six, in parallel
#   ./run_case.sh --dry-run ../runs/ross_comparison_anvil/*.yaml
#
# Each job asks for as many CPUs as `num_workers` in its YAML (at least 4),
# so keep `num_workers` equal to `num_ens`. Anvil's shared partition gives
# about 1.9 GB of memory per CPU.
#
# When a run finishes, its evaluation/ folder, provenance.json and
# config.resolved.json are copied to $SNOWDA_RESULTS/<case_name>/<run folder>.
# Model output stays in the config's output_dir (scratch, which is purged).

set -euo pipefail

REPO="${SNOWDA_REPO:-/home/x-dcasson/GitRepos/snowda}"
VENV="${SNOWDA_VENV:-/home/x-dcasson/snowda/bin/activate}"
RESULTS="${SNOWDA_RESULTS:-$REPO/outputs/ross_comparison}"
MIN_CPUS=4

yaml_value() {
  # Top-level scalar from a YAML file: yaml_value <file> <key>
  sed -n "s/^$2:[[:space:]]*\([^#[:space:]]*\).*/\1/p" "$1" | head -n 1
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  DRY_RUN=0
  if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
  fi
  if [[ $# -eq 0 ]]; then
    echo "Usage: $(basename "$0") [--dry-run] <config.yaml> [<config.yaml> ...]" >&2
    exit 1
  fi
  SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  for CONFIG in "$@"; do
    if [[ ! -f "$CONFIG" ]]; then
      echo "Config not found: $CONFIG" >&2
      exit 1
    fi
    CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
    MODE="$(yaml_value "$CONFIG" run_mode)"
    if [[ -z "$MODE" ]]; then
      echo "No run_mode in $CONFIG; run_case needs one." >&2
      exit 1
    fi
    CPUS="$(yaml_value "$CONFIG" num_workers)"
    CPUS="${CPUS:-$MIN_CPUS}"
    if (( CPUS < MIN_CPUS )); then
      CPUS=$MIN_CPUS
    fi
    CMD=(sbatch -J "snowda_$MODE" --cpus-per-task="$CPUS" "$SCRIPT" "$CONFIG")
    if (( DRY_RUN )); then
      echo "${CMD[*]}"
    else
      echo "Submitting $MODE ($CPUS CPUs): $CONFIG"
      mkdir -p "$REPO/shell_scripts/logs"
      "${CMD[@]}"
    fi
  done
  exit 0
fi

CONFIG="$1"
echo "Job started: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unset}"
echo "Config: $CONFIG"
echo "Commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"

source "$VENV"
which python

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

cd "$REPO/src"
python - "$CONFIG" "$RESULTS" <<'PY'
import shutil
import sys
from pathlib import Path

from snowda.config.loader import load_config
from snowda.workflows.run_case import run_case

config, results_root = sys.argv[1], Path(sys.argv[2])
run_dir = Path(run_case(config))
print(f"Run dir: {run_dir}")

# Keep the small files the comparison notebooks read, off scratch.
case_name = str(load_config(config).get("case_name", "run"))
keep = results_root / case_name / run_dir.name
keep.mkdir(parents=True, exist_ok=True)
for name in ("provenance.json", "config.resolved.json"):
    if (run_dir / name).is_file():
        shutil.copy2(run_dir / name, keep / name)
if (run_dir / "evaluation").is_dir():
    shutil.copytree(run_dir / "evaluation", keep / "evaluation", dirs_exist_ok=True)
    print(f"Copied evaluation to {keep}")
else:
    print("No evaluation folder was written; see the run log.")
if (run_dir / "evaluation" / "FAILED.txt").is_file():
    sys.exit("Model run finished but evaluation failed; see evaluation/FAILED.txt.")
PY

echo "Job finished: $(date)"
