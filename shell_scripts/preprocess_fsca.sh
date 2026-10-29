#!/bin/bash
#SBATCH --account=ees250064
#SBATCH -J snowda_fsca
#SBATCH -o /home/x-dcasson/GitRepos/snowda/shell_scripts/logs/%x-%j.out
#SBATCH -e /home/x-dcasson/GitRepos/snowda/shell_scripts/logs/%x-%j.err
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=128G

# fSCA preprocessing (runs/examples/fsca_preprocess.yaml).
# Running this script submits it. The workload starts only inside the job.
#   ./preprocess_fsca.sh

set -euo pipefail

REPO=/home/x-dcasson/GitRepos/snowda
mkdir -p "$REPO/shell_scripts/logs"

# #SBATCH lines are comments unless sbatch reads them. Submit from here
# when this process is not already a SLURM job, then let the job run below.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submitting $(basename "$0")"
  exec sbatch "$0" "$@"
fi

echo "Job started: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unset}"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"

# --- Modules / Conda env ---

source /home/x-dcasson/snowda/bin/activate

# Sanity checks: confirm we are NOT using /apps/anvil/... site-packages for installs
which python
python -c "import sys; print('python:', sys.executable); print('prefix:', sys.prefix)"

cd /home/x-dcasson/GitRepos/snowda/src/
# --- Run ---
python - <<'PY'
from pathlib import Path
from snowda.io.fsca_preprocess import preprocess_fsca

config_path = Path("../runs/examples/fsca_preprocess.yaml").resolve()
out_paths = preprocess_fsca(config_path)
for p in out_paths:
    print("Preprocessed fSCA netCDF:", p)
PY

echo "Job finished: $(date)"