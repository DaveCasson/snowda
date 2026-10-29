#!/bin/bash
#SBATCH --account=ees250064
#SBATCH -J snowda_summa
#SBATCH -o /home/x-dcasson/GitRepos/snowda/shell_scripts/logs/%x-%j.out
#SBATCH -e /home/x-dcasson/GitRepos/snowda/shell_scripts/logs/%x-%j.err
#SBATCH -N 1
#SBATCH --cpus-per-task=50
#SBATCH --time=0-12:00:00
#SBATCH --mem=4G

# One deterministic SUMMA run (notebooks/run_summa_historic.ipynb).
# Anvil shared is the default partition: 4-day limit, MaxMemPerCPU=1896M.
# Memory above cpus * 1896M is scheduled by raising the CPU count.
#
# Running this script submits it. The workload starts only inside the job.
#   ./run_summa.sh
#   ./run_summa.sh /path/to/other_summa.yaml
#
# Example YAMLs still use laptop paths.

set -euo pipefail

REPO=/home/x-dcasson/GitRepos/snowda
CONFIG="${1:-$REPO/runs/examples/ross_summa_historic_ensemble.yaml}"
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi
CONFIG="$(realpath "$CONFIG")"

mkdir -p "$REPO/shell_scripts/logs"

# #SBATCH lines are comments unless sbatch reads them. Submit from here
# when this process is not already a SLURM job, then let the job run below.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submitting $(basename "$0") (config: $CONFIG)"
  exec sbatch "$0" "$CONFIG"
fi

echo "Job started: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unset}"
echo "Config: $CONFIG"

source /home/x-dcasson/snowda/bin/activate

which python
python -c "import sys; print('python:', sys.executable); print('prefix:', sys.prefix)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$REPO/src"
python -m snowda.workflows.run_summa_historic "$CONFIG"

echo "Job finished: $(date)"
