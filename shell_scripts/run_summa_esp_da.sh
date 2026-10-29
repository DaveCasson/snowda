#!/bin/bash
#SBATCH --account=ees250064
#SBATCH -J snowda_da
#SBATCH -o /home/x-dcasson/GitRepos/snowda/shell_scripts/logs/%x-%j.out
#SBATCH -e /home/x-dcasson/GitRepos/snowda/shell_scripts/logs/%x-%j.err
#SBATCH -N 1
#SBATCH --cpus-per-task=50
#SBATCH --time=1-00:00:00
#SBATCH --mem=4G

# SUMMA particle-filter assimilation, then ESP traces (notebooks/run_summa_da.ipynb).
# Anvil shared is the default partition: 4-day limit, MaxMemPerCPU=1896M.
# Memory above cpus * 1896M is scheduled by raising the CPU count.
# Keep --cpus-per-task equal to num_workers in the YAML.
#
# Running this script submits it. The workload starts only inside the job.
#   ./run_summa_da.sh
#   ./run_summa_da.sh /path/to/other_da.yaml
#
# Example YAMLs still use laptop paths.

set -euo pipefail

REPO=/home/x-dcasson/GitRepos/snowda
CONFIG="${1:-$REPO/runs/examples/ross_summa_esp_da.yaml}"
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
python -m snowda.workflows.run_summa_da "$CONFIG"

echo "Job finished: $(date)"
