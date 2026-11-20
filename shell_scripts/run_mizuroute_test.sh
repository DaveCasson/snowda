#!/bin/bash
#SBATCH --account=ees250064
#SBATCH -J snowda_route
#SBATCH -o /home/x-dcasson/GitRepos/snowda/shell_scripts/logs/%x-%j.out
#SBATCH -e /home/x-dcasson/GitRepos/snowda/shell_scripts/logs/%x-%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=1G

# mizuRoute for the lumped Ross setup (mizurouteANVIL.control).
# route_runoff.exe is a single process; its RPATH already points at the
# Anvil netCDF and OpenMPI libraries it was built with.
# Inside a batch job, SLURM_STEP_ID makes that Open MPI treat a direct
# exec as srun. The build has no SLURM PMI support, so MPI_Init aborts.
# Dropping the step id starts one local process, the same way it runs
# on a login node. OMPI_MCA_ess=singleton does not help: Open MPI
# overwrites it while SLURM_STEP_ID is set.
#
# Submit with either of:
#   sbatch shell_scripts/run_mizuroute_test.sh
#   ./shell_scripts/run_mizuroute_test.sh

set -euo pipefail

REPO=/home/x-dcasson/GitRepos/snowda
mkdir -p "$REPO/shell_scripts/logs"

# #SBATCH lines are comments unless sbatch reads them. Submit from here
# when this process is not already a SLURM job, then let the job run below.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submitting $(basename "$0")"
  exec sbatch "$0"
fi

echo "Job started: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Node: $(hostname)"

env -u SLURM_STEP_ID -u SLURM_STEPID \
  /home/x-dcasson/GitRepos/mizuRoute/route/bin/route_runoff.exe \
  "$REPO/models/ross_lumped/mizuRoute/mizurouteANVIL.control"

echo "Job finished: $(date)"
