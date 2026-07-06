#!/bin/bash

# Parametrized variant of run_queue.slurm: pass the executable and its arguments
# on the sbatch command line instead of editing this file per run, e.g.:
#     sbatch bash/run_exe.slurm ./queue_matrix 32 8
#     sbatch bash/run_exe.slurm ./queue_dsweep 0 24

#SBATCH --output=/home/rbergsma/queueing_project/dynaplex_queue/Log/%j.out
#SBATCH --error=/home/rbergsma/queueing_project/dynaplex_queue/Log/%j.out
#SBATCH --time 3:00:00 --exclusive --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --partition genoa --mem=64G

module purge
module load 2024
module load GCCcore/13.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Running on $(hostname): $*"

cd /home/rbergsma/queueing_project/dynaplex_queue/out/LinRel/bin

"$@"
