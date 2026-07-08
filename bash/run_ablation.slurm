#!/bin/bash

# Ablation-matrix runner: one cell x seed(s) per SMALL shared-node job.
#     sbatch bash/run_ablation.slurm cell=base seeds=1
#     sbatch bash/run_ablation.slurm cell=dper seeds=3 updates=6000
# Results land in Log/abl_<jobid>.out.

#SBATCH --output=/home/rbergsma/queueing_project/dynaplex_queue/Log/abl_%j.out
#SBATCH --error=/home/rbergsma/queueing_project/dynaplex_queue/Log/abl_%j.out
#SBATCH --time 4:00:00 --nodes=1 --ntasks=1 --cpus-per-task=16 --partition genoa --mem=28G

module purge
module load 2024
module load GCCcore/13.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Running on $(hostname): queue_ablation $*"

cd /home/rbergsma/queueing_project/dynaplex_queue/out/LinRel/bin

./queue_ablation "$@"
