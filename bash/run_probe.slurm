#!/bin/bash

# Fast-iteration runner: one probe configuration per SMALL shared-node job.
# 16 cores schedules quickly via backfill; PPO training is single-threaded and
# the eval parallelizes fine over 16 cores.  All probe knobs pass through:
#     sbatch bash/run_probe.slurm exp=2 reward=2 seeds=4 updates=6000 gamma=0.997 bench=0
#     sbatch bash/run_probe.slurm method=dcl exp=3 reward=2 seeds=1 bench=0
#     sbatch --time=12:00:00 bash/run_probe.slurm exp=2 seeds=2 updates=30000 ...
# Results land in Log/probe_<jobid>.out.

#SBATCH --output=/home/rbergsma/queueing_project/dynaplex_queue/Log/probe_%j.out
#SBATCH --error=/home/rbergsma/queueing_project/dynaplex_queue/Log/probe_%j.out
#SBATCH --time 4:00:00 --nodes=1 --ntasks=1 --cpus-per-task=16 --partition genoa --mem=28G

module purge
module load 2024
module load GCCcore/13.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Running on $(hostname): queue_dcl_probe $*"

cd /home/rbergsma/queueing_project/dynaplex_queue/out/LinRel/bin

./queue_dcl_probe "$@"
