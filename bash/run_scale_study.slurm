#!/bin/bash

# One scaling-study cell per job. Examples:
#   sbatch bash/run_scale_study.slurm level=small phase=all seed=1 preset=0 rvi_m=40
#   sbatch bash/run_scale_study.slurm level=large phase=ppo seed=3 preset=2

#SBATCH --output=/home/rbergsma/queueing_project/dynaplex_queue/Log/scale_%j.out
#SBATCH --error=/home/rbergsma/queueing_project/dynaplex_queue/Log/scale_%j.out
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=genoa
#SBATCH --mem=32G

module purge
module load 2024
module load GCCcore/13.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /home/rbergsma/queueing_project/dynaplex_queue/out/LinRel/bin
echo "Running on $(hostname): queue_scale_study $*"
./queue_scale_study "$@"
