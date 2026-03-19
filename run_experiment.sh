#!/bin/bash
#SBATCH --account=e33110
#SBATCH --job-name=bayes_tuning
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=46:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal

module purge
module load python/3.13.0-gcc-8.5.0

cd $SLURM_SUBMIT_DIR

source $(poetry env info -p)/bin/activate

export REP_ID=$SLURM_ARRAY_TASK_ID

python -m src.main
