#!/bin/bash
#SBATCH --job-name=deg_array
#SBATCH --output=../logs/deg/slurm_%A_%a.out
#SBATCH --error=../logs/deg/slurm_%A_%a.err
#SBATCH --array=0-1    # Adjust based on your file count
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=256G

# Load modules or source Conda
source ~/.bashrc
micromamba activate nen_env

# Run script
python /home/users/thiemea/projects/NullEffectNet/scripts/deg_array.py --job_id $SLURM_ARRAY_TASK_ID