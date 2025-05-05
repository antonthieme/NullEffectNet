#!/bin/bash
#SBATCH --job-name=edist_array
#SBATCH --output=../logs/run_edistance/slurm_%A_%a.out
#SBATCH --error=../logs/run_edistance/slurm_%A_%a.err
#SBATCH --array=0-0    # Adjust based on your file count
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=50
#SBATCH --time=24:00:00
#SBATCH --mem=400G

# Load modules or source Conda
source ~/.bashrc
micromamba activate nen_env

# Run script
python /home/users/thiemea/projects/NullEffectNet/scripts/edistance_array.py --job_id $SLURM_ARRAY_TASK_ID