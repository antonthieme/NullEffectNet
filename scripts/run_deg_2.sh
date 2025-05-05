#!/bin/bash
#SBATCH --job-name=run_deg_2
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=../logs/deg_2/%j.out
#SBATCH --error=../logs/deg_2/%j.err

# Load modules or source Conda
source ~/.bashrc
micromamba activate nen_env

# Run script
python /home/users/thiemea/projects/NullEffectNet/scripts/deg_2.py
