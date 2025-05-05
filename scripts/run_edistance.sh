#!/bin/bash
#SBATCH --job-name=run_edistance
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=../logs/run_edistance/%j.out
#SBATCH --error=../logs/run_edistance/%j.err

# Load modules or source Conda
source ~/.bashrc
micromamba activate nen_env

# Run script
python /home/users/thiemea/projects/NullEffectNet/scripts/edistance.py
