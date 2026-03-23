#!/bin/bash
#SBATCH --job-name=xgboost                          # Job name
#SBATCH --cpus-per-task=1                                   # Ask for 1 CPUs
#SBATCH --mem=8G                                           # Ask for 2 GB of RAM
#SBATCH --time=3:00:00		# Ask for 2h runtime
#SBATCH --output=slurm-%j-%x.out   # Output file 
#SBATCH --error=slurm-%j-%x.error  # Error file
#
#

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"


set -x
module load python

source /home/linhnguyen/projet/bin/activate


python -u ift3710_brut-fr.py

echo "Job finished at $(date)"


