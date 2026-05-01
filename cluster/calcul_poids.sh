#!/bin/bash
##SBATCH --account=def-sponsor00
#SBATCH --job-name=calcul_poids
#SBATCH --time=00:10:00       # 10 minutes suffisent largement
#SBATCH --cpus-per-task=1     # 1 seul coeur
#SBATCH --mem=8G              # 8 Go de RAM pour charger le CSV avec pandas
# Ne demande AUCUN GPU ici (--gres=gpu:0 si tu l'avais par défaut)

module load python/3.10                    # (Ajuste avec ton module habituel)
source $SCRATCH/venv_cluster/bin/activate  # (Si tu as un env virtuel)

python calculer_poids.py
