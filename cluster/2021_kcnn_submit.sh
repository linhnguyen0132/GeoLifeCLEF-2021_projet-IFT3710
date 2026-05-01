#!/bin/bash
#SBATCH --account=def-sponsor00               # Compte du cours
#SBATCH --job-name=kcnn_2021                  # Nom du job
#SBATCH --time=24:00:00                       # Temps max (6h, early stopping vers ~3h)
#SBATCH --cpus-per-task=10                    # CPU pour le dataloader (num_workers=4)
#SBATCH --gres=gpu:1                          # 1 GPU
#SBATCH --mem=64G                             # Mémoire
#SBATCH --output=out/resultats_2021_%j.out    # Logs (%j = job ID)
#SBATCH --error=err/erreurs_2021_%j.err       # Le fichier où s'écriront les crashs



echo "--- Préparation de l'environnement ---"
# Chargement des modules de base (Standard sur les clusters universitaires)
module load python/3.10

# Activation de ton environnement virtuel (Si tu en as créé un sur le cluster)
source $SCRATCH/venv_cluster/bin/activate 

echo "--- DÉBUT DU SBATCH ! ---"

python main.py

echo "--- FIN DU SBATCH ! ---"
