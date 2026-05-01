# Fichier: calculer_poids.py
import pandas as pd
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

print("Chargement du CSV...")
df = pd.read_csv("/home/abdkarimouatt/projects/def-sponsor00/geolifeclef/data/observations/observations_fr_train.csv", sep=';')
y_train = df['species_id'].values

print("Calcul des poids...")
poids_bruts = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
poids_ajustes = np.clip(poids_bruts, a_min=0, a_max=50.0)

# Convertir en tenseur et SAUVEGARDER
class_weights_tensor = torch.FloatTensor(poids_ajustes)
torch.save(class_weights_tensor, 'poids_classes_geolifeclef.pt')

print("Terminé ! Poids sauvegardés dans 'poids_classes_geolifeclef.pt'")
