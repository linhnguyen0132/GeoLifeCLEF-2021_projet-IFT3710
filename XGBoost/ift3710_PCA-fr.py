
import pandas as pd
import xgboost as xgb
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import accuracy_score, classification_report
#from google.colab import drive
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from metrics import top_30_error_rate
from metrics import top_k_error_rate

from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print('XGBoost PCA - Données 2022')
#import cupy as cp
path_2021 = "~/projects/def-sponsor00/geolifeclef/data"
path_2022 = "~/projects/def-sponsor00/geolifeclef/data_2022"

path_obs_fr = path_2022 + "/observations/observations_fr_train.csv"
df_obs= pd.read_csv(path_obs_fr, sep=";", index_col="observation_id")

print(df_obs.columns)
df_obs.head()

path_env = path_2022 + "/pre-extracted/environmental_vectors.csv"
df_env = pd.read_csv(
    path_env,
    sep=';',
    index_col="observation_id"
)

df_env.head()

df_merged = df_obs.merge(
    df_env,
    on='observation_id',
    how='left'
)

df_merged.head()

# prendre les espèces avec observations plus grandes que 20
species_counts = df_merged["species_id"].value_counts()

valid_species = species_counts[species_counts >= 10].index


#top_100_species = df_merged['species_id'].value_counts().nlargest(100).index

nb_especes = 500
print('Échantillion aléatoire de ' + str(nb_especes))
top_n_random = (pd.Series(valid_species).sample(n=nb_especes, random_state=42).values)

# 2. Créer un dataset réduit
df_reduced = df_merged[df_merged['species_id'].isin(top_n_random)].copy()

print(f"Ancien nombre de lignes : {len(df_merged)}")
print(f"Nouveau nombre de lignes : {len(df_reduced)}")

# enlever longitude et latitude et utiliser les clusters
nb_cluster = 100
coords = df_reduced[["latitude", "longitude"]]
kmeans = KMeans(n_clusters = nb_cluster, random_state =42).fit(coords)
df_reduced["coords"]= kmeans.labels_

df_reduced.drop(columns=["latitude", "longitude"], inplace=True)
print('Nombre de clusters =  ' + str(nb_cluster))



#Garder une version originale des données
df = df_reduced.copy()

classes = df_reduced["species_id"].unique()

obs_id_train = df_reduced.index[df_reduced["subset"] == "train"].values
obs_id_val = df_reduced.index[df_reduced["subset"] == "val"].values

bio_cols = [f'bio_{i}' for i in range(1, 20)]
scaler = StandardScaler()
x_train_bio_scaled = scaler.fit_transform(df_reduced.loc[obs_id_train, bio_cols])
x_val_bio_scaled = scaler.transform(df_reduced.loc[obs_id_val, bio_cols])

#remplacer les NaNs par la médiane
imputer = SimpleImputer(strategy='median')

# 2. On "fit" sur le train et on "transform" tout le monde
# On fait ça AVANT le StandardScaler
x_train_bio = imputer.fit_transform(df_reduced.loc[obs_id_train, bio_cols])
x_val_bio = imputer.transform(df_reduced.loc[obs_id_val, bio_cols])

# 3. Maintenant tu peux faire ton Scaling et ta PCA normalement
scaler = StandardScaler()
x_train_bio_scaled = scaler.fit_transform(x_train_bio)
x_val_bio_scaled = scaler.transform(x_val_bio)

nb_components = 10

print('Nombre de composantes PCA : ' + str(nb_components))
pca = PCA(n_components=nb_components)
pca_train = pca.fit_transform(x_train_bio_scaled)
pca_val = pca.transform(x_val_bio_scaled)


#3. LABEL ENCODING
le = LabelEncoder()
le.fit(df_reduced["species_id"]) # On fit sur tout l'échantillon réduit
y_train = le.transform(df_reduced.loc[obs_id_train, "species_id"])
y_val   = le.transform(df_reduced.loc[obs_id_val, "species_id"])

# 4. ASSEMBLAGE FINAL (hstack)
# On s'assure que coords est bien en 2D pour hstack avec .values.reshape(-1, 1)
train_coords = df_reduced.loc[obs_id_train, 'coords'].values.reshape(-1, 1)
val_coords = df_reduced.loc[obs_id_val, 'coords'].values.reshape(-1, 1)

x_train = np.hstack([pca_train, train_coords])
x_val = np.hstack([pca_val, val_coords])
#x_train_gpu = cp.array(x_train)
#x_val_gpu   = cp.array(x_val)

params = {
    'objective': 'multi:softprob',
    'num_class': len(le.classes_),
    'eval_metric': 'mlogloss',
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'tree_method': 'hist',
    'device' : 'cpu',
    'seed': 42,
    'grow_policy': 'lossguide',
    'max_bin': 64,
    'n_estimators': 500,
    'max_leaves': 15,
    'reg_lambda' : 15
}
xgb_cls = xgb.XGBClassifier(**params, early_stopping_rounds=20)
# --- 1. Entraînement sur GPU ---
# On s'assure d'utiliser les données GPU créées
xgb_cls.fit(x_train, y_train, eval_set=[(x_train, y_train),(x_val, y_val)], verbose=0)

# --- 2. Prédictions pour les métriques ---
# On récupère les probabilités pour le Top-30 (besoin des scores complets)
y_score_train = xgb_cls.predict_proba(x_train)
y_score_val = xgb_cls.predict_proba(x_val)

# On récupère les classes pour le Top-1 (Accuracy classique)
y_pred_train = xgb_cls.predict(x_train)
y_pred_val   = xgb_cls.predict(x_val)

# --- 3. Calcul des scores ---
# Métriques Train
acc_train = accuracy_score(y_train, y_pred_train)
error_30_train = top_30_error_rate(y_train, y_score_train)

# Métriques Val
acc_val = accuracy_score(y_val, y_pred_val)
error_30_val = top_30_error_rate(y_val, y_score_val)

print(f"--- RÉSULTATS VERSION ---")
print(f"TRAIN | Accuracy: {acc_train:.2%} | Top-30 error rate: {error_30_train:.2%}")
print(f"VAL   | Accuracy: {acc_val:.2%} | Top-30 error rate: {error_30_val:.2%}")

error_1_train = top_k_error_rate(y_train, y_score_train,1)
error_1_val = top_k_error_rate(y_val, y_score_val,1)

print(f"TRAIN | Top-1 error rate: {error_1_train:.2%}")
print(f"VAL   | Top-1 error rate: {error_1_val:.2%}")

# Loss Entropy
results = xgb_cls.evals_result()

# Accéder aux listes de scores pour le train et la val
train_loss = results['validation_0']['mlogloss']
val_loss = results['validation_1']['mlogloss']

print(f"Dernière Cross-Entropy Train : {train_loss[-1]:.4f}")
print(f"Dernière Cross-Entropy Val   : {val_loss[-1]:.4f}")


# courbe d'entraînement

job_id = os.environ.get('SLURM_JOB_ID', 'local')
 
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Train (mlogloss)')
plt.plot(val_loss, label='Validation (mlogloss)')
plt.axvline(x=xgb_cls.best_iteration, color='r', linestyle='--', label='Meilleur Arbre')

plt.title("Évolution de la Cross-Entropy (Log Loss)")
plt.xlabel("Nombre d'arbres (n_estimators)")
plt.ylabel("M-LogLoss")
plt.legend()
plt.grid(True)
plt.savefig(f'courbe_train_{job_id}.png')

# 1. Générer les noms dynamiquement selon la taille réelle des données
n_pca = pca_train.shape[1]  # Récupère le nombre exact de colonnes PCA (ex: 10)
pca_cols = [f'pca_bio_{i}' for i in range(1, n_pca + 1)]
feature_names = pca_cols + ['coords']

# 2. Vérification de sécurité (pour ton terminal)
importances = xgb_cls.feature_importances_
print(f"Nombre de variables détectées : {len(feature_names)}")
print(f"Nombre d'importances fournies : {len(importances)}")

# 3. Création du graphique
plt.figure(figsize=(10, 8))
# On trie pour que ce soit plus lisible
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)

feat_imp.plot(kind='barh')
plt.xlabel("Importance")
plt.title(f"Importance des variables ({n_pca} PCA + Clusters)")
plt.tight_layout() # Évite que les noms soient coupés
plt.savefig(f'var_importantes__{job_id}.png')