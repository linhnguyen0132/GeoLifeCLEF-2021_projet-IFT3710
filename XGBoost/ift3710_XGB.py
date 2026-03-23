
import pandas as pd
import xgboost as xgb
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


#import cupy as cp

path_obs_fr = "~/projects/def-sponsor00/geolifeclef/data/observations/observations_fr_train.csv"
df_obs_fr = pd.read_csv(path_obs_fr, sep=";", index_col="observation_id")

path_obs_us = "~/projects/def-sponsor00/geolifeclef/data/observations/observations_us_train.csv"
df_obs_us = pd.read_csv(path_obs_us, sep=";", index_col="observation_id")

df_obs = pd.concat((df_obs_fr, df_obs_us))

print(df_obs.columns)
df_obs.head()

path_env = "~/projects/def-sponsor00/geolifeclef/data/pre-extracted/environmental_vectors.csv"
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

df_filtered = df_merged[df_merged["species_id"].isin(valid_species)]

# Vérification
print(f"Nombre d'espèces restantes : {len(valid_species)}")
print(f"Nombre de lignes restantes : {len(df_filtered)}")
#top_100_species = df_merged['species_id'].value_counts().nlargest(100).index

#nb_especes = 300
#print('Échantillion aléatoire de ' + str(nb_especes))
#top_n_random = (pd.Series(valid_species).sample(n=nb_especes, random_state=42).values)

# 2. Créer un dataset réduit
#df_reduced = df_merged[df_merged['species_id'].isin(top_n_random)].copy()

#print(f"Ancien nombre de lignes : {len(df_merged)}")
#print(f"Nouveau nombre de lignes : {len(df_reduced)}")

#Garder une version originale des données
#df = df_reduced.copy()


# enlever longitude et latitude et utiliser les clusters
coords = df_filtered[["latitude", "longitude"]]
kmeans = KMeans(n_clusters = 500, random_state =42).fit(coords)
df_filtered["coords"]= kmeans.labels_

df_filtered.drop(columns=["latitude", "longitude"], inplace=True)


classes = df_filtered["species_id"].unique()

obs_id_train = df_filtered.index[df_filtered["subset"] == "train"].values
obs_id_val = df_filtered.index[df_filtered["subset"] == "val"].values

le = LabelEncoder()
# Fit on ALL species ids so both train and val share the same mapping
le.fit(classes)

y_train = le.transform(df_filtered.loc[obs_id_train]["species_id"].values)
y_val   = le.transform(df_filtered.loc[obs_id_val]["species_id"].values)

n_val = len(obs_id_val)
print("Validation set size: {} ({:.1%} of train observations)".format(n_val, n_val / len(df_obs)))

x_train = df_filtered.loc[obs_id_train].drop(columns=["species_id", "subset"])
x_val = df_filtered.loc[obs_id_val].drop(columns=["species_id", "subset"])



#x_train_gpu = cp.array(x_train)
#x_val_gpu   = cp.array(x_val)

params = {
    'objective': 'multi:softprob',
    'num_class': len(le.classes_),
    'eval_metric': 'mlogloss',
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'tree_method': 'hist',
    'device' : 'cuda',
    'seed': 42,
    'grow_policy': 'lossguide',
    'max_bin': 64,
    'n_estimators': 500,
    'max_leaves': 15
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
# Récupérer l'importance des variables
importances = xgb_cls.feature_importances_
feature_names = x_train.columns

# Créer un graphique simple
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Importance")
plt.title("Quelles variables influencent le plus la présence des espèces ?")
plt.savefig(f'var_importantes__{job_id}.png')

# Création d'un DataFrame pour les importances
feat_imp = pd.Series(xgb_cls.feature_importances_, index=x_train.columns)
feat_imp.nlargest(15).plot(kind='barh', figsize=(10,6))
plt.title("Top 15 des variables les plus importantes")
plt.savefig(f'top_15f_{job_id}.png')
