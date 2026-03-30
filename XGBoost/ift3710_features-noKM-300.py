import numpy as np
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
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
from metrics import top_30_error_rate
from metrics import top_k_error_rate
from soumission import generate_submission_file

from sklearn.impute import SimpleImputer
print('XGBoost features - Données FR - 2021 - weighted sampling ')

#import cupy as cp
path_2021 = "~/projects/def-sponsor00/geolifeclef/data"
path_2022 = "~/projects/def-sponsor00/geolifeclef/data_2022"

print('Données utilisées = 2021')

path_obs_fr = path_2021 + "/observations/observations_fr_train.csv"
df_obs = pd.read_csv(path_obs_fr, sep=";", index_col="observation_id")

#TEST

path_test = path_2021 + "/observations/observations_fr_test.csv"
df_test = pd.read_csv(path_test, sep=';', index_col='observation_id')

print(df_obs.columns)
df_obs.head()

path_env = path_2021 + "/pre-extracted/environmental_vectors.csv"
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

#merge test
df_test_merged = df_test.merge(
    df_env,
    on='observation_id',
    how='left'
)

corrected_observation_ids = df_test_merged.index.values
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
#nb_cluster = 100
#coords = df_reduced[["latitude", "longitude"]]
#kmeans = KMeans(n_clusters = nb_cluster, random_state =42).fit(coords)
#df_reduced["coords"]= kmeans.labels_

#df_reduced.drop(columns=["latitude", "longitude"], inplace=True)
#print('Nombre de clusters =  ' + str(nb_cluster))


# Polynomial features
# On définit les colonnes numériques à nettoyer
#bio_features = ['bldfie', 'bio_11', 'bio_1', 'bio_10']

bio_features = ['bio_18', 'bio_7', 'bldfie', 'bio_5', 'phihox', 'bio_15', 'bio_10', 'longitude', 'bio_17', 'bio_9', 'latitude', 'bio_1', 'bio_6', 'bio_14', 'bio_11']

print('list des features utilisées : ' + str(bio_features))


# Créer l'imputeur
imputer = SimpleImputer(strategy='mean')

# Appliquer sur le dataset réduit (on le fait avant de crécer les poly features)
df_reduced[bio_features] = imputer.fit_transform(df_reduced[bio_features])
df_test_clean = pd.DataFrame(imputer.transform(df_test_merged[bio_features]), 
                             columns=bio_features, 
                             index=df_test_merged.index)

print("Nombre de NaNs après imputation :", df_reduced[bio_features].isna().sum().sum())

# poly features
poly = PolynomialFeatures(degree=3,interaction_only=True, include_bias=False)

interactions = poly.fit_transform(df_reduced[bio_features])
feature_names_poly = poly.get_feature_names_out(bio_features)


# On crée un nouveau DataFrame pour le train/val qui contient toutes les interactions
df_poly = pd.DataFrame(interactions, columns=feature_names_poly, index=df_reduced.index)
df_poly['species_id'] = df_reduced['species_id']
df_poly['subset'] = df_reduced['subset']

# transf pour test
x_test_poly_raw = poly.transform(df_test_clean)
x_test = pd.DataFrame(x_test_poly_raw, columns=feature_names_poly)


classes = df_reduced["species_id"].unique()

obs_id_train = df_reduced.index[df_reduced["subset"] == "train"].values
obs_id_val = df_reduced.index[df_reduced["subset"] == "val"].values

le = LabelEncoder()
# Fit on ALL species ids so both train and val share the same mapping
le.fit(classes)

y_train = le.transform(df_poly.loc[obs_id_train]["species_id"].values)
y_val   = le.transform(df_poly.loc[obs_id_val]["species_id"].values)

n_val = len(obs_id_val)
print("Validation set size: {} ({:.1%} of train observations)".format(n_val, n_val / len(df_obs)))

x_train = df_poly.loc[obs_id_train].drop(columns=["species_id", "subset"])
x_val = df_poly.loc[obs_id_val].drop(columns=["species_id", "subset"])


# weights
print('Weights = log')

# compter le nombres d'obs
counts = np.bincount(y_train)

#adoucir les weights
weights_map = 1.0 / np.log1p(counts)

# Appliquer le poids à chaque ligne du dataset
weights = weights_map[y_train]

# Normaliser pour que la moyenne des poids soit 1.0 (très important pour XGBoost)
weights *= len(weights) / np.sum(weights)

#weights = compute_sample_weight(class_weight = 'balanced', y = y_train)

#x_train_gpu = cp.array(x_train)
#x_val_gpu   = cp.array(x_val)

params = {
    'objective': 'multi:softprob',
    'num_class': len(le.classes_),
    'eval_metric': 'mlogloss',
    'max_depth': 3,
    'learning_rate': 0.04,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'tree_method': 'hist',
    'device' : 'cpu',
    'seed': 42,
    'grow_policy': 'lossguide',
    'max_bin': 64,
    'n_estimators': 500,
    'max_leaves': 64,
    'reg_lambda' : 15,
    'min_child_weight' : 10
}
xgb_cls = xgb.XGBClassifier(**params, early_stopping_rounds=20)

# --- 1. Entraînement sur GPU ---
# On s'assure d'utiliser les données GPU créées
xgb_cls.fit(x_train, y_train, sample_weight=weights, eval_set=[(x_train, y_train),(x_val, y_val)], verbose=0)

# --- 2. Prédictions pour les métriques ---
# On récupère les probabilités pour le Top-30 (besoin des scores complets)
y_score_train = xgb_cls.predict_proba(x_train)
y_score_val = xgb_cls.predict_proba(x_val)
y_score_test = xgb_cls.predict_proba(x_test)

# On récupère les classes pour le Top-1 (Accuracy classique)
y_pred_train = xgb_cls.predict(x_train)
y_pred_val   = xgb_cls.predict(x_val)
y_pred_test = xgb_cls.predict(x_test)


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

# SUBMISSION FILE
top_30_indices = np.argsort(-y_score_test, axis=1)[:, :30]
s_pred = [le.inverse_transform(indices) for indices in top_30_indices]

submission_name = f"submission_{job_id}.csv"

generate_submission_file(submission_name, corrected_observation_ids, s_pred)

print(f"Fichier de soumission généré : {submission_name}")