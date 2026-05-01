import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from v2_dataset import GLC_DATASET, TransformWrapper

# --- 1. CONFIGURATION ---
BASE_DIR = os.environ.get("SLURM_TMPDIR", "/home/abdkarimouatt/projects/def-sponsor00/geolifeclef")
CSV_OBS_PATH = f"{BASE_DIR}/data_2022/observations/observations_fr_train.csv"
CSV_ENV_PATH = f"{BASE_DIR}/data_2022/pre-extracted/environmental_vectors.csv"
IMG_DIR      = f"{BASE_DIR}/data_2022/patches-fr"

CHEMIN_CSV_MENACEES = "/home/abdkarimouatt/projects/def-sponsor00/geolifeclef/especes_menacees/especes_menacees_2022.csv"
CHEMIN_MODELE = "meilleur_modele_sensio.pth"

# --- CONSTRUCTION DU MODÈLE  ---
class ModeleGagnant(nn.Module):
    def __init__(self, num_classes, num_tabular_features=2):
        super().__init__()
        self.cnn = models.resnet50(weights=None) 
        ancienne_conv1 = self.cnn.conv1
        self.cnn.conv1 = nn.Conv2d(4, ancienne_conv1.out_channels, kernel_size=ancienne_conv1.kernel_size, stride=ancienne_conv1.stride, padding=ancienne_conv1.padding, bias=False)
        self.cnn_out_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_out_features + 128, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, images, tabular):
        x_vision = self.cnn(images)
        x_tabulaire = self.mlp(tabular)
        x_fusion = torch.cat((x_vision, x_tabulaire), dim=1)
        return self.classifier(x_fusion)

# --- 3. CHARGEMENT DES DONNÉES ---
print("1. Chargement de la liste ...")
# Charge le fichier des animaux en danger
df_menacees = pd.read_csv(CHEMIN_CSV_MENACEES, sep=",")
# On store les ID dans un set pour accélerer la vérification
liste_id_menaces = set(df_menacees['species_id'].values)
# On paire les ID avec le nom de l'espèce correspondante avec zip() et ensuite on crée un  dictionnaire pour les tuples avec dict()
traducteur_noms = dict(zip(df_menacees['species_id'], df_menacees['GBIF_species_name']))

print("2. Préparation du Dataloader...")
val_transform = transforms.Compose([
    transforms.Resize((256, 256), antialias=True),
    transforms.CenterCrop(224),
])

dataset_complet = GLC_DATASET(csv_file=CSV_OBS_PATH, root_dir=IMG_DIR)
test_dataset = TransformWrapper(dataset_complet, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

# --- 4. INITIALISATION  ---
print("3. Chargement des poids ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ModeleGagnant(num_classes=4911, num_tabular_features=4).to(device)
model.load_state_dict(torch.load(CHEMIN_MODELE, map_location=device)['model_state_dict'])
model.eval()

# --- 5. LE SCAN GÉOGRAPHIQUE ---
print("4. Début du scan géographique ... ")
alertes_conservation = []

with torch.no_grad():
    pbar = tqdm(test_loader, desc="Analyse en cours")
    for images, tabular, labels in pbar:
        images = images.to(device)
        tabular = tabular.to(device)

        outputs = model(images, tabular)
        probabilites = torch.softmax(outputs, dim=1)
        probs_top30, ids_top30 = probabilites.topk(30, dim=1)

        for i in range(ids_top30.size(0)):
            predictions_locales = ids_top30[i].cpu().numpy()
            probs_locales = probs_top30[i].cpu().numpy()

            lat = tabular[i][0].item()
            lon = tabular[i][1].item()

            for rang, (sp_id, proba) in enumerate(zip(predictions_locales, probs_locales)):
                if sp_id in liste_id_menaces:
                    nom_espece = traducteur_noms.get(sp_id, "Nom inconnu")
                    alertes_conservation.append({
                        'latitude': lat,
                        'longitude': lon,
                        'species_id': sp_id,
                        'nom_espece': nom_espece,
                        'probabilite_modele': round(proba * 100, 2),
                        'rang_top30': rang + 1
                    })

# --- 6. SAUVEGARDE ---
df_alertes = pd.DataFrame(alertes_conservation)
df_alertes.to_csv("rapport_habitats_critiques.csv", index=False)
print(f"\n Prédicton terminée : {len(df_alertes)} .")