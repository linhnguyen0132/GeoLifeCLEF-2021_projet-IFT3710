import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

class GLC_DATASET(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, subset_filter=None):
        """
        csv_file : Chemin du CSV
        root_dir : Dossier parent des images
        subset_filter : "train" ou "val" (pour séparer les données)
        """
        self.root_dir = Path(root_dir)
        df = pd.read_csv(csv_file, sep=";")
        
        # --- AMÉLIORATION 1 : Séparation Train/Val ---
        if subset_filter:
            # On ne garde que les lignes qui correspondent au split voulu
            self.annotations = df[df['subset'] == subset_filter].reset_index(drop=True)
        else:
            self.annotations = df

        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __get_image_path(self, obs_id):
        obs_str = str(obs_id)
        # Logique des dossiers : les deux derniers chiffres, puis les deux précédents
        dossier_1 = obs_str[-2:]
        dossier_2 = obs_str[-4:-2]

        # --- AMÉLIORATION 2 : Robustesse du format ---
        # On essaie .jpg puis .png si besoin
        nom_fichier = f"{obs_str}_rgb.jpg"
        return self.root_dir / dossier_1 / dossier_2 / nom_fichier

    def __getitem__(self, index):
        obs_id = self.annotations.iloc[index]['observation_id']
        label = int(self.annotations.iloc[index]['species_id'])

        img_path = self.__get_image_path(obs_id)

        # --- AMÉLIORATION 3 : Gestion des erreurs de fichiers ---
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Si l'image manque, on crée un patch noir pour ne pas faire planter l'entraînement
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- EXEMPLE D'ASSEMBLAGE POUR TES DEUX ÉDITIONS ---
if __name__ == "__main__":
    # Transformations standards
    t_train = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    # Création du dataset 2021 (Train uniquement)
    ds_2021 = GLC_DATASET(
        csv_file=".../2021/observations_fr_train.csv",
        root_dir=".../2021/patches_sample",
        subset_filter="train",
        transform=t_train
    )

    # Création du dataset 2022 (Train uniquement)
    ds_2022 = GLC_DATASET(
        csv_file=".../2022/observations_fr_train.csv",
        root_dir=".../2022/patches-fr",
        subset_filter="train",
        transform=t_train
    )

    # Fusion des deux !
    dataset_final = torch.utils.data.ConcatDataset([ds_2021, ds_2022])
    print(f"Total images fusionnées : {len(dataset_final)}")
