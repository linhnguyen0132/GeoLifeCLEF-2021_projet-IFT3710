import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

class GLC_DATASET(Dataset):
    def __init__(self, csv_obs, csv_env, root_dir, transform=None):
        df_obs = pd.read_csv(csv_obs, sep=";")
        df_env = pd.read_csv(csv_env, sep=";")

        self.annotations = pd.merge(df_obs, df_env, on="observation_id", how="left")
        # On limite chaque espèce à un maximum de 800 images pour équilibrer le dataset
        seuil_max = 800
        print(f"Plafonnement des classes à {seuil_max} images max...")
        self.annotations = self.annotations.groupby('species_id').apply(lambda x: x.sample(n=min(len(x), seuil_max), random_state=42)).reset_index(drop=True)
        print(f"Taille finale du dataset : {len(self.annotations)} lignes.")

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.cols_env = [
            "bio_1", "bio_2", "bio_3", "bio_4", "bio_5", "bio_6", "bio_7", "bio_8",
            "bio_9", "bio_10", "bio_11", "bio_12", "bio_13", "bio_14", "bio_15",
            "bio_16", "bio_17", "bio_18", "bio_19",
            "bdticm", "bldfie", "cecsol", "clyppt", "orcdrc", "phihox", "sltppt", "sndppt"
        ]

        self.annotations[self.cols_env] = self.annotations[self.cols_env].fillna(0.0)

        # --- NORMALISATION DE TOUTES LES DONNÉES TABULAIRES ---
        # On regroupe la latitude, la longitude et les 27 variables
        colonnes_a_normaliser = ['latitude', 'longitude'] + self.cols_env

        print("Normalisation Z-Score des 29 variables tabulaires en cours...")
        for col in colonnes_a_normaliser:
            moyenne = self.annotations[col].mean()
            ecart_type = self.annotations[col].std()
            if ecart_type != 0:
                self.annotations[col] = (self.annotations[col] - moyenne) / ecart_type
        # On stocke tout dans des arrays NumPy
        self.obs_ids = self.annotations['observation_id'].values
        self.labels = self.annotations['species_id'].values.astype(np.int64)

        # On pré-calcule et on nettoie tout le tableau tabulaire
        toutes_les_variables = self.annotations[colonnes_a_normaliser].values.astype(np.float32)
        toutes_les_variables = np.nan_to_num(toutes_les_variables, nan=0.0, posinf=10.0, neginf=-10.0)
        self.tabular_data = np.clip(toutes_les_variables, -1e6, 1e6)

        #On instancie les transformateurs une seule fois
        self.to_tensor = transforms.ToTensor()
        self.norm_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.norm_nir = transforms.Normalize(mean=[0.5], std=[0.5])


    def __len__(self):
        return len(self.annotations)

    def __get_image_paths(self, obs_id):
        obs_str = str(obs_id)
        dossier_1 = obs_str[-2:]
        dossier_2 = obs_str[-4:-2]

        chemin_rgb = self.root_dir / dossier_1 / dossier_2 / f"{obs_str}_rgb.jpg"
        chemin_nir = self.root_dir / dossier_1 / dossier_2 / f"{obs_str}_near_ir.jpg"
        return chemin_rgb, chemin_nir

    def __getitem__(self, index):
        obs_id = self.obs_ids[index]
        label  = self.labels[index]

        donnees_tabulaires = torch.tensor(self.tabular_data[index], dtype=torch.float32)
        img_rgb_path, img_nir_path = self.__get_image_paths(obs_id)

        try:
            image_rgb = Image.open(img_rgb_path).convert("RGB")
            image_nir = Image.open(img_nir_path).convert("L")

            tensor_rgb = self.to_tensor(image_rgb)
            tensor_nir = self.to_tensor(image_nir)

            tensor_rgb = self.norm_rgb(tensor_rgb)
            tensor_nir = self.norm_nir(tensor_nir)

            image_bimodale = torch.cat((tensor_rgb, tensor_nir), dim=0)

        except (FileNotFoundError, OSError):
            image_bimodale = torch.zeros((4, 256, 256))

        if self.transform:
            image_bimodale = self.transform(image_bimodale)

        return image_bimodale, donnees_tabulaires, label

class TransformWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        x_img, x_tab, y = self.subset[index]
        if self.transform:
            x_img = self.transform(x_img)
        return x_img, x_tab, y
