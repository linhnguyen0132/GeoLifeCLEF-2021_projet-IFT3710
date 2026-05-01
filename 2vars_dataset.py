import pandas as pd
import torch
import math
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

class GLC_DATASET(Dataset):
    def __init__(self, csv_file, root_dir, subset_filter=None,transform=None):
        df = pd.read_csv(csv_file, sep=";")
        # Filtre selon la colonne 'subset' du CSV (train ou val)
        if subset_filter:
            self.annotations = df[df['subset'] == subset_filter].copy().reset_index(drop=True)
        else:
            self.annotations = df
            
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def _get_image_paths(self, obs_id):
        obs_str = str(obs_id)
        dossier_1 = obs_str[-2:]
        dossier_2 = obs_str[-4:-2]
        
        chemin_rgb = self.root_dir / dossier_1 / dossier_2 / f"{obs_str}_rgb.jpg"
        chemin_nir = self.root_dir / dossier_1 / dossier_2 / f"{obs_str}_near_ir.jpg"
        return chemin_rgb, chemin_nir

    def __getitem__(self, index):
        # 1. Extraction des identifiants
        obs_id = self.annotations.iloc[index]['observation_id']
        label  = int(self.annotations.iloc[index]['species_id'])
        
        # On va chercher les coordonnées GPS dans le CSV
        lat = float(self.annotations.iloc[index]['latitude'])
        lon = float(self.annotations.iloc[index]['longitude'])
        
        # On regroupe ces valeurs dans un tenseur (pour le MLP)
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        donnees_tabulaires = torch.tensor([
            math.sin(lat_rad),
            math.cos(lat_rad),
            math.sin(lon_rad),
            math.cos(lon_rad),
            ], dtype=torch.float32)
        # ---------------------------------------------------------

        img_rgb_path, img_nir_path = self._get_image_paths(obs_id)

        try:
            # 2. Charger RGB et NIR
            image_rgb = Image.open(img_rgb_path).convert("RGB")
            image_nir = Image.open(img_nir_path).convert("L")

            # 3. Convertir en tenseurs
            tensor_rgb = transforms.ToTensor()(image_rgb)
            tensor_nir = transforms.ToTensor()(image_nir)

            # 4. Normaliser
            tensor_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_rgb)
            tensor_nir = transforms.Normalize(mean=[0.5], std=[0.5])(tensor_nir)

            # 5. FUSIONNER (Concaténer sur la dimension des canaux) -> Devient [4, H, W]
            image_bimodale = torch.cat((tensor_rgb, tensor_nir), dim=0)

        except (FileNotFoundError, OSError):
            # Tenseur noir par défaut (4 canaux) en cas d'image corrompue
            image_bimodale = torch.zeros((4, 256, 256))

        # Application des transformations si passées directement
        if self.transform:
            image_bimodale = self.transform(image_bimodale)

        # On retourne maintenant 3 éléments pour notre modèle à deux têtes !
        return image_bimodale, donnees_tabulaires, label

# Wrapper pour appliquer les transforms après le random_split (utilisé dans main.py)
class TransformWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        # NOUVEAU : On dépaquette bien les 3 éléments (Image, Tabulaire, Label)
        x_img, x_tab, y = self.subset[index]
        
        if self.transform:
            # Les transformations ne s'appliquent QUE sur l'image (pas sur le GPS !)
            x_img = self.transform(x_img)
            
        # On renvoie le tout bien formaté
        return x_img, x_tab, y