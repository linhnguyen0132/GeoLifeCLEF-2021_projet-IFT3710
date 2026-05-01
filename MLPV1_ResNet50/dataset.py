# Ce script permet de gérer le dataset#
# Import des librairies utiles : Pytorch
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class GLC_DATASET(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file : Le chemin vers observations_us_train.csv
        root_dir : Le chemin vers le patches_us
        """
        self.annotations = pd.read_csv(csv_file, sep=";")
        self.root_dir = root_dir

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __get_image_path(self, obs_id):
        obs_str = str(obs_id)
        dossier_1 = obs_str[-2:]
        dossier_2 = obs_str[-4:-2]

        nom_fichier = f"{obs_str}_rgb.jpg"
        chemin_complet = os.path.join(self.root_dir, dossier_1, dossier_2, nom_fichier)
        return chemin_complet
    
    def __getitem__(self, index):
        obs_id = self.annotations.iloc[index]['observation_id']
        label = int(self.annotations.iloc[index]['species_id'])

        img_path = self.__get_image_path(obs_id)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color='black')
        if self.transform:
            image = self.transform(image)
        return image, label
    
# --- TEST DU CHARGEUR ---
if __name__ == "__main__":
    # chemins
    CHEMIN_CSV = "C:/Users/abdou/Downloads/GLC_DATASET/2021/observations/observations_fr_train.csv"
    CHEMIN_IMAGES = "c:/Users/abdou/Downloads/GLC_DATASET/2021/patches_sample"
    
    mon_vrai_dataset = GLC_DATASET(csv_file=CHEMIN_CSV, root_dir=CHEMIN_IMAGES)
    
    print(f"Succès ! Le dataset contient {len(mon_vrai_dataset)} images.")
    
    # On extrait la première image
    premiere_image, premier_label = mon_vrai_dataset[0]
    print(f"Forme du tenseur image : {premiere_image.shape}")
    print(f"ID de l'espèce (Label) : {premier_label}")