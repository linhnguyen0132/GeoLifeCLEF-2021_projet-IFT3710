#Ce script implémente un convolutional neuron network CNN je me suis inspiré de : https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html et https://www.digitalocean.com/community/tutorials/writing-cnns-from-scratch-in-pytorch

#Import des librairies
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from torchvision import dat
import torch.nn.functional as F

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Utilisation : {device}")

class K_CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # --- 1ère partie BackBone ---
        # On va répéter 3 opérations d'affilée : convolution - ReLU - Pooling Max
        # Nouveautés : Pour améliorer le modèle : batch normalization aprés la convolution
        self.backbone = nn.Sequential(

            # taille (224 224)
            # - 1 fois
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (16 112 112)

            # - 2 fois 
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (32 56 56)

            # - 3 fois
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (64 28 28)

            # - 4 fois
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (128 14 14)

            # - 5 fois
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # (256 7 7)
        )
        
        # Global pooling
        self.pool_global = nn.AdaptiveAvgPool2d((1,1))   
        # (256 1 1)

        # --- 2ème partie Head ---
        #On va implémenter la tête de classification 
        self.head = nn.Sequential(
            nn.Linear(in_features=256,out_features= num_classes,bias=True)
        )

    def forward(self, x):
        # - 1 : l'image passe dans le backbone
        x = self.backbone(x)
        # - 2 : golbal pooling
        x = self.pool_global(x)
        # - 3 : l'image est applatie
        x = torch.flatten(x,1)
        # - 4 : l'image passe dans head pour la classification
        x = self.head(x)
        return x


K_CNN_model = K_CNN(num_classes=100).to(device=device)

print(K_CNN_model)
