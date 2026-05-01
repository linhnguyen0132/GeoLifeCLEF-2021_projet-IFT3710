import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR

from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from v2_dataset import GLC_DATASET, TransformWrapper

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = os.environ.get("SLURM_TMPDIR", "/home/abdkarimouatt/projects/def-sponsor00/geolifeclef")
CSV_PATH = f"{BASE_DIR}/data_2022/observations/observations_fr_train.csv"
IMG_DIR  = f"{BASE_DIR}/data_2022/patches-fr"

BATCH_SIZE = 64
NUM_WORKERS = 10
EPOCHS = 30
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 2. TRANSFORMATIONS (Inclus des Rotations comme l'équipe gagnante)
# ---------------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256), antialias=True),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256), antialias=True),
    transforms.CenterCrop(224),
])


# ---------------------------------------------------------------------------
# 3. CHARGEMENT DES DONNÉES (CORRIGÉ POUR LES SUBSETS)
# ---------------------------------------------------------------------------
# On charge deux fois le dataset, une fois filtré sur 'train', une fois sur 'val'
dataset_train_brut = GLC_DATASET(csv_file=CSV_PATH, root_dir=IMG_DIR, subset_filter="train")
dataset_val_brut   = GLC_DATASET(csv_file=CSV_PATH, root_dir=IMG_DIR, subset_filter="val")

print(f"Nombre d'images Train : {len(dataset_train_brut)}")
print(f"Nombre d'images Val   : {len(dataset_val_brut)}")

# On applique les transformations (Data Augmentation pour le train, Normal pour le val)
train_dataset = TransformWrapper(dataset_train_brut, transform=train_transform)
val_dataset   = TransformWrapper(dataset_val_brut, transform=val_transform)

# Création des loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

# Calcul du nombre de classes sur le fichier complet pour ne rien oublier
df_all = pd.read_csv(CSV_PATH, sep=";")
NUM_CLASSES = int(df_all['species_id'].max() + 1)


# ---------------------------------------------------------------------------
# 4. L'ARCHITECTURE (ResNet + MLP Tabulaire)
# ---------------------------------------------------------------------------
class ModeleGagnant(nn.Module):
    def __init__(self, num_classes, num_tabular_features=2):
        super().__init__()
        
        # (ResNet50 4 Canaux)
        # Transfert learning ResNet50
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        #On récupère la première couche convolutionnelle
        ancienne_conv1 = self.cnn.conv1
        #On modofie l'entrée pour avoir 4 canaux
        self.cnn.conv1 = nn.Conv2d(4, ancienne_conv1.out_channels, kernel_size=ancienne_conv1.kernel_size, stride=ancienne_conv1.stride, padding=ancienne_conv1.padding, bias=False)
        self.bn_vision = nn.BatchNorm1d(512)
        self.bn_tab = nn.BatchNorm1d(512)

        with torch.no_grad():
            # On copie les poids pré-entrainés sur ImageNet pour RGB
            self.cnn.conv1.weight[:, :3, :, :] = ancienne_conv1.weight.data
            nn.init.constant_(self.cnn.conv1.weight[:, 3:4, :, :], 0.0) # Zero-Init pour mettre NIR à 0
            
        self.cnn_out_features = self.cnn.fc.in_features # 2048 caractéristiques
        self.cnn.fc = nn.Identity() # On arrache la tête de classification standard

        self.cnn_proj = nn.Linear(self.cnn_out_features, 512)
        self.mlp_proj = nn.Linear(128, 512)


        # (MLP 2 couches Linear+ReLu+BatchNorm+DropOut)
        self.mlp = nn.Sequential(
            #1ère couche
            #On transforme nos features en vecteur de taille 64
            nn.Linear(num_tabular_features, 64),
            # Normalisation
            nn.BatchNorm1d(64),
            #Non linearité
            nn.ReLU(),
            # Régularisation pour réduire le surapprentissage
            nn.Dropout(0.2),

            #2ème couche
            #On transforme nos features en vecteur de taille 64
            nn.Linear(64, 128),
            # Normalisation
            nn.BatchNorm1d(128),
            #Non linearité
            nn.ReLU(),
            # Régularisation pour réduire le surapprentissage
            nn.Dropout(0.2)
        )
        
        
        # LA FUSION : Concaténation 
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, tabular):
        x_vision = self.cnn(images)      # Vecteur de taille [Batch, 2048]
        x_tabulaire = self.mlp(tabular)  # Vecteur de taille [Batch, 128]
        # Équilibrage de chaque cerveau
        x_vision = self.bn_vision(self.cnn_proj(x_vision)) # [Batch, 512]
        x_tabulaire = self.bn_tab(self.mlp_proj(x_tabulaire)) # [Batch, 512]
        x_fusion = torch.cat((x_vision, x_tabulaire), dim=1) # [Batch, 1024]
        return self.classifier(x_fusion)

print("Construction du modèle ...")
model = ModeleGagnant(num_classes=NUM_CLASSES, num_tabular_features=4).to(device)
#model = torch.compile(model)

# Geler tout le backbone
for param in model.cnn.parameters():
    param.requires_grad = False

# Dégeler layer3 ET layer4
for param in model.cnn.layer3.parameters():
    param.requires_grad = True
for param in model.cnn.layer4.parameters():
    param.requires_grad = True

# ---------------------------------------------------------------------------
# 5. OPTIMISATION
# ---------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# On applique le LR_BACKBONE au ResNet, et le LR_HEAD au MLP et au Classifieur
optimizer = optim.AdamW([
    {'params': model.cnn.parameters(), 'lr': LR_BACKBONE, 'weight_decay': 1e-5},

    {'params': model.mlp.parameters(), 'lr': LR_HEAD},
    {'params': model.mlp_proj.parameters(), 'lr': LR_HEAD},

    {'params': model.cnn_proj.parameters(), 'lr': LR_HEAD},

    {'params': model.classifier.parameters(), 'lr': LR_HEAD}
], weight_decay=1e-4)

#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler('cuda')

# ---------------------------------------------------------------------------
# 6. BOUCLE D'ENTRAÎNEMENT
# ---------------------------------------------------------------------------

meilleure_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    pbar_train = tqdm(train_loader, desc=f"Train [{epoch+1}/{EPOCHS}]", leave=False)
    
    # La boucle for extrait maintenant 3 éléments
    for images, tabular, labels in pbar_train:
        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)
        
        if torch.isnan(images).any():
            print("NaN dans les images !"); break
        if torch.isnan(tabular).any():
            print("NaN dans le GPS !"); break

        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            # Le modèle prend deux entrées : images et données GPS
            outputs = model(images, tabular)
            if torch.isnan(outputs).any():
                print("NaN dans les outputs !"); break
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar_train.set_postfix(loss=loss.item())

    # --- VALIDATION ---
    model.eval()
    val_loss, correct_top1, correct_top30, total = 0.0, 0, 0, 0
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for images, tabular, labels in val_loader:
                images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            
                outputs = model(images, tabular) 
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            
                _, pred_top30 = outputs.topk(30, 1, True, True)
                pred_top30 = pred_top30.t()
                correct = pred_top30.eq(labels.view(1, -1).expand_as(pred_top30))
            
                correct_top1 += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
                correct_top30 += correct[:30].reshape(-1).float().sum(0, keepdim=True).item()
                total += labels.size(0)

    val_loss_epoch = val_loss / len(val_loader)
    acc_top1_val = correct_top1 / total
    acc_top30_val = correct_top30 / total
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | TRAIN Loss: {running_loss/len(train_loader):.4f} | VAL Loss: {val_loss_epoch:.4f} Top-30: {acc_top30_val:.4f}")

    scheduler.step()

    if acc_top30_val > meilleure_val_acc:
        meilleure_val_acc = acc_top30_val
        torch.save({'model_state_dict': model.state_dict()}, 'meilleur_modele_sensio.pth')