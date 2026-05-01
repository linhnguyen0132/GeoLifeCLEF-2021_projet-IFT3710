import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from tqdm import tqdm
import pandas as pd

from v2_dataset import GLC_DATASET, TransformWrapper

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
#Différents path
BASE_DIR = os.environ.get("SLURM_TMPDIR", "/home/abdkarimouatt/projects/def-sponsor00/geolifeclef")
CSV_OBS_PATH = f"{BASE_DIR}/data_2022/observations/observations_fr_train.csv"
CSV_ENV_PATH = f"{BASE_DIR}/data_2022/pre-extracted/environmental_vectors.csv"
IMG_DIR      = f"{BASE_DIR}/data_2022/patches-fr"

BATCH_SIZE = 128
NUM_WORKERS = 8
EPOCHS = 30
LR_BACKBONE = 1e-5
LR_HEAD = 5e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f" --- Utilisation : {device} --- ")

# ---------------------------------------------------------------------------
# 2. TRANSFORMATIONS
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
# 3. CHARGEMENT DES DONNÉES
# ---------------------------------------------------------------------------
dataset_complet = GLC_DATASET(csv_obs=CSV_OBS_PATH, csv_env=CSV_ENV_PATH, root_dir=IMG_DIR)

taille_train = int(0.8 * len(dataset_complet))
taille_val   = len(dataset_complet) - taille_train
split_train, split_val = random_split(dataset_complet, [taille_train, taille_val], generator=torch.Generator().manual_seed(42))

train_dataset = TransformWrapper(split_train, transform=train_transform)
val_dataset   = TransformWrapper(split_val, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

df = pd.read_csv(CSV_OBS_PATH, sep=";")
NUM_CLASSES = int(df['species_id'].max() + 1)

# ---------------------------------------------------------------------------
# 4. ARCHITECTURE (ResNet50 + MLP)
# ---------------------------------------------------------------------------
class ModeleGagnant(nn.Module):
    def __init__(self, num_classes, num_tabular_features=29):
        super().__init__()

        #BackBone
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        ancienne_conv1 = self.cnn.conv1
        self.cnn.conv1 = nn.Conv2d(4, ancienne_conv1.out_channels, kernel_size=ancienne_conv1.kernel_size, stride=ancienne_conv1.stride, padding=ancienne_conv1.padding, bias=False)

        with torch.no_grad():
            self.cnn.conv1.weight[:, :3, :, :] = ancienne_conv1.weight.data
            nn.init.constant_(self.cnn.conv1.weight[:, 3:4, :, :], 0.0)

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

        #Head
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_out_features + 128, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, images, tabular):
        # On passe les images dans CNN
        x_vision = self.cnn(images)
        # On passe les données tabulaires dans MLP
        x_tabulaire = self.mlp(tabular)
        # On fusionne les résultats des 2
        x_fusion = torch.cat((x_vision, x_tabulaire), dim=1)
        # On passe à la classification
        return self.classifier(x_fusion)

print(f"Construction du modèle Hybride Sensio (Vision + {29} vars tabulaires)...")
model = ModeleGagnant(num_classes=NUM_CLASSES, num_tabular_features=29).to(device)

# ---------------------------------------------------------------------------
# 5. OPTIMISATION (Fine Tunning)
# ---------------------------------------------------------------------------
# Reprise des anciens poids si il y en a 
chemin_sauvegarde = '/home/abdkarimouatt/scratch/k_cnn/meilleur_modele_sensio_29vars.pth'
if os.path.exists(chemin_sauvegarde):
    print(f"Chargement de {chemin_sauvegarde}")
    checkpoint = torch.load(chemin_sauvegarde, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Modèle précédent chargé.Reprise de l'entraînement !")
else:
    print("Aucun ancien modèle trouvé. On commence l'entraînement de zéro.")


# Label smooting : Meilleure généralisation : Réduit le surapprentissage en empêchant le modèle de devenir trop confiant
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW([
    {'params': model.cnn.parameters(), 'lr': LR_BACKBONE},
    {'params': model.mlp.parameters(), 'lr': LR_HEAD},
    {'params': model.classifier.parameters(), 'lr': LR_HEAD}
], weight_decay=1e-4)
# Scheduler permet d'ajuster le learning rate si on a un plateau ou une non amélioration
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
# can use the torch.cuda.amp.GradScaler in PyTorch to implement automatic Gradient Scaling for writing compute efficient training loops.
scaler = torch.amp.GradScaler('cuda')

# ---------------------------------------------------------------------------
# 6. BOUCLE D'ENTRAÎNEMENT et Early stopping
# ---------------------------------------------------------------------------
meilleure_val_top30 = 0.0
patience_early_stopping = 10
epochs_sans_amelioration = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    pbar_train = tqdm(train_loader, desc=f"Train [{epoch+1}/{EPOCHS}]", leave=False)

    for images, tabular, labels in pbar_train:
        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            outputs = model(images, tabular)
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

    lr_actuel = optimizer.param_groups[-1]['lr']

    print(f"Epoch [{epoch+1}/{EPOCHS}] | TRAIN Loss: {running_loss/len(train_loader):.4f} | VAL Loss: {val_loss_epoch:.4f} Top-30: {acc_top30_val:.4f} | LR: {lr_actuel:.1e}")

    scheduler.step(acc_top30_val)

    #Early stopping si pas d'amélioration du top-30
    if acc_top30_val > meilleure_val_top30:
        meilleure_val_top30 = acc_top30_val
        epochs_sans_amelioration = 0
        torch.save({'model_state_dict': model.state_dict()}, 'meilleur_modele_sensio_29vars.pth')
        print(f"Amélioration Val Top-30 : {acc_top30_val:.4f} — Modèle sauvegardé")
    
    else:
        epochs_sans_amelioration += 1
        print(f"Aucune amélioration {epochs_sans_amelioration}")

        if epochs_sans_amelioration >= patience_early_stopping :
            print(f"Early stopping : Fin d'entraînement !")
            break
