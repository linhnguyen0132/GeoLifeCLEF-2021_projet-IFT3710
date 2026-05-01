# Ce script exécute les fonctions implémentées pour l'entraînement #

# --- Import des librairies ---
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.models as models
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import GLC_DATASET
from k_cnn import K_CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" --- Utilisation : {device} --- ")


def afficher_telemetrie(train_loss, val_loss, train_acc, val_acc):
    # Crée l'axe des X (1, 2, 3, 4, 5...)
    epochs = range(1, len(train_loss) + 1)
    # Crée une grande figure
    plt.figure(figsize=(14, 5))

    # --- Graphique 1 : La Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # --- Graphique 2 : L'Accuracy ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'g-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_acc, 'g-', label='Validation Accuracy', linewidth=2)
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # --- Affichage ---
    plt.tight_layout()
    # --- 2021 ---
    #chemin_image = "img/courbes_entrainement.png"

    # --- 2022 ---
    chemin_image = "img/courbes_entrainement2.png"
    plt.savefig(chemin_image)
    print(f"Graphique sauvegardé avec succès dans : {chemin_image}")


# --- Transformations ---

transformations_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transformations_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456,0.406 ], std=[0.229, 0.224, 0.225])

])




# --- Chemin des données ---

# --- GeoLifeClef 2021 ---
csv_2021 = "/home/abdkarimouatt/projects/def-sponsor00/geolifeclef/data/observations/observations_fr_train.csv"
images_2021 = "/home/abdkarimouatt/projects/def-sponsor00/geolifeclef/data/patches_sample/fr"

# --- GeoLifeClef 2022 --- 
csv_2022 = "/home/abdkarimouatt/projects/def-sponsor00/geolifeclef/data_2022/observations/observations_fr_train.csv"
images_2022 = "/home/abdkarimouatt/projects/def-sponsor00/geolifeclef/data_2022/patches-fr"
# --- Transformations ---
t_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

t_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Ce qu'on va utiliser ---
#CHEMIN_CSV = csv_2022
#CHEMIN_IMAGES = images_2022
train_21 = GLC_DATASET(csv_2021, images_2021, t_train)
train_22 = GLC_DATASET(csv_2022, images_2022, t_train)

val_21 = GLC_DATASET(csv_2021, images_2021, t_val)
val_22 = GLC_DATASET(csv_2022, images_2022, t_val)

train_dataset = ConcatDataset([train_21, train_22])
val_dataset = ConcatDataset([val_21, val_22])

print("Chargement du dataset complet")
#dataset_train = GLC_DATASET(csv_file=CHEMIN_CSV, root_dir=CHEMIN_IMAGES, transform=transformations_train)
#dataset_val = GLC_DATASET( csv_file=CHEMIN_CSV, root_dir=CHEMIN_IMAGES, transform=transformations_val)

#indices_test = list(range(1000))
#dataset_reduit = Subset(dataset_complet, indices_test)


# --- Split Train/Val (80/20)
#taille_totale = len(dataset_train)
#taille_train = int(0.8 * taille_totale)

#indices_melanges = torch.randperm(taille_totale).tolist()

#indices_train = indices_melanges[:taille_train]
#indices_val = indices_melanges[taille_train:]

#train_dataset = Subset(dataset_train, indices_train)
#val_dataset = Subset(dataset_val, indices_val)

#print(f"Images pour l'entraînement : {len(train_dataset)}")
#print(f"Images pour la validation : {len(val_dataset)}")

# --- Instancier le dataloader pour l'entraînement et la validation ---
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(dataset= val_dataset, batch_size=32, shuffle=False , num_workers=8, pin_memory=True)

# --- Nombre de classes ---
#nombre_classes_fr = dataset_train.annotations['species_id'].max() + 1
#print(f"{nombre_classes_fr} classes possibles")

# --- Instancie le model CNN ---
#model = K_CNN(num_classes=nombre_classes_fr).to(device)

# --- TRANSFERT LEARNING RESNET50 ---
#model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#num_ftrs = model.fc.in_features
#model.fc = torch.nn.Linear(num_ftrs, nombre_classes)
#model = model.to(device)

# --- Instancie la fonction de perte : on choisit Entropie croisée ---
print("Chargement des poids de classes...")
#poids_tensor = torch.load('poids_classes_geolifeclef.pt').cuda()
#perte = nn.CrossEntropyLoss(weight=poids_tensor)
y_train = np.concatenate([train_21.annotations['species_id'].values, train_22.annotations['species_id'].values])
classes_uniques = np.unique(y_train)
nombre_classes = int(y_train.max()+1)
print(f"{nombre_classes} classes possibles")
poids_bruts = compute_class_weight('balanced', classes=classes_uniques, y=y_train)
poids_ajustes = np.clip(poids_bruts, a_min=0, a_max=50.0)


poids_tensor = torch.ones(nombre_classes).to(device)
for idx, cls in enumerate(classes_uniques):
    poids_tensor[cls] = poids_ajustes[idx]
#poids_tensor = torch.FloatTensor(poids_finaux).cuda()    
perte = torch.nn.CrossEntropyLoss(weight=poids_tensor)

# --- TRANSFERT LEARNING RESNET50 ---
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, nombre_classes)
model = model.to(device)

# --- Instancie l'optimiseur : on choisit Stochastic Gradient Descent ---
# Nouveauté : Adam a remplacé SGD et on sépare lr des couches dans l'optimiseur
learning_rate = 1e-4
parametres_backbone = [params for nom, params in model.named_parameters() if "fc" not in nom]
parametres_head = model.fc.parameters()
optimiseur = optim.Adam([{'params': parametres_backbone, 'lr':1e-5} ,{'params': parametres_head, 'lr':1e-3}] , weight_decay=1e-4)
# --- Instancie le Scheduler
scheduler = ReduceLROnPlateau(optimiseur, mode='min', factor=0.1, patience=3, verbose=True)
# --- Instancie le nombre d'époques pour déterminer le nombre d'itérations d'entraînement du model ---
num_epochs = 30

    
historique= {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
meilleure_accuracy = 0.0  # Initialisation
print(" --- DÉBUT DE L'ENTRAÎNEMENT DU MODÈLE ---")

for epoch in range(num_epochs):
    model.train()
    loss_totale_train = 0.0
    correctes_top1_train = 0
    correctes_top30_train = 0
    total_images_train = 0

    boucle_train = tqdm(train_loader, desc=f"Train Epoch [{epoch+1}/{num_epochs}]", leave=False, miniters=100)
    for batch_idx, (images, labels) in enumerate(boucle_train):
        
        images, labels = images.to(device), labels.to(device)
        
        # 1 - Mettre les gradients de l'optimiseur à zéro avant l'entraînement
        optimiseur.zero_grad()

        # 2 - Forward pass
        outputs = model(images)

        # 3 - Calculer la perte
        loss = perte(outputs, labels)

        # 4 - Backward pass pour calculer les nouveaux gradients
        loss.backward()

        # 5 - MAJ des poids
        optimiseur.step()

        #print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))
        loss_totale_train += loss.item()
        total_images_train += labels.size(0)
        labels_reshaped = labels.view(-1,1)

        # Top-1
        _, preds_top1 = outputs.topk(1, dim=1)
        correctes_top1_train += (preds_top1 == labels_reshaped).sum().item()

        # Top-30
        _, preds_top30 = outputs.topk(30, dim=1)
        correctes_top30_train += (preds_top30 == labels_reshaped).sum().item()
        boucle_train.set_postfix(loss = loss.item())
    
    #print(" --- DÉBUT DE LA VALIDATION --- ")
    model.eval()
    loss_totale_val = 0.0
    correctes_top1_val = 0
    correctes_top30_val = 0
    total_images_val = 0

    boucle_val = tqdm(val_loader, desc=f"Val Epoch   [{epoch+1}/{num_epochs}]", leave=False, miniters=100)

    with torch.no_grad():
        for images, labels in boucle_val:
            
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = perte(outputs, labels)

            loss_totale_val += loss.item() 
            total_images_val += labels.size(0)

            labels_reshaped = labels.view(-1,1)
            
            # Top-1
            _, preds_top1 = outputs.topk(1, dim=1)
            correctes_top1_val += (preds_top1 == labels_reshaped).sum().item()

            # Top-30
            _, preds_top30 = outputs.topk(30, dim=1)
            correctes_top30_val += (preds_top30 == labels_reshaped).sum().item()
            boucle_val.set_postfix(loss=loss.item())

    moyenne_loss_train = loss_totale_train / len(train_loader)
    accuracy_top1_train = correctes_top1_train / total_images_train
    accuracy_top30_train = correctes_top30_train / total_images_train

    moyenne_loss_val = loss_totale_val / len(val_loader)
    accuracy_top1_val = correctes_top1_val / total_images_val
    accuracy_top30_val = correctes_top30_val / total_images_val

    historique['train_loss'].append(moyenne_loss_train)
    historique['train_acc'].append(accuracy_top30_train)
    historique['val_loss'].append(moyenne_loss_val)
    historique['val_acc'].append(accuracy_top30_val)

    print(f"Epoch [{epoch + 1}/{num_epochs}] | "
          f"TRAIN Loss: {moyenne_loss_train:.4f} | Top-1: {accuracy_top1_train:.4f} | Top-30: {accuracy_top30_train:.4f} || "
          f"VAL Loss: {moyenne_loss_val:.4f} | Top-1: {accuracy_top1_val:.4f} | Top-30: {accuracy_top30_val:.4f}")

    # À la fin de chaque époque, dans ta boucle principale
    if accuracy_top1_val > meilleure_accuracy:
        meilleure_accuracy = accuracy_top1_val
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiseur.state_dict(),
            'loss': moyenne_loss_val,
        }, "meilleur_modele_geolife.pth")
        print("--- Nouveau record ! Modèle sauvegardé ---")


    scheduler.step(moyenne_loss_val)

# --- APPEL DE LA FONCTION À LA TOUTE FIN DU SCRIPT ---
print("--- FIN DE L'ENTRAÎNEMENT DU MODÈLE ---")
afficher_telemetrie(historique['train_loss'], historique['val_loss'], historique['train_acc'], historique['val_acc'])





