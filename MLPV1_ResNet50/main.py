# Ce script exécute les fonctions implémentées pour l'entraînement #

# --- Import des librairies ---
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, random_split
from dataset import GLC_DATASET
from k_cnn import K_CNN



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
    plt.show()

# --- Chemin des données ---
CHEMIN_CSV = "C:/Users/abdou/Downloads/GLC_DATASET/2021/observations/observations_fr_train.csv"
CHEMIN_IMAGES = CHEMIN_IMAGES = "c:/Users/abdou/Downloads/GLC_DATASET/2021/patches_sample"

print("Chargement du dataset complet")
dataset_complet = GLC_DATASET(csv_file=CHEMIN_CSV, root_dir=CHEMIN_IMAGES)

indices_test = list(range(1000))
dataset_reduit = Subset(dataset_complet, indices_test)


# --- Split Train/Val (80/20)
taille_totale = len(dataset_reduit)
taille_train = int(0.8 * taille_totale)
taille_val = taille_totale - taille_train
train_dataset, val_dataset = random_split(dataset_reduit, [taille_train, taille_val])
print(f"Images pour l'entraînement : {len(train_dataset)}")
print(f"Images pour la validation : {len(val_dataset)}")

# --- Instancier le dataloader pour l'entraînement et la validation ---
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset= val_dataset, batch_size=16, shuffle=False )

# --- Nombre de classes ---
nombre_classes_fr = dataset_complet.annotations['species_id'].max() + 1
print(f"{nombre_classes_fr} classes possibles")

# --- Instancie le model CNN ---
model = K_CNN(num_classes=nombre_classes_fr)
# --- Instancie la fonction de perte : on choisit Entropie croisée ---
perte = nn.CrossEntropyLoss()
# --- Instancie l'optimiseur : on choisit Stochastic Gradient Descent ---
learning_rate = 1e-3
optimiseur = torch.optim.SGD(params=model.parameters(), lr = learning_rate, weight_decay=0.05, momentum=0.9)
# --- Instancie le nombre d'époques pour déterminer le nombre d'itérations d'entraînement du model ---
num_epochs = 5

    
historique= {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

print(" --- DÉBUT DE L'ENTRAÎNEMENT DU MODÈLE ---")

for epoch in range(num_epochs):
    model.train()
    loss_totale_train = 0.0
    predictions_correctes_train = 0
    total_images_train = 0

    boucle_train = tqdm(train_loader, desc=f"Train Epoch [{epoch+1}/{num_epochs}]", leave=False)
    for batch_idx, (images, labels) in enumerate(train_loader):
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
        valeur_max, predictions = torch.max(outputs, 1)
        predictions_correctes_train += (predictions == labels).sum().item()
        total_images_train += labels.size(0)

        boucle_train.set_postfix(loss = loss.item())
    
    #print(" --- DÉBUT DE LA VALIDATION --- ")
    model.eval()
    loss_totale_val = 0.0
    predictions_correctes_val = 0
    total_images_val = 0

    boucle_val = tqdm(val_loader, desc=f"Val Epoch   [{epoch+1}/{num_epochs}]", leave=False)

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = perte(outputs, labels)

            loss_totale_val += loss.item()
            valeur_max, predictions = torch.max(outputs, 1)
            predictions_correctes_val += (predictions == labels).sum().item()
            total_images_val += labels.size(0)

    moyenne_loss_train = loss_totale_train / len(train_loader)
    accuracy_train = predictions_correctes_train / total_images_train

    moyenne_loss_val = loss_totale_val / len(val_loader)
    accuracy_val = predictions_correctes_val / total_images_val

    historique['train_loss'].append(moyenne_loss_train)
    historique['train_acc'].append(accuracy_train)
    historique['val_loss'].append(moyenne_loss_val)
    historique['val_acc'].append(accuracy_val)

    print(f"Epoch [{epoch + 1}/{num_epochs}] |" 
          f"Train Loss : {moyenne_loss_train:.4f}, Accuracy : {accuracy_train:.4f} |"
          f"Val Loss : {moyenne_loss_val:.4f}, Accuracy: {accuracy_val:.4f}")


# -- APPEL DE LA FONCTION À LA TOUTE FIN DU SCRIPT ===
print("--- FIN DE L'ENTRAÎNEMENT DU MODÈLE ---")
afficher_telemetrie(historique['train_loss'], historique['val_loss'], historique['train_acc'], historique['val_acc'])





