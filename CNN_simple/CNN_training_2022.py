from CNN import ConvBlock,CNN

from  obs_and_patches_2022 import GeoLifeDataset2

import torch
import torch.nn as nn
import torchvision as torchvision

import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np

# Change this path to adapt to where you downloaded the data
BASE_PATH = Path("../..")
DATA_PATH = BASE_PATH / "/home/dani/projects/def-sponsor00/geolifeclef/data_2022"

# Create the path to save submission files
SUBMISSION_PATH = Path("submissions")
os.makedirs(SUBMISSION_PATH, exist_ok=True)

# Hyper paramètres
batch_size = 32
epochs = 15
learning_rate = 0.00005
decay = 0.0001
# create device based on GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

from torchvision import transforms
transform = transforms.Compose([
    #transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomVerticalFlip(p = 0.5),
    transforms.RandomRotation(degrees = (6,12)),
    #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    #transforms.ToTensor(),
    #transforms.Normalize(
    #    mean=[0.485, 0.456, 0.406],
    #    std=[0.229, 0.224, 0.225]
    #)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406, 0.5],
        std=[0.229, 0.224, 0.225, 0.5]
    )
])


# Seperate Dataset
train_dataset = GeoLifeDataset2(
    DATA_PATH,
    subset = "train",
    transform=transform
)

val_dataset = GeoLifeDataset2(
    DATA_PATH,
    subset="val",
    transform=transform,
    label_map=train_dataset.label_map
)

# Calculer les poids des classes pour attribuer une plus grande importance à la classe minoritaire

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

labels = np.array(train_dataset.labels)
num_classes = len(train_dataset.label_map)
class_count = np.bincount(labels, minlength=num_classes)
class_weights = 1.0 / (class_count + 1)
class_weights = class_weights / class_weights.sum() * len(class_weights)
class_weights = torch.tensor( class_weights, dtype=torch.float32 ).to(device)

# Initialize loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)

num_classes = len(train_dataset.label_map)

# Create CNN model
model = CNN(classes=num_classes)
model = model.to(device)
print(model)

# Create optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), weight_decay=decay,lr=learning_rate)  # weight_decay=decay
loss = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing = 0.1)

# store train/val error & accuracy
errors = {"train": [],"val": [],}
top30_errors = {"train": [], "val": []}
accuracies = {"train": [], "val": []}


print("Train dataset size:", len(train_dataset))
print("Train loader batches:", len(train_loader))

# Fonction de validation
def validate(model, val_loader, loss_fn, device):

    model.eval()

    val_loss = 0
    correct_top1 = 0
    correct_top30 = 0
    total = 0

    with torch.no_grad():

        for images,labels, coords in val_loader:

            images = images.to(device)
            coords = coords.to(device)
            labels = labels.to(device)
        
            outputs = model(images, coords)

            loss_value = loss_fn(outputs, labels)
            val_loss += loss_value.item()

            # Top-1 prediction
            _, predicted = torch.max(outputs, 1)
            correct_top1 += (predicted == labels).sum().item()

            # Top-30 predictions
            _, top30 = torch.topk(outputs, 30, dim=1)
            correct_top30 += (top30 == labels.unsqueeze(1)).sum().item()
            #print(correct_top30)
            total += labels.size(0)

    val_loss = val_loss / len(val_loader)

    val_accuracy = correct_top1 / total
    val_top30_accuracy = correct_top30 / total
    val_top30_error = 1 - val_top30_accuracy

    return val_loss, val_accuracy, val_top30_error

# Training
for epoch in range(epochs):

    model.train()

    running_loss = 0
    correct_top1 = 0
    correct_top30 = 0
    total = 0

    for images, labels, coords in tqdm(train_loader):

        images = images.to(device)
        coords = coords.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images, coords)

        loss_value = loss(outputs, labels)
        loss_value.backward()

        optimizer.step()

        running_loss += loss_value.item()

        # Top-1
        _, predicted = torch.max(outputs, 1)
        correct_top1 += (predicted == labels).sum().item()

        # Top-30
        _, top30 = torch.topk(outputs, 30, dim=1)
        correct_top30 += (top30 == labels.unsqueeze(1)).sum().item()
        #print(correct_top30)

        total += labels.size(0)

    train_loss = running_loss / len(train_loader)

    train_accuracy = correct_top1 / total
    train_top30_error = 1 - (correct_top30 / total)

    val_loss, val_accuracy, val_top30_error = validate(model, val_loader, loss, device)

    # sauvegarde des métriques
    errors["train"].append(train_loss)
    errors["val"].append(val_loss)

    accuracies["train"].append(train_accuracy)
    accuracies["val"].append(val_accuracy)

    top30_errors["train"].append(train_top30_error)
    top30_errors["val"].append(val_top30_error)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Train Top-30 Error: {train_top30_error:.4f}")

    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_accuracy:.4f}")
    print(f"Val Top-30 Error: {val_top30_error:.4f}")

torch.save(model.state_dict(), "cnn_model_final_2022.pth")
print("Model saved!")

# Sauvegarder les scores
results = pd.DataFrame({
    "epoch": range(1, epochs+1),
    "train_loss": errors["train"],
    "val_loss": errors["val"],
    "train_accuracy": accuracies["train"],
    "val_accuracy": accuracies["val"],
    "train_top30_error": top30_errors["train"],
    "val_top30_error": top30_errors["val"]
})


results.to_csv(SUBMISSION_PATH / "training_scores_final_2022.csv", index=False)
print("Scores saved!")

import matplotlib.pyplot as plt

# Courbe des losses
plt.figure()
plt.plot(results["epoch"], results["train_loss"], label="Train Loss")
plt.plot(results["epoch"], results["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig(SUBMISSION_PATH / "training_curves_final_2022.png")
plt.show()


# Courbe de l'accuracy
plt.figure()
plt.plot(results["epoch"], results["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy over Epochs")
plt.legend()
plt.show()
plt.savefig(SUBMISSION_PATH / "accuracy_curve_final_2022.png")
print("Graph saved!")
