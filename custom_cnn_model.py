"""
Custom CNN Training Script for Sheep Classification
Author: [Your Name]
Description: This script defines, trains, and evaluates a custom CNN model
for classifying sheep breeds. It uses data augmentation, learning rate scheduling,
and computes macro F1-score during training and validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# =============================
# 🔧 Data Augmentation
# =============================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =============================
# 📦 Custom Dataset
# =============================
class SheepDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        label = row['label_idx']
        if self.transform:
            image = self.transform(image)
        return image, label

# =============================
# 📄 Load Dataset and Labels
# =============================
df = pd.read_csv('/content/sheep_data/Sheep Classification Images/train_labels.csv')
df['image_path'] = df['filename'].apply(lambda x: f"/content/sheep_data/Sheep Classification Images/train/{x}")

le = LabelEncoder()
df['label_idx'] = le.fit_transform(df['label'])

train_df, val_df = train_test_split(df, test_size=0.12, stratify=df['label_idx'], random_state=42)

train_dataset = SheepDataset(train_df, transform=train_transform)
val_dataset = SheepDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# =============================
# 🧠 Define CNN Model
# =============================
class CustomCNN(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2)
        )
        self.layer2 = self._make_block(hidden_units)
        self.layer3 = self._make_block(hidden_units)
        self.layer4 = self._make_block(hidden_units)
        self.layer5 = self._make_block(hidden_units)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, output_shape)
        )

    def _make_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fc(x)
        return x

# =============================
# 🏁 Training Setup
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CustomCNN(input_shape=3, hidden_units=64, output_shape=7).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# =============================
# 📊 Training Loop
# =============================
num_epochs = 15
train_losses, val_losses = [], []
train_f1s, val_f1s = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_preds_train, all_labels_train = [], []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds_train.extend(preds.cpu().numpy())
        all_labels_train.extend(labels.cpu().numpy())

    train_loss = running_loss / len(train_loader.dataset)
    train_f1 = f1_score(all_labels_train, all_preds_train, average='macro')

    model.eval()
    val_loss = 0.0
    all_preds_val, all_labels_val = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds_val.extend(preds.cpu().numpy())
            all_labels_val.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_f1 = f1_score(all_labels_val, all_preds_val, average='macro')

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    scheduler.step()

# =============================
# 📈 Plot Metrics
# =============================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_f1s, label='Train Macro F1')
plt.plot(range(1, num_epochs+1), val_f1s, label='Val Macro F1')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Macro F1 Score per Epoch')
plt.legend()

plt.tight_layout()
plt.show()
