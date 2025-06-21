"""
Training script for pretrained ConvNeXt model on sheep breed classification.

Uses transfer learning with ConvNeXt base pretrained on ImageNet.

Requirements:
- torch
- torchvision
- sklearn
- pandas
- matplotlib
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Data augmentation and normalization transforms for training and validation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset class for sheep images
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

def main():
     # --- Data Loading and Preprocessing ---
    # Load image paths and labels, encode labels, split into train and val
    df = pd.read_csv('/content/sheep_data/Sheep Classification Images/train_labels.csv')
    df['image_path'] = df['filename'].apply(lambda x: f"/content/sheep_data/Sheep Classification Images/train/{x}")

    le = LabelEncoder()
    df['label_idx'] = le.fit_transform(df['label'])

    train_df, val_df = train_test_split(df, test_size=0.12, stratify=df['label_idx'], random_state=42)

    train_dataset = SheepDataset(train_df, transform=train_transform)
    val_dataset = SheepDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(le.classes_)

     # --- Model Initialization ---
    # Load pretrained ConvNeXt base model, replace classifier head for our task
    weights = models.ConvNeXt_Base_Weights.DEFAULT
    model = models.convnext_base(weights=weights)
    model.classifier[2] = nn.Sequential(
        nn.Linear(model.classifier[2].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)

    # Freeze all layers except classifier head initially
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier[2].parameters():
        param.requires_grad = True
      
      # --- Training setup ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
      # --- Training loop ---
    # Unfreeze entire model after a few epochs for fine-tuning
    num_epochs = 20
    best_val_f1 = 0.0
    unfreeze_epoch = 4

    train_losses = []
    train_f1s = []
    val_losses = []
    val_f1s = []

    for epoch in range(num_epochs):
        if epoch == unfreeze_epoch:
            for param in model.parameters():
                param.requires_grad = True

        model.train()
        running_loss = 0.0
        train_labels = []
        train_preds = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_dataset)
        train_f1 = f1_score(train_labels, train_preds, average='macro')

        model.eval()
        val_loss = 0.0
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_dataset)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        train_losses.append(train_loss)
        train_f1s.append(train_f1)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Macro F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')

        scheduler.step()

    # Plot loss and F1 score curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_f1s, label='Train F1')
    plt.plot(range(1, num_epochs + 1), val_f1s, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score over epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
