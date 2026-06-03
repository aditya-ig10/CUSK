import os
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd


class CIFAR10Dataset(Dataset):
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

    def __init__(self, img_dir, labels_csv, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.labels_df = pd.read_csv(labels_csv)
        self.images = self.labels_df.iloc[:, 0].values
        self.labels = [self.CLASS_TO_IDX.get(label, 0) for label in self.labels_df.iloc[:, 1].values]

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, f"{img_name}.png")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


class SyntheticDataset(Dataset):
    """Lightweight synthetic dataset for smoke tests."""
    def __init__(self, length=512, num_classes=10, image_size=(3, 32, 32), transform=None):
        self.length = length
        self.num_classes = num_classes
        self.transform = transform
        self.image_shape = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.randn(self.image_shape)
        label = random.randint(0, self.num_classes - 1)
        if self.transform:
            img = self.transform(img)
        return img, label


class Net(nn.Module):
    """Improved CNN with BatchNorm and reduced overfitting."""
    def __init__(self):
        super().__init__()
        # Conv block 1: 3x32x32 -> 32x32x32 -> 32x16x16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        
        # Conv block 2: 32x16x16 -> 64x16x16 -> 64x8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv block 3: 64x8x8 -> 128x8x8 -> 128x4x4 (additional layer for better feature extraction)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # FC layers: 128x4x4 = 2048 -> 256 -> 10
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        
        # Reduced dropout to avoid underfitting (from 0.5 to 0.3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def make_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(10),  # ±10 degree rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


def make_dataloaders(img_dir=None, labels_csv=None, batch_size=64, val_split=0.1, num_workers=0, quick=False):
    if quick:
        dataset = SyntheticDataset(length=1024)
        val_len = int(len(dataset) * val_split)
        train_len = len(dataset) - val_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len])
    else:
        train_transform = make_transforms(train=True)
        val_transform = make_transforms(train=False)
        full_dataset = CIFAR10Dataset(img_dir, labels_csv, transform=train_transform)
        val_len = int(len(full_dataset) * val_split)
        train_len = len(full_dataset) - val_len
        train_ds, val_ds = random_split(full_dataset, [train_len, val_len])
        # override val transform
        val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def train(model, device, train_loader, val_loader=None, epochs=5, lr=1e-3, patience=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler: cosine annealing for smooth decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.to(device)

    best_val_loss = float('inf')
    best_epoch = 0
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [],
        'overfitting_gap': []  # track train_acc - val_acc to detect overfitting
    }

    start_train = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        val_loss = None
        val_acc = None
        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    out = model(xb)
                    loss = criterion(out, yb)
                    v_loss += loss.item() * xb.size(0)
                    _, preds = torch.max(out, 1)
                    v_correct += (preds == yb).sum().item()
                    v_total += yb.size(0)
            val_loss = v_loss / v_total
            val_acc = 100.0 * v_correct / v_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Track overfitting: gap between train and val accuracy
        overfitting_gap = train_acc - val_acc if val_acc is not None else 0
        history['overfitting_gap'].append(overfitting_gap)

        epoch_info = f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.6f} | Acc: {train_acc:.2f}%"
        if val_loss is not None:
            epoch_info += f" | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.2f}% | Gap: {overfitting_gap:.2f}%"
        print(epoch_info)

        # Early stopping
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        elif val_loss is not None and (epoch - best_epoch) >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Step scheduler
        scheduler.step()

    total_time = time.time() - start_train
    return history, total_time
