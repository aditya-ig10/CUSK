import time
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

# -------------------------
# Config
# -------------------------
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
TRAIN_DIR = "CIFAR-10/train"
LABELS_CSV = "CIFAR-10/trainLabels.csv"

# -------------------------
# Custom Dataset for CIFAR-10
# -------------------------
class CIFAR10Dataset(Dataset):
    # CIFAR-10 class names
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
    
    def __init__(self, img_dir, labels_csv, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.labels_df = pd.read_csv(labels_csv)
        self.images = self.labels_df.iloc[:, 0].values  # First column is image name/id
        # Convert class names to indices
        self.labels = [self.CLASS_TO_IDX.get(label, 0) for label in self.labels_df.iloc[:, 1].values]
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, f"{img_name}.png")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# -------------------------
# Load Dataset
# -------------------------
print("Loading CIFAR-10 dataset...")
start_total = time.time()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create dataset and loader
dataset = CIFAR10Dataset(TRAIN_DIR, LABELS_CSV, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print(f"Dataset size: {len(dataset)}")
print(f"Number of batches: {len(loader)}")

# -------------------------
# Model (CNN for CIFAR-10)
# -------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 3x32x32 (RGB images)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # After 2 pooling layers: 64x8x8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for CIFAR-10
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = Net()

# -------------------------
# Training Setup
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Training Loop
# -------------------------
print("\nStarting training on CIFAR-10...\n")
start_train = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    total_loss = 0
    correct = 0
    total = 0

    for i, (xb, yb) in enumerate(loader):
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(out.data, 1)
        total += yb.size(0)
        correct += (predicted == yb).sum().item()

    epoch_time = time.time() - epoch_start
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.6f} | Accuracy: {accuracy:.2f}% | Time: {epoch_time:.2f}s")

end_train = time.time()
end_total = time.time()

# -------------------------
# Benchmark Output
# -------------------------
print("\n--- Benchmark Results ---")
print(f"Training Time: {end_train - start_train:.2f} seconds")
print(f"Total Execution Time: {end_total - start_total:.2f} seconds")