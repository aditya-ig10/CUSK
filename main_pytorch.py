import time
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# -------------------------
# Config
# -------------------------
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.001

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Dataset
# -------------------------
TRAIN_DIR = "CIFAR-10/train"
LABELS_CSV = "CIFAR-10/trainLabels.csv"
PIN_MEMORY = device.type == "cuda"

transform = transforms.Compose([
    transforms.ToTensor(),
])

class CIFAR10Dataset(Dataset):
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}

    def __init__(self, img_dir, labels_csv, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.labels_df = pd.read_csv(labels_csv)
        self.images = self.labels_df.iloc[:, 0].values
        self.labels = [self.CLASS_TO_IDX[label] for label in self.labels_df.iloc[:, 1].values]

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image_path = os.path.join(self.img_dir, f"{img_name}.png")
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


trainset = CIFAR10Dataset(TRAIN_DIR, LABELS_CSV, transform=transform)

# 🔥 OPTIMIZED DATALOADER
trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=PIN_MEMORY
)

print(f"Dataset size: {len(trainset)}")
print(f"Number of batches: {len(trainloader)}")

# -------------------------
# Model
# -------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = Net().to(device)

# -------------------------
# Training
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("\nStarting training on CIFAR-10...\n")
start_total = time.time()
start_train = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    total_loss = 0
    correct = 0
    total = 0

    for xb, yb in trainloader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(out, 1)
        total += yb.size(0)
        correct += (predicted == yb).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}% | Time: {time.time()-epoch_start:.2f}s")

end_train = time.time()
end_total = time.time()

# -------------------------
# Benchmark Output
# -------------------------
print("\n--- Benchmark Results ---")
print(f"Training Time: {end_train - start_train:.2f} seconds")
print(f"Total Execution Time: {end_total - start_total:.2f} seconds")
