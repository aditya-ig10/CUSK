import copy
import os
import sys
import time
from contextlib import nullcontext

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


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
    def __init__(self, length=1024, num_classes=10, image_size=(3, 32, 32)):
        self.length = length
        self.num_classes = num_classes
        self.image_shape = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label = idx % self.num_classes
        image = torch.zeros(self.image_shape, dtype=torch.float32)

        channel = label % self.image_shape[0]
        row = (label * 3) % (self.image_shape[1] - 8)
        col = (label * 5) % (self.image_shape[2] - 8)
        image[channel, row:row + 8, col:col + 8] = 1.0
        image += torch.randn(self.image_shape, dtype=torch.float32) * 0.05
        image = image.clamp(0.0, 1.0)

        return image, torch.tensor(label, dtype=torch.long)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class Net(nn.Module):
    def __init__(self, num_classes=10, dropout=0.15, classifier_dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_stage(32, 32, blocks=2, stride=1, dropout=dropout)
        self.layer2 = self._make_stage(32, 64, blocks=2, stride=2, dropout=dropout)
        self.layer3 = self._make_stage(64, 128, blocks=2, stride=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(128, num_classes)

    def _make_stage(self, in_channels, out_channels, blocks, stride, dropout):
        layers = [ResidualBlock(in_channels, out_channels, stride=stride, dropout=dropout)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)


def resolve_device(prefer='auto'):
    if prefer == 'cpu':
        return torch.device('cpu')
    if prefer == 'mps':
        return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if prefer == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def recommended_num_workers():
    if sys.platform == 'darwin':
        return 0
    cpu_count = os.cpu_count() or 1
    return max(0, min(8, cpu_count // 2))


def make_transforms(train=True, use_autoaugment=True, random_erasing=0.1):
    if train:
        train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if use_autoaugment:
            train_transforms.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
        train_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        if random_erasing > 0:
            train_transforms.append(transforms.RandomErasing(p=random_erasing, scale=(0.02, 0.2), ratio=(0.3, 3.3)))
        return transforms.Compose(train_transforms)

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def make_dataloaders(
    img_dir=None,
    labels_csv=None,
    batch_size=64,
    val_split=0.1,
    num_workers=0,
    quick=False,
    seed=42,
    use_autoaugment=True,
    random_erasing=0.1,
    pin_memory=False,
):
    if quick:
        full_dataset = SyntheticDataset(length=1024)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(full_dataset), generator=generator).tolist()
        val_len = max(1, int(len(full_dataset) * val_split))
        train_indices = indices[val_len:]
        val_indices = indices[:val_len]
        train_ds = Subset(full_dataset, train_indices)
        val_ds = Subset(full_dataset, val_indices)
    else:
        train_transform = make_transforms(train=True, use_autoaugment=use_autoaugment, random_erasing=random_erasing)
        val_transform = make_transforms(train=False)

        train_dataset = CIFAR10Dataset(img_dir, labels_csv, transform=train_transform)
        val_dataset = CIFAR10Dataset(img_dir, labels_csv, transform=val_transform)

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_dataset), generator=generator).tolist()
        val_len = max(1, int(len(train_dataset) * val_split))
        train_indices = indices[val_len:]
        val_indices = indices[:val_len]
        train_ds = Subset(train_dataset, train_indices)
        val_ds = Subset(val_dataset, val_indices)

    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def mixup_batch(inputs, targets, alpha=0.2):
    if alpha <= 0 or inputs.size(0) < 2:
        return inputs, targets, targets, 1.0

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    return mixed_inputs, targets, targets[index], lam


def _base_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _autocast_context(device, enabled):
    if not enabled or device.type not in {'cuda', 'mps'}:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def train(
    model,
    device,
    train_loader,
    val_loader=None,
    epochs=5,
    lr=3e-4,
    patience=8,
    weight_decay=5e-4,
    label_smoothing=0.1,
    mixup_alpha=0.2,
    use_amp=False,
    channels_last=False,
    grad_clip=1.0,
    checkpoint_path=None,
):
    torch.set_float32_matmul_precision('high')

    synthetic_mode = isinstance(_base_dataset(train_loader.dataset), SyntheticDataset)
    effective_label_smoothing = 0.0 if synthetic_mode else label_smoothing
    effective_mixup_alpha = 0.0 if synthetic_mode else mixup_alpha

    criterion = nn.CrossEntropyLoss(label_smoothing=effective_label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device.type == 'cuda')
    model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    best_val_loss = float('inf')
    best_epoch = -1
    best_state = None
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'overfitting_gap': [],
        'best_epoch': None,
        'best_val_acc': None,
    }

    start_train = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0.0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)

            xb, targets_a, targets_b, lam = mixup_batch(xb, yb, alpha=effective_mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, use_amp):
                out = model(xb)
                loss = lam * criterion(out, targets_a) + (1.0 - lam) * criterion(out, targets_b)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            scheduler.step()

            running_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += lam * (preds == targets_a).sum().item()
            correct += (1.0 - lam) * (preds == targets_b).sum().item()
            total += yb.size(0)

        train_loss = running_loss / max(1, total)
        train_acc = 100.0 * correct / max(1, total)

        val_loss = None
        val_acc = None
        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    if channels_last:
                        xb = xb.contiguous(memory_format=torch.channels_last)
                    with _autocast_context(device, use_amp):
                        out = model(xb)
                        loss = criterion(out, yb)
                    v_loss += loss.item() * xb.size(0)
                    preds = out.argmax(dim=1)
                    v_correct += (preds == yb).sum().item()
                    v_total += yb.size(0)
            val_loss = v_loss / max(1, v_total)
            val_acc = 100.0 * v_correct / max(1, v_total)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        overfitting_gap = train_acc - val_acc if val_acc is not None else 0.0
        history['overfitting_gap'].append(overfitting_gap)

        epoch_info = f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%"
        if val_loss is not None:
            epoch_info += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Gap: {overfitting_gap:.2f}%"
        print(epoch_info)

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            history['best_epoch'] = epoch + 1
            history['best_val_acc'] = val_acc
            if checkpoint_path:
                torch.save(best_state, checkpoint_path)
        elif val_loss is not None and (epoch - best_epoch) >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    total_time = time.time() - start_train
    return history, total_time
