import torch
from train_helpers import make_dataloaders, Net, train


def test_cpu_quick_train():
    device = torch.device('cpu')
    train_loader, val_loader = make_dataloaders(batch_size=32, quick=True, num_workers=0)
    model = Net()
    history, t = train(model, device, train_loader, val_loader, epochs=1)
    assert 'train_loss' in history and len(history['train_loss']) >= 1


def test_mps_quick_train_if_available():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    train_loader, val_loader = make_dataloaders(batch_size=32, quick=True, num_workers=0)
    model = Net()
    history, t = train(model, device, train_loader, val_loader, epochs=1)
    assert 'train_acc' in history and len(history['train_acc']) >= 1
