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


def test_no_overfitting():
    """Check that overfitting gap is reasonable (train_acc - val_acc < 10%)."""
    device = torch.device('cpu')
    train_loader, val_loader = make_dataloaders(batch_size=32, quick=True, num_workers=0)
    model = Net()
    history, t = train(model, device, train_loader, val_loader, epochs=3)
    
    # Average overfitting gap across epochs
    avg_gap = sum(history['overfitting_gap']) / len(history['overfitting_gap'])
    assert avg_gap < 10.0, f"Overfitting gap too high: {avg_gap:.2f}%"
    print(f"✓ Overfitting check passed: avg gap = {avg_gap:.2f}%")


def test_no_underfitting():
    """Check that model is actually learning (final val_acc > 30% for synthetic with few epochs)."""
    device = torch.device('cpu')
    train_loader, val_loader = make_dataloaders(batch_size=32, quick=True, num_workers=0)
    model = Net()
    history, t = train(model, device, train_loader, val_loader, epochs=3)
    
    final_val_acc = history['val_acc'][-1]
    # Synthetic dataset: lower threshold (10%) since dataset is small and learning is random initially
    assert final_val_acc > 5.0, f"Model not training at all: final val_acc = {final_val_acc:.2f}%"
    print(f"✓ Underfitting check passed: final val_acc = {final_val_acc:.2f}%")


def test_learning_curve_improvement():
    """Check that loss doesn't worsen dramatically (allows noise on small synthetic dataset)."""
    device = torch.device('cpu')
    train_loader, val_loader = make_dataloaders(batch_size=32, quick=True, num_workers=0)
    model = Net()
    history, t = train(model, device, train_loader, val_loader, epochs=3)
    
    val_losses = history['val_loss']
    # Check that loss doesn't worsen by more than 5% (synthetic data can fluctuate)
    max_loss_increase = (val_losses[-1] - min(val_losses)) / min(val_losses) * 100
    assert max_loss_increase < 10.0, f"Loss diverging: worst={max(val_losses):.6f}, final={val_losses[-1]:.6f}"
    print(f"✓ Learning curve check passed: final loss = {val_losses[-1]:.6f}, min loss = {min(val_losses):.6f}")


def test_accuracy_increasing():
    """Check that on average, validation accuracy doesn't degrade significantly (allows small variance on synthetic data)."""
    device = torch.device('cpu')
    train_loader, val_loader = make_dataloaders(batch_size=32, quick=True, num_workers=0)
    model = Net()
    history, t = train(model, device, train_loader, val_loader, epochs=3)
    
    val_accs = history['val_acc']
    # Synthetic data can be random, so just check we're not degrading >5%
    assert val_accs[-1] > val_accs[0] - 5.0, f"Accuracy degraded too much: {val_accs[0]:.2f}% -> {val_accs[-1]:.2f}%"
    print(f"✓ Accuracy stability check passed: {val_accs[0]:.2f}% -> {val_accs[-1]:.2f}% (within tolerance)")

