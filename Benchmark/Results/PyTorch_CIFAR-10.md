# Benchmark Result for CIFAR-10 (PyTorch Optimized)
## Improved Model with BatchNorm, Enhanced Augmentation, LR Scheduling, num_workers=4

```
(venv) aditya@MacBook-Pro-2 Benchmark % python3 main_pytorch_optimized.py --epochs 5 --num-workers 4
Using device: mps
Loading dataset and building dataloaders (optimized)...

Starting optimized training...

Epoch 1/5 | Loss: 1.373904 | Acc: 50.02% | Val Loss: 1.060961 | Val Acc: 61.80% | Gap: -11.78%
Epoch 2/5 | Loss: 0.964484 | Acc: 65.95% | Val Loss: 0.927497 | Val Acc: 67.24% | Gap: -1.29%
Epoch 3/5 | Loss: 0.804215 | Acc: 71.99% | Val Loss: 0.781393 | Val Acc: 73.00% | Gap: -1.01%
Epoch 4/5 | Loss: 0.690549 | Acc: 76.02% | Val Loss: 0.713984 | Val Acc: 74.78% | Gap: 1.24%
Epoch 5/5 | Loss: 0.603915 | Acc: 79.13% | Val Loss: 0.679829 | Val Acc: 76.50% | Gap: 2.63%

--- Benchmark Results (PyTorch Optimized) ---
Training Time: 295.52 seconds
Total Execution Time: 295.59 seconds
```

### Improvements vs Previous Run
- Final Val Acc: **76.50%** (↑ 9.42% from 67.08%)
- Overfitting gap well-controlled (2.63% in final epoch)
- Larger batch size (128) and num_workers=4 used