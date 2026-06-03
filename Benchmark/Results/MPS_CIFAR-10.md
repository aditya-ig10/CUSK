# Benchmark Result for CIFAR-10 (MPS)
## Improved Model with BatchNorm, Enhanced Augmentation, LR Scheduling

```
(venv) aditya@MacBook-Pro-2 Benchmark % python3 main_mps.py --epochs 5
Using device: mps
Loading dataset and building dataloaders...

Starting training...

Epoch 1/5 | Loss: 1.292540 | Acc: 53.02% | Val Loss: 0.996827 | Val Acc: 64.64% | Gap: -11.62%
Epoch 2/5 | Loss: 0.933103 | Acc: 66.82% | Val Loss: 0.863069 | Val Acc: 68.90% | Gap: -2.08%
Epoch 3/5 | Loss: 0.774403 | Acc: 72.82% | Val Loss: 0.732703 | Val Acc: 74.92% | Gap: -2.10%
Epoch 4/5 | Loss: 0.645919 | Acc: 77.32% | Val Loss: 0.694345 | Val Acc: 75.52% | Gap: 1.80%
Epoch 5/5 | Loss: 0.551065 | Acc: 80.69% | Val Loss: 0.621639 | Val Acc: 78.52% | Gap: 2.17%

--- Benchmark Results (MPS) ---
Training Time: 113.61 seconds
Total Execution Time: 113.67 seconds
```

### Improvements vs Previous Run
- Final Val Acc: **78.52%** (↑ 8.56% from 69.96%)
- Overfitting gap well-controlled (2.17% in final epoch)
- Consistently best performer across training