# Benchmark Result for CIFAR-10 (CPU)
## Improved Model with BatchNorm, Enhanced Augmentation, LR Scheduling

```
(venv) aditya@MacBook-Pro-2 Benchmark % python3 main_cpu.py --epochs 5
Using device: cpu
Loading dataset and building dataloaders...

Starting training...

Epoch 1/5 | Loss: 1.335390 | Acc: 51.71% | Val Loss: 1.012168 | Val Acc: 63.32% | Gap: -11.61%
Epoch 2/5 | Loss: 0.968063 | Acc: 65.61% | Val Loss: 0.836402 | Val Acc: 70.14% | Gap: -4.53%
Epoch 3/5 | Loss: 0.805786 | Acc: 71.63% | Val Loss: 0.744271 | Val Acc: 73.58% | Gap: -1.95%
Epoch 4/5 | Loss: 0.681056 | Acc: 76.15% | Val Loss: 0.664228 | Val Acc: 76.60% | Gap: -0.45%
Epoch 5/5 | Loss: 0.588540 | Acc: 79.62% | Val Loss: 0.636663 | Val Acc: 77.46% | Gap: 2.16%

--- Benchmark Results (CPU) ---
Training Time: 201.18 seconds
Total Execution Time: 201.22 seconds
```

### Improvements vs Previous Run
- Final Val Acc: **77.46%** (↑ 9.58% from 67.88%)
- Overfitting gap well-controlled (2.16% in final epoch)
- Better generalization with BatchNorm and improved augmentation