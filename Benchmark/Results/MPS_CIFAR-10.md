# Benchmark Result for CIFAR-10
## This is the result using MPS (Metal Performance Shaders)

``` (venv) aditya@MacBook-Pro-2 Benchmark % python3 main_mps.py
Using device: mps
Loading dataset and building dataloaders...

Starting training...

Epoch 1/5 | Loss: 1.556418 | Accuracy: 43.16% | Val Loss: 1.198781 | Val Acc: 56.38%
Epoch 2/5 | Loss: 1.238290 | Accuracy: 55.81% | Val Loss: 1.071678 | Val Acc: 62.60%
Epoch 3/5 | Loss: 1.106870 | Accuracy: 60.70% | Val Loss: 0.968281 | Val Acc: 66.04%
Epoch 4/5 | Loss: 1.012335 | Accuracy: 64.38% | Val Loss: 0.943936 | Val Acc: 66.98%
Epoch 5/5 | Loss: 0.945802 | Accuracy: 66.77% | Val Loss: 0.878047 | Val Acc: 69.96%

--- Benchmark Results (MPS) ---
Training Time: 101.66 seconds
Total Execution Time: 101.71 seconds ```

Note: scripts updated to include validation, augmentation, and early stopping. Run `python3 main_mps.py` (or `--quick`) to reproduce and update these numbers.