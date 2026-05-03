# Benchmark Result for CIFAR-10
## This is the result completely on CPU (No PyTorch, No MLX, No CUSK)

```(venv) aditya@MacBook-Pro-2 Benchmark % python3 main_cpu.py
Using device: cpu
Loading dataset and building dataloaders...

Starting training...

Epoch 1/5 | Loss: 1.613271 | Accuracy: 41.32% | Val Loss: 1.289158 | Val Acc: 54.54%
Epoch 2/5 | Loss: 1.298879 | Accuracy: 53.07% | Val Loss: 1.099877 | Val Acc: 60.72%
Epoch 3/5 | Loss: 1.160165 | Accuracy: 58.71% | Val Loss: 1.019876 | Val Acc: 64.42%
Epoch 4/5 | Loss: 1.067757 | Accuracy: 62.19% | Val Loss: 0.957373 | Val Acc: 65.94%
Epoch 5/5 | Loss: 0.996002 | Accuracy: 64.90% | Val Loss: 0.920277 | Val Acc: 67.88%

--- Benchmark Results (CPU) ---
Training Time: 138.21 seconds
Total Execution Time: 138.24 seconds
```

Note: scripts updated to include validation, augmentation, and early stopping. Run `python3 main_cpu.py` (or `--quick`) to reproduce and update these numbers.