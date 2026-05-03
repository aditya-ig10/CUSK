# Benchmark Result for CIFAR-10
## This is the result completely on CPU (No PyTorch, No MLX, No CUSK)

```(venv) aditya@MacBook-Pro-2 Benchmark % python3 main_cpu.py
Loading CIFAR-10 dataset...
Dataset size: 50000
Number of batches: 782

Starting training on CIFAR-10...

Epoch 1/5 | Loss: 1178.180793 | Accuracy: 45.41% | Time: 31.04s
Epoch 2/5 | Loss: 926.347111 | Accuracy: 57.86% | Time: 31.10s
Epoch 3/5 | Loss: 823.378290 | Accuracy: 62.83% | Time: 31.76s
Epoch 4/5 | Loss: 759.806261 | Accuracy: 66.11% | Time: 28.95s
Epoch 5/5 | Loss: 702.740551 | Accuracy: 68.22% | Time: 32.00s

--- Benchmark Results ---
Training Time: 154.85 seconds
Total Execution Time: 154.88 seconds
```

Note: scripts updated to include validation, augmentation, and early stopping. Run `python3 main_cpu.py` (or `--quick`) to reproduce and update these numbers.