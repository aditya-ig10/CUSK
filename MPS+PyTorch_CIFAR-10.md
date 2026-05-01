# Benchmark Result for CIFAR-10
## This is the result using MPS + PyTorch (Optimised)

```(venv) aditya@MacBook-Pro-2 CUSK % python3 main_pytorch.py
Using device: mps
Dataset size: 50000
Number of batches: 391

Starting training on CIFAR-10...

Epoch 1/5 | Loss: 12.8245 | Accuracy: 98.18% | Time: 15.17s
Epoch 2/5 | Loss: 0.0001 | Accuracy: 98.44% | Time: 14.00s
Epoch 3/5 | Loss: 0.0000 | Accuracy: 98.44% | Time: 14.23s
Epoch 4/5 | Loss: 0.1471 | Accuracy: 98.41% | Time: 14.16s
Epoch 5/5 | Loss: 0.0001 | Accuracy: 98.44% | Time: 14.37s

--- Benchmark Results ---
Training Time: 71.93 seconds
Total Execution Time: 71.93 seconds
```
