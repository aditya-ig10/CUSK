# Benchmark Result for CIFAR-10
## This is the result using PyTorch Optimised 

```(venv) aditya@MacBook-Pro-2 Benchmark % python3 main_pytorch_optimized.py          
Using device: mps
Loading dataset and building dataloaders (optimized)...

Starting optimized training...

Epoch 1/5 | Loss: 1.608135 | Accuracy: 41.63% | Val Loss: 1.301569 | Val Acc: 54.38%
Epoch 2/5 | Loss: 1.269080 | Accuracy: 54.31% | Val Loss: 1.118496 | Val Acc: 60.30%
Epoch 3/5 | Loss: 1.120572 | Accuracy: 60.35% | Val Loss: 1.030596 | Val Acc: 63.42%
Epoch 4/5 | Loss: 1.028016 | Accuracy: 63.84% | Val Loss: 0.981083 | Val Acc: 65.36%
Epoch 5/5 | Loss: 0.963734 | Accuracy: 65.87% | Val Loss: 0.919480 | Val Acc: 67.08%

--- Benchmark Results (PyTorch Optimized) ---
Training Time: 285.41 seconds
Total Execution Time: 285.46 seconds

```