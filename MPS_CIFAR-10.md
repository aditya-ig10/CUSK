# Benchmark Result for CIFAR-10
## This is the result using MPS (Metal Performance Shaders)

``` (venv) aditya@MacBook-Pro-2 CUSK % python3 main_pt.py
Using device: mps
Loading CIFAR-10 dataset...
Dataset size: 50000
Number of batches: 782

Starting training on CIFAR-10 with MPS...

Epoch 1/5 | Loss: 1219.723210 | Accuracy: 43.09% | Time: 23.95s
Epoch 2/5 | Loss: 964.317696 | Accuracy: 55.68% | Time: 19.72s
Epoch 3/5 | Loss: 855.111617 | Accuracy: 61.31% | Time: 19.64s
Epoch 4/5 | Loss: 778.476578 | Accuracy: 64.78% | Time: 21.00s
Epoch 5/5 | Loss: 731.625965 | Accuracy: 67.05% | Time: 20.10s

--- Benchmark Results (MPS) ---
Training Time: 104.41 seconds```