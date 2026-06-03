# CUSK - Computed for Unified Silicon Kernel

## Benchmark Results (CIFAR-10)

The `Benchmark/Results` folder contains the latest benchmark outputs for CIFAR-10 training using different backends. All scripts use a train/validation split (90/10), data augmentation (RandomHorizontalFlip, RandomCrop, RandomRotation, ColorJitter), BatchNorm layers, and early stopping.

| Setup               | Training Time (s) | Total Time (s) | Final Val Acc | Overfitting Gap |
|---------------------|-------------------|----------------|---------------|-----------------|
| CPU                 | 201.18            | 201.22         | 77.46%        | +2.16%          |
| MPS (Apple GPU)     | 113.61            | 113.67         | 78.52%        | +2.17%          |
| PyTorch Optimized   | 295.52            | 295.59         | 76.50%        | +2.63%          |

**Key Improvements (vs Previous Baseline):**
- ↑ 9.58% accuracy on CPU (67.88% → 77.46%)
- ↑ 8.56% accuracy on MPS (69.96% → 78.52%)
- ↑ 9.42% accuracy on PyTorch Optimized (67.08% → 76.50%)
- Overfitting gap well-controlled (~2% in final epochs)

**How to reproduce:**

From the `Benchmark/` directory, run:

```bash
python3 main_cpu.py --epochs 5
python3 main_mps.py --epochs 5
python3 main_pytorch_optimized.py --epochs 5 --num-workers 4
```

For quick smoke tests (synthetic data, fast):

```bash
python3 main_cpu.py --quick --epochs 1
python3 main_mps.py --quick --epochs 1
python3 main_pytorch_optimized.py --quick --epochs 1
```

See the `Benchmark/Results` folder for full logs and per-epoch details.