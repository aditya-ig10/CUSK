# CUSK - Computed for Unified Silicon Kernel

## Benchmark Results (CIFAR-10)

The `Benchmark/Results` folder contains the latest benchmark outputs for CIFAR-10 training using different backends. All scripts use a train/validation split, data augmentation, and early stopping.

| Setup               | Training Time (s) | Total Time (s) | Final Train Acc | Final Val Acc |
|---------------------|-------------------|----------------|-----------------|---------------|
| CPU                 | 138.21            | 138.24         | 64.90%          | 67.88%        |
| MPS (Apple GPU)     | 101.66            | 101.71         | 66.77%          | 69.96%        |
| PyTorch Optimized   | 285.41            | 285.46         | 65.87%          | 67.08%        |

**How to reproduce:**

From the `Benchmark/` directory, run:

```bash
python3 main_cpu.py --epochs 5
python3 main_mps.py --epochs 5
python3 main_pytorch_optimized.py --epochs 5
```

For quick smoke tests (synthetic data, fast):

```bash
python3 main_cpu.py --quick --epochs 1
python3 main_mps.py --quick --epochs 1
python3 main_pytorch_optimized.py --quick --epochs 1
```

See the `Benchmark/Results` folder for full logs and per-epoch details.