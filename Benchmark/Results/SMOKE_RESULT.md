(venv) aditya@MacBook-Pro-2 Benchmark %  python3 run_smoke_tests.py
============================================================
SMOKE TESTS: Quick validation of training pipeline
============================================================

1. Testing CPU quick train...
Epoch 1/1 | Loss: 2.516008 | Acc: 9.76% | Val Loss: 2.337077 | Val Acc: 11.76% | Gap: -2.00%
   ✓ CPU quick train test passed

2. Testing MPS quick train (device fallback to CPU if MPS unavailable)...
Epoch 1/1 | Loss: 2.514604 | Acc: 10.74% | Val Loss: 2.350755 | Val Acc: 6.86% | Gap: 3.87%
   ✓ MPS quick train test passed

3. Testing no overfitting...
Epoch 1/3 | Loss: 2.543420 | Acc: 9.65% | Val Loss: 2.348963 | Val Acc: 10.78% | Gap: -1.13%
Epoch 2/3 | Loss: 2.314293 | Acc: 10.09% | Val Loss: 2.309950 | Val Acc: 9.80% | Gap: 0.28%
Epoch 3/3 | Loss: 2.307975 | Acc: 8.68% | Val Loss: 2.307991 | Val Acc: 8.82% | Gap: -0.15%
✓ Overfitting check passed: avg gap = -0.33%

4. Testing no underfitting...
Epoch 1/3 | Loss: 2.548877 | Acc: 9.11% | Val Loss: 2.320019 | Val Acc: 8.82% | Gap: 0.29%
Epoch 2/3 | Loss: 2.308843 | Acc: 8.89% | Val Loss: 2.309632 | Val Acc: 7.84% | Gap: 1.05%
Epoch 3/3 | Loss: 2.302850 | Acc: 10.85% | Val Loss: 2.300862 | Val Acc: 9.80% | Gap: 1.04%
✓ Underfitting check passed: final val_acc = 9.80%

5. Testing learning curve improvement...
Epoch 1/3 | Loss: 2.524542 | Acc: 9.54% | Val Loss: 2.317453 | Val Acc: 11.76% | Gap: -2.22%
Epoch 2/3 | Loss: 2.314814 | Acc: 10.95% | Val Loss: 2.329527 | Val Acc: 10.78% | Gap: 0.17%
Epoch 3/3 | Loss: 2.313003 | Acc: 10.20% | Val Loss: 2.305168 | Val Acc: 7.84% | Gap: 2.35%
✓ Learning curve check passed: final loss = 2.305168, min loss = 2.305168

6. Testing accuracy increase over epochs...
Epoch 1/3 | Loss: 2.482747 | Acc: 11.06% | Val Loss: 2.336970 | Val Acc: 7.84% | Gap: 3.22%
Epoch 2/3 | Loss: 2.311498 | Acc: 11.61% | Val Loss: 2.301211 | Val Acc: 13.73% | Gap: -2.12%
Epoch 3/3 | Loss: 2.306303 | Acc: 9.76% | Val Loss: 2.306101 | Val Acc: 8.82% | Gap: 0.94%
✓ Accuracy stability check passed: 7.84% -> 8.82% (within tolerance)

============================================================
ALL TESTS PASSED!
============================================================