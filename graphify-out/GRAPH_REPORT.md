# Graph Report - CUSK  (2026-05-03)

## Corpus Check
- 6 files · ~27,962,191 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 28 nodes · 43 edges · 5 communities detected
- Extraction: 60% EXTRACTED · 40% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]

## God Nodes (most connected - your core abstractions)
1. `make_dataloaders()` - 9 edges
2. `Net` - 8 edges
3. `SyntheticDataset` - 7 edges
4. `CIFAR10Dataset` - 6 edges
5. `train()` - 6 edges
6. `test_cpu_quick_train()` - 5 edges
7. `test_mps_quick_train_if_available()` - 5 edges
8. `main()` - 4 edges
9. `main()` - 4 edges
10. `main()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `make_dataloaders()`  [INFERRED]
  main_mps.py → train_helpers.py
- `main()` --calls--> `make_dataloaders()`  [INFERRED]
  main_cpu.py → train_helpers.py
- `test_cpu_quick_train()` --calls--> `Net`  [INFERRED]
  tests/test_training_smoke.py → train_helpers.py
- `test_mps_quick_train_if_available()` --calls--> `Net`  [INFERRED]
  tests/test_training_smoke.py → train_helpers.py
- `main()` --calls--> `make_dataloaders()`  [INFERRED]
  main_pytorch_optimized.py → train_helpers.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.24
Nodes (5): main(), main(), main(), Net, train()

### Community 1 - "Community 1"
Cohesion: 0.5
Nodes (3): main(), test_cpu_quick_train(), test_mps_quick_train_if_available()

### Community 2 - "Community 2"
Cohesion: 0.4
Nodes (2): CIFAR10Dataset, Dataset

### Community 3 - "Community 3"
Cohesion: 0.4
Nodes (2): Lightweight synthetic dataset for smoke tests., SyntheticDataset

### Community 4 - "Community 4"
Cohesion: 1.0
Nodes (2): make_dataloaders(), make_transforms()

## Knowledge Gaps
- **1 isolated node(s):** `Lightweight synthetic dataset for smoke tests.`
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 2`** (5 nodes): `CIFAR10Dataset`, `.__getitem__()`, `.__init__()`, `.__len__()`, `Dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3`** (5 nodes): `Lightweight synthetic dataset for smoke tests.`, `SyntheticDataset`, `.__getitem__()`, `.__init__()`, `.__len__()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 4`** (3 nodes): `make_dataloaders()`, `make_transforms()`, `train_helpers.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `make_dataloaders()` connect `Community 4` to `Community 0`, `Community 1`, `Community 2`, `Community 3`?**
  _High betweenness centrality (0.429) - this node is a cross-community bridge._
- **Why does `SyntheticDataset` connect `Community 3` to `Community 2`, `Community 4`?**
  _High betweenness centrality (0.305) - this node is a cross-community bridge._
- **Why does `CIFAR10Dataset` connect `Community 2` to `Community 4`?**
  _High betweenness centrality (0.239) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `make_dataloaders()` (e.g. with `main()` and `main()`) actually correct?**
  _`make_dataloaders()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `Net` (e.g. with `main()` and `main()`) actually correct?**
  _`Net` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `train()` (e.g. with `main()` and `main()`) actually correct?**
  _`train()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Lightweight synthetic dataset for smoke tests.` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._