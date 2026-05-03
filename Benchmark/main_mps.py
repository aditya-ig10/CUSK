import argparse
import time
import torch
from train_helpers import make_dataloaders, Net, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train-dir', type=str, default='CIFAR-10/train')
    parser.add_argument('--labels-csv', type=str, default='CIFAR-10/trainLabels.csv')
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--quick', action='store_true', help='Run on a small synthetic dataset for quick tests')
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading dataset and building dataloaders...")
    start_total = time.time()
    train_loader, val_loader = make_dataloaders(
        img_dir=args.train_dir,
        labels_csv=args.labels_csv,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        quick=args.quick
    )

    model = Net()

    print("\nStarting training...\n")
    history, train_time = train(model, device, train_loader, val_loader, epochs=args.epochs, lr=args.lr)

    end_total = time.time()

    print("\n--- Benchmark Results (MPS) ---")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Total Execution Time: {end_total - start_total:.2f} seconds")


if __name__ == '__main__':
    main()
