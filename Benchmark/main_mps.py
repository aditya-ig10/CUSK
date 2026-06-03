import argparse
import time
from pathlib import Path

from train_helpers import Net, make_dataloaders, recommended_num_workers, resolve_device, train


def main():
    benchmark_dir = Path(__file__).resolve().parent
    default_train_dir = benchmark_dir / 'CIFAR-10' / 'train'
    default_labels_csv = benchmark_dir / 'CIFAR-10' / 'trainLabels.csv'
    default_checkpoint = benchmark_dir / 'best_cifar10_mps.pt'

    parser = argparse.ArgumentParser(description='Train CIFAR-10 on Apple Silicon GPU with stronger regularization.')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--train-dir', type=str, default=str(default_train_dir))
    parser.add_argument('--labels-csv', type=str, default=str(default_labels_csv))
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=recommended_num_workers())
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--random-erasing', type=float, default=0.1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--checkpoint-path', type=str, default=str(default_checkpoint))
    parser.add_argument('--disable-autoaugment', action='store_true')
    parser.add_argument('--disable-amp', action='store_true')
    parser.add_argument('--disable-channels-last', action='store_true')
    parser.add_argument('--quick', action='store_true', help='Run on a small synthetic dataset for quick tests')
    args = parser.parse_args()

    device = resolve_device(prefer='mps')
    use_amp = device.type == 'mps' and not args.disable_amp
    channels_last = device.type in {'mps', 'cuda'} and not args.disable_channels_last

    print(f'Using device: {device}')
    if device.type != 'mps':
        print('MPS is unavailable, so training is falling back to CPU.')
    if use_amp:
        print('Automatic mixed precision enabled for MPS.')
    if channels_last:
        print('Channels-last memory format enabled.')

    print('Loading dataset and building dataloaders...')
    start_total = time.time()
    train_loader, val_loader = make_dataloaders(
        img_dir=args.train_dir,
        labels_csv=args.labels_csv,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        quick=args.quick,
        seed=args.seed,
        use_autoaugment=not args.disable_autoaugment,
        random_erasing=args.random_erasing,
        pin_memory=device.type == 'cuda',
    )

    model = Net()

    print('\nStarting optimized MPS training...\n')
    history, train_time = train(
        model,
        device,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        use_amp=use_amp,
        channels_last=channels_last,
        grad_clip=args.grad_clip,
        checkpoint_path=None if args.quick else args.checkpoint_path,
    )

    end_total = time.time()

    print('\n--- Benchmark Results (MPS Optimized) ---')
    print(f'Training Time: {train_time:.2f} seconds')
    print(f'Total Execution Time: {end_total - start_total:.2f} seconds')
    if history['best_epoch'] is not None:
        print(f"Best Epoch: {history['best_epoch']}")
    if history['best_val_acc'] is not None:
        print(f"Best Validation Accuracy: {history['best_val_acc']:.2f}%")
    if history['overfitting_gap']:
        print(f"Final Overfitting Gap: {history['overfitting_gap'][-1]:.2f}%")


if __name__ == '__main__':
    main()
