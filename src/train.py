import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import ProtonDoseDataset
from .losses import WeightedMSEWithPDDLoss
from .model_unet3d import PhysicsAwareUNet3D


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, optimizer, criterion, device, training: bool, residual_learning: bool) -> float:
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    num_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(inputs)
            if residual_learning:
                predictions = predictions + inputs[:, :1]
            predictions = torch.clamp(predictions, min=0.0)
            loss = criterion(predictions, targets)

            if training:
                loss.backward()
                optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Physics-Aware U-Net 3D MVP")
    parser.add_argument("--train-dir", required=True, type=str)
    parser.add_argument("--val-dir", required=True, type=str)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--high-dose-weight", type=float, default=4.0)
    parser.add_argument("--threshold-ratio", type=float, default=0.5)
    parser.add_argument("--pdd-loss-weight", type=float, default=0.0)
    parser.add_argument("--residual-learning", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    train_dataset = ProtonDoseDataset(args.train_dir)
    val_dataset = ProtonDoseDataset(args.val_dir)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = get_device()
    model = PhysicsAwareUNet3D(in_channels=2, out_channels=1, base_channels=args.base_channels).to(device)

    criterion = WeightedMSEWithPDDLoss(
        threshold_ratio=args.threshold_ratio,
        high_dose_weight=args.high_dose_weight,
        pdd_loss_weight=args.pdd_loss_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    print(f"Using device: {device}")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            training=True,
            residual_learning=args.residual_learning,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            device,
            training=False,
            residual_learning=args.residual_learning,
        )

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        last_path = checkpoint_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "base_channels": args.base_channels,
                "residual_learning": args.residual_learning,
                "pdd_loss_weight": args.pdd_loss_weight,
            },
            last_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "base_channels": args.base_channels,
                    "residual_learning": args.residual_learning,
                    "pdd_loss_weight": args.pdd_loss_weight,
                },
                best_path,
            )

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
