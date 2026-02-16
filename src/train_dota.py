import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .dataset_sequence import SequenceDoseDataset
from .losses import WeightedMSEWithPDDLoss
from .model_dota import DoTAModel


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_crop_shape(text: Optional[str]):
    if not text:
        return None
    parts = [int(v.strip()) for v in text.split(",")]
    if len(parts) != 3:
        raise ValueError("crop-shape must be D,H,W")
    return tuple(parts)


def run_epoch(model, loader, optimizer, criterion, device, training: bool) -> float:
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    n = 0

    for ct, energy, target in loader:
        ct = ct.to(device, non_blocking=True)
        energy = energy.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            pred = model(ct, energy)
            loss = criterion(pred, target)
            if training:
                loss.backward()
                optimizer.step()

        running_loss += float(loss.item())
        n += 1

    return running_loss / max(n, 1)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DoTA-like sequence model")
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--feat-channels", type=int, default=32)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--crop-shape", type=str, default=None, help="D,H,W e.g. 150,24,24")
    p.add_argument("--default-energy-mev", type=float, default=150.0)
    p.add_argument("--high-dose-weight", type=float, default=4.0)
    p.add_argument("--threshold-ratio", type=float, default=0.5)
    p.add_argument("--pdd-loss-weight", type=float, default=0.2)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/dota")
    p.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is not available")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    crop_shape = parse_crop_shape(args.crop_shape)

    train_ds = SequenceDoseDataset(
        args.train_dir,
        crop_shape=crop_shape,
        default_energy_mev=args.default_energy_mev,
    )
    val_ds = SequenceDoseDataset(
        args.val_dir,
        crop_shape=crop_shape,
        default_energy_mev=args.default_energy_mev,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = get_device()
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required (--require-cuda) but no GPU is available")

    model = DoTAModel(
        in_channels=1,
        feat_channels=args.feat_channels,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    criterion = WeightedMSEWithPDDLoss(
        threshold_ratio=args.threshold_ratio,
        high_dose_weight=args.high_dose_weight,
        pdd_loss_weight=args.pdd_loss_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    print(f"Using device: {device}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, training=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device, training=False)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        last_path = ckpt_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "arch": "dota",
                "feat_channels": args.feat_channels,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "ff_dim": args.ff_dim,
                "dropout": args.dropout,
                "crop_shape": crop_shape,
                "default_energy_mev": args.default_energy_mev,
            },
            last_path,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "arch": "dota",
                    "feat_channels": args.feat_channels,
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "num_layers": args.num_layers,
                    "ff_dim": args.ff_dim,
                    "dropout": args.dropout,
                    "crop_shape": crop_shape,
                    "default_energy_mev": args.default_energy_mev,
                },
                best_path,
            )

    print(f"Training complete. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
