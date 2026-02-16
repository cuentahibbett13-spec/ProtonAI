import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

from .dataset_sequence import SequenceDoseDataset
from .losses import WeightedMSEWithPDDLoss
from .model_dota import DoTAModel


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_rocm_runtime_dirs() -> None:
    user = os.environ.get("USER", "user")

    tmp_root = os.environ.get("TMPDIR")
    if not tmp_root or not Path(tmp_root).exists():
        tmp_root = f"/tmp/{user}/protonai_tmp"
        os.environ["TMPDIR"] = tmp_root

    tmp_path = Path(tmp_root)
    tmp_path.mkdir(parents=True, exist_ok=True)

    miopen_cache = os.environ.get("MIOPEN_CACHE_DIR")
    if not miopen_cache:
        miopen_cache = f"/tmp/{user}/miopen_cache"
        os.environ["MIOPEN_CACHE_DIR"] = miopen_cache
    Path(miopen_cache).mkdir(parents=True, exist_ok=True)

    miopen_user_db = os.environ.get("MIOPEN_USER_DB_PATH")
    if not miopen_user_db:
        miopen_user_db = f"/tmp/{user}/miopen_user_db"
        os.environ["MIOPEN_USER_DB_PATH"] = miopen_user_db
    Path(miopen_user_db).mkdir(parents=True, exist_ok=True)


def parse_crop_shape(text: Optional[str]):
    if not text:
        return None
    parts = [int(v.strip()) for v in text.split(",")]
    if len(parts) != 3:
        raise ValueError("crop-shape must be D,H,W")
    return tuple(parts)


def run_epoch(model, loader, optimizer, criterion, device, training: bool, epoch: int) -> float:
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    n = 0

    phase = "train" if training else "val"
    progress = tqdm(loader, desc=f"Epoch {epoch:03d} [{phase}]", leave=False)

    for ct, energy, target in progress:
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
        progress.set_postfix(loss=f"{running_loss / max(n, 1):.6f}")

    return running_loss / max(n, 1)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DoTA-like sequence model")
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--in-channels", type=int, default=2)
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
    p.add_argument("--exp-weight-scale", type=float, default=0.0, help="0 disables exponential weighting")
    p.add_argument("--exp-weight-gamma", type=float, default=6.0, help="Exponent strength for dose weighting")
    p.add_argument(
        "--decay-exp-alpha",
        type=float,
        default=0.0,
        help="Article-style decaying exponential voxel weight alpha (e.g., 3.0)",
    )
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/dota")
    p.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is not available")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    crop_shape = parse_crop_shape(args.crop_shape)

    configure_rocm_runtime_dirs()

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
        in_channels=args.in_channels,
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
        exp_weight_scale=args.exp_weight_scale,
        exp_weight_gamma=args.exp_weight_gamma,
        decay_exp_alpha=args.decay_exp_alpha,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    print(f"Using device: {device}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, training=True, epoch=epoch)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device, training=False, epoch=epoch)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        last_path = ckpt_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "arch": "dota",
                "in_channels": args.in_channels,
                "feat_channels": args.feat_channels,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "ff_dim": args.ff_dim,
                "dropout": args.dropout,
                "crop_shape": crop_shape,
                "default_energy_mev": args.default_energy_mev,
                "exp_weight_scale": args.exp_weight_scale,
                "exp_weight_gamma": args.exp_weight_gamma,
                "decay_exp_alpha": args.decay_exp_alpha,
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
                    "in_channels": args.in_channels,
                    "feat_channels": args.feat_channels,
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "num_layers": args.num_layers,
                    "ff_dim": args.ff_dim,
                    "dropout": args.dropout,
                    "crop_shape": crop_shape,
                    "default_energy_mev": args.default_energy_mev,
                    "exp_weight_scale": args.exp_weight_scale,
                    "exp_weight_gamma": args.exp_weight_gamma,
                    "decay_exp_alpha": args.decay_exp_alpha,
                },
                best_path,
            )

    print(f"Training complete. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
