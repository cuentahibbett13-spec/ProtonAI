#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from swin_dose_mc.data import DoseNPZDataset
from swin_dose_mc.losses import physics_informed_loss
from swin_dose_mc.model import SwinDoseMC


def configure_rocm_runtime_dirs() -> None:
    user = os.environ.get("USER", "user")

    tmp_root = os.environ.get("TMPDIR")
    if not tmp_root or not Path(tmp_root).exists():
        slurm_tmp = os.environ.get("SLURM_TMPDIR")
        if slurm_tmp and Path(slurm_tmp).exists():
            tmp_root = slurm_tmp
        else:
            tmp_root = f"/tmp/{user}/protonai_tmp"
    os.environ["TMPDIR"] = tmp_root
    Path(tmp_root).mkdir(parents=True, exist_ok=True)

    miopen_cache = os.environ.get("MIOPEN_CACHE_DIR", f"{tmp_root}/miopen_cache")
    os.environ["MIOPEN_CACHE_DIR"] = miopen_cache
    Path(miopen_cache).mkdir(parents=True, exist_ok=True)

    miopen_user_db = os.environ.get("MIOPEN_USER_DB_PATH", f"{tmp_root}/miopen_user_db")
    os.environ["MIOPEN_USER_DB_PATH"] = miopen_user_db
    Path(miopen_user_db).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Swin-Dose MC model")
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints/swin_dose_mc")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-shape", type=str, default="96,96,96")
    parser.add_argument("--feature-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--target-threshold", type=float, default=0.7)
    parser.add_argument("--energy-min", type=float, default=70.0)
    parser.add_argument("--energy-max", type=float, default=250.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_rocm_runtime_dirs()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    crop_shape = tuple(int(v) for v in args.crop_shape.split(","))
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DoseNPZDataset(
        data_dir=Path(args.train_dir),
        crop_shape=crop_shape,
        random_crop=True,
        include_energy=True,
        energy_min_mev=args.energy_min,
        energy_max_mev=args.energy_max,
        max_samples=args.max_train_samples,
    )
    val_ds = DoseNPZDataset(
        data_dir=Path(args.val_dir),
        crop_shape=crop_shape,
        random_crop=False,
        include_energy=True,
        energy_min_mev=args.energy_min,
        energy_max_mev=args.energy_max,
        max_samples=args.max_val_samples,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = SwinDoseMC(
        img_size=crop_shape,
        in_channels=2,
        out_channels=1,
        feature_size=args.feature_size,
        use_energy_token=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    start_epoch = 0
    best_val = float("inf")
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))))
        print(f"Resuming from epoch {start_epoch}, best_val={best_val:.6f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            e = batch["energy"].to(device)

            optimizer.zero_grad()
            pred = model(x, e)
            loss, metrics = physics_informed_loss(
                pred,
                y,
                alpha=args.alpha,
                beta=args.beta,
                target_threshold=args.target_threshold,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{metrics['loss']:.6f}", "g": f"{metrics['loss_global']:.6f}"})

        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in pbar:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                e = batch["energy"].to(device)
                pred = model(x, e)
                loss, metrics = physics_informed_loss(
                    pred,
                    y,
                    alpha=args.alpha,
                    beta=args.beta,
                    target_threshold=args.target_threshold,
                )
                val_loss += float(loss.item())
                pbar.set_postfix({"loss": f"{metrics['loss']:.6f}"})

        val_loss /= max(len(val_loader), 1)
        scheduler.step()

        print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": min(best_val, val_loss),
            "crop_shape": crop_shape,
            "feature_size": args.feature_size,
            "energy_min": args.energy_min,
            "energy_max": args.energy_max,
            "arch": "swin_unetr",
        }
        torch.save(ckpt, output_dir / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            ckpt["best_val_loss"] = best_val
            torch.save(ckpt, output_dir / "best.pt")
            print(f"  -> Saved best checkpoint (val_loss={best_val:.6f})")

    config = vars(args)
    config["crop_shape"] = crop_shape
    config["best_val_loss"] = float(best_val)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Training complete. Best val_loss={best_val:.6f}")


if __name__ == "__main__":
    main()
