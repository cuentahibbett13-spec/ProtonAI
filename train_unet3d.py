#!/usr/bin/env python
"""
Entrenador para 3D UNet en dataset sintético (2400 muestras, variable crop).
"""
import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.model_unet3d_clean import UNet3D


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

    miopen_cache = os.environ.get("MIOPEN_CACHE_DIR")
    if not miopen_cache:
        miopen_cache = f"{tmp_root}/miopen_cache"
        os.environ["MIOPEN_CACHE_DIR"] = miopen_cache
    Path(miopen_cache).mkdir(parents=True, exist_ok=True)

    miopen_user_db = os.environ.get("MIOPEN_USER_DB_PATH")
    if not miopen_user_db:
        miopen_user_db = f"{tmp_root}/miopen_user_db"
        os.environ["MIOPEN_USER_DB_PATH"] = miopen_user_db
    Path(miopen_user_db).mkdir(parents=True, exist_ok=True)


class DoseDataset3D(Dataset):
    """Dataset simple de NPZ con cropping aleatorio."""
    
    def __init__(
        self,
        data_dir: Path,
        crop_shape: Tuple[int, int, int] = (128, 128, 128),
        random_crop: bool = True,
    ):
        self.npz_files = sorted(data_dir.glob("*.npz"))
        self.crop_shape = crop_shape
        self.random_crop = random_crop
        
        if not self.npz_files:
            raise ValueError(f"No NPZ files in {data_dir}")
        
        print(f"Loaded {len(self.npz_files)} samples")

    def _crop_or_pad_triplet(
        self,
        noisy: np.ndarray,
        target: np.ndarray,
        density: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the same crop/pad to noisy, target and density to keep voxel alignment."""
        shape = noisy.shape

        if not (target.shape == shape and density.shape == shape):
            raise ValueError(f"Shape mismatch in sample: noisy={shape}, target={target.shape}, density={density.shape}")

        if all(s >= c for s, c in zip(shape, self.crop_shape)):
            if self.random_crop:
                starts = [np.random.randint(0, s - c + 1) for s, c in zip(shape, self.crop_shape)]
            else:
                starts = [(s - c) // 2 for s, c in zip(shape, self.crop_shape)]
            slices = tuple(slice(start, start + crop) for start, crop in zip(starts, self.crop_shape))
            return noisy[slices], target[slices], density[slices]

        pad_width = [(0, max(0, c - s)) for s, c in zip(shape, self.crop_shape)]
        noisy_p = np.pad(noisy, pad_width, mode='constant', constant_values=0)
        target_p = np.pad(target, pad_width, mode='constant', constant_values=0)
        density_p = np.pad(density, pad_width, mode='constant', constant_values=0)

        starts = [0] * len(self.crop_shape)
        slices = tuple(slice(s, s + c) for s, c in zip(starts, self.crop_shape))
        return noisy_p[slices], target_p[slices], density_p[slices]

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        npz = np.load(self.npz_files[idx])
        
        noisy = npz["noisy_dose"].astype(np.float32)
        target = npz["target_dose"].astype(np.float32)
        density = npz["density"].astype(np.float32)
        
        noisy, target, density = self._crop_or_pad_triplet(noisy, target, density)
        
        # Normalize per volume
        target_max = max(target.max(), 1e-6)
        density_max = max(density.max(), 1e-6)
        
        noisy = noisy / target_max
        target = target / target_max
        density = density / density_max
        
        # Stack channels: [noisy, density]
        x = np.stack([noisy, density], axis=0).astype(np.float32)
        y = target[np.newaxis, ...].astype(np.float32)
        
        return torch.from_numpy(x), torch.from_numpy(y)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train 3D UNet for dose denoising")
    parser.add_argument("--train-dir", type=str, default="data/dataset_yuca_atomic_sweep/train")
    parser.add_argument("--val-dir", type=str, default="data/dataset_yuca_atomic_sweep/val")
    parser.add_argument("--output-dir", type=str, default="checkpoints/unet3d_clean")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-shape", type=str, default="128,128,128")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--arch", type=str, default="clean", choices=["clean", "simple", "physics"])
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def main():
    args = build_arg_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    configure_rocm_runtime_dirs()
    
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    crop_shape = tuple(int(x) for x in args.crop_shape.split(","))
    device = torch.device(args.device)
    
    print(f"Device: {device}")
    print(f"Crop shape: {crop_shape}")
    
    # Datasets
    train_dataset = DoseDataset3D(train_dir, crop_shape, random_crop=True)
    val_dataset = DoseDataset3D(val_dir, crop_shape, random_crop=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Model
    if args.arch == "clean":
        model = UNet3D(in_channels=2, out_channels=1, base_filters=args.base_filters).to(device)
    elif args.arch == "simple":
        try:
            from src.model_unet3d_simple import UNet3D as SimpleUNet3D
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Architecture 'simple' requires src/model_unet3d_simple.py in this checkout"
            ) from exc
        model = SimpleUNet3D(
            in_channels=2,
            out_channels=1,
            base_channels=args.base_channels,
            depth=args.depth,
        ).to(device)
    elif args.arch == "physics":
        from src.model_unet3d import PhysicsAwareUNet3D
        model = PhysicsAwareUNet3D(
            in_channels=2,
            out_channels=1,
            base_channels=args.base_channels,
        ).to(device)
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    # Training
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=True)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        train_loss /= len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0.0
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=True)
        with torch.no_grad():
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "crop_shape": crop_shape,
            }
            torch.save(ckpt, output_dir / "best.pt")
            print(f"  → Saved best checkpoint (val_loss={val_loss:.6f})")
    
    # Save config
    config = {
        "arch": args.arch,
        "crop_shape": crop_shape,
        "in_channels": 2,
        "out_channels": 1,
        "base_filters": args.base_filters,
        "base_channels": args.base_channels,
        "depth": args.depth,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "best_val_loss": float(best_val_loss),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    
    print(f"\nTraining complete!")
    print(f"Best val_loss: {best_val_loss:.6f}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
