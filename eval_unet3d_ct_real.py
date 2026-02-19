#!/usr/bin/env python
"""
Evaluar 3D UNet en CT real contra ground truth.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.model_unet3d_clean import UNet3D


def center_crop_or_pad(volume: np.ndarray, crop_shape):
    z, y, x = volume.shape
    cz, cy, cx = crop_shape

    pad_z = max(0, cz - z)
    pad_y = max(0, cy - y)
    pad_x = max(0, cx - x)
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        volume = np.pad(volume, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant", constant_values=0)
        z, y, x = volume.shape

    z0 = (z - cz) // 2
    y0 = (y - cy) // 2
    x0 = (x - cx) // 2
    return volume[z0:z0 + cz, y0:y0 + cy, x0:x0 + cx]


def evaluate_ct_real(checkpoint_path: str, ct_dir: str, output_dir: str = "outputs/unet3d_eval_ct"):
    """
    Evalúa UNet3D en CT real.
    Carga pares NPZ de CT: (noisy, target, density) y evalúa.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    crop_shape = ckpt.get("crop_shape", (128, 128, 128))
    
    # Load model
    model = UNet3D(in_channels=2, out_channels=1, base_filters=32).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Crop shape: {crop_shape}")
    print(f"Device: {device}")
    
    ct_dir = Path(ct_dir)
    npz_files = sorted(ct_dir.glob("*.npz"))
    
    if not npz_files:
        print(f"No NPZ files found in {ct_dir}")
        return
    
    results = []
    
    with torch.no_grad():
        for npz_file in tqdm(npz_files, desc="Evaluating CT real"):
            data = np.load(npz_file)
            noisy = data["noisy_dose"].astype(np.float32)
            target = data["target_dose"].astype(np.float32)
            density = data["density"].astype(np.float32)
            
            if crop_shape is not None:
                noisy = center_crop_or_pad(noisy, crop_shape)
                target = center_crop_or_pad(target, crop_shape)
                density = center_crop_or_pad(density, crop_shape)

            # Normalize (same convention as train_unet3d.py)
            target_max = max(target.max(), 1e-6)
            density_max = max(density.max(), 1e-6)
            
            noisy_norm = noisy / target_max
            density_norm = density / density_max
            
            # Stack channels
            x = np.stack([noisy_norm, density_norm], axis=0)[np.newaxis, ...].astype(np.float32)
            x = torch.from_numpy(x).to(device)
            
            # Predict
            pred = model(x).squeeze(0).squeeze(0).cpu().numpy()
            pred = pred * target_max  # Denormalize
            
            # Metrics
            mae = np.mean(np.abs(pred - target))
            rmse = np.sqrt(np.mean((pred - target) ** 2))
            
            results.append({
                "sample": npz_file.name,
                "mae": mae,
                "rmse": rmse,
            })
            
            print(f"  {npz_file.name}: MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    # Save results
    import csv
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "mae", "rmse"])
        writer.writeheader()
        writer.writerows(results)
    
    # Report
    mae_mean = np.mean([r["mae"] for r in results])
    rmse_mean = np.mean([r["rmse"] for r in results])
    
    report = f"""
=== 3D UNet Evaluation on CT Real ===
Samples: {len(results)}
Mean MAE: {mae_mean:.4f}
Mean RMSE: {rmse_mean:.4f}

Results saved to: {csv_path}
"""
    
    report_path = output_dir / "report.txt"
    report_path.write_text(report)
    print(report)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate 3D UNet on CT real")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ct-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/unet3d_eval_ct")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    evaluate_ct_real(args.checkpoint, args.ct_dir, args.output_dir)
