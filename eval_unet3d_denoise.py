#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import torch

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


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate UNet denoising: noisy vs prediction against target")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/unet3d_denoise_eval")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    crop_shape = ckpt.get("crop_shape", (128, 128, 128))
    if crop_shape is not None:
        crop_shape = tuple(crop_shape)

    model = UNet3D(in_channels=2, out_channels=1, base_filters=32).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_dir = Path(args.val_dir)
    files = sorted(val_dir.glob("*.npz"))
    if not files:
        raise ValueError(f"No .npz files found in {val_dir}")

    rows = []
    with torch.no_grad():
        for f in files:
            d = np.load(f)
            noisy = d["noisy_dose"].astype(np.float32)
            target = d["target_dose"].astype(np.float32)
            density = d["density"].astype(np.float32)

            if crop_shape is not None:
                noisy = center_crop_or_pad(noisy, crop_shape)
                target = center_crop_or_pad(target, crop_shape)
                density = center_crop_or_pad(density, crop_shape)

            target_max = max(float(target.max()), 1e-6)
            density_max = max(float(density.max()), 1e-6)

            noisy_norm = noisy / target_max
            density_norm = density / density_max

            x = np.stack([noisy_norm, density_norm], axis=0)[np.newaxis, ...].astype(np.float32)
            x_t = torch.from_numpy(x).to(device)

            pred_norm = model(x_t).squeeze(0).squeeze(0).cpu().numpy()
            pred = pred_norm * target_max

            rmse_noisy = rmse(noisy, target)
            rmse_pred = rmse(pred, target)
            mae_noisy = mae(noisy, target)
            mae_pred = mae(pred, target)

            improve_rmse = 100.0 * (rmse_noisy - rmse_pred) / max(rmse_noisy, 1e-12)
            improve_mae = 100.0 * (mae_noisy - mae_pred) / max(mae_noisy, 1e-12)

            rows.append(
                {
                    "sample": f.name,
                    "rmse_noisy": rmse_noisy,
                    "rmse_pred": rmse_pred,
                    "rmse_improvement_pct": improve_rmse,
                    "mae_noisy": mae_noisy,
                    "mae_pred": mae_pred,
                    "mae_improvement_pct": improve_mae,
                }
            )

    import csv

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample",
                "rmse_noisy",
                "rmse_pred",
                "rmse_improvement_pct",
                "mae_noisy",
                "mae_pred",
                "mae_improvement_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    rmse_noisy_mean = float(np.mean([r["rmse_noisy"] for r in rows]))
    rmse_pred_mean = float(np.mean([r["rmse_pred"] for r in rows]))
    mae_noisy_mean = float(np.mean([r["mae_noisy"] for r in rows]))
    mae_pred_mean = float(np.mean([r["mae_pred"] for r in rows]))
    improve_rmse_mean = float(np.mean([r["rmse_improvement_pct"] for r in rows]))
    improve_mae_mean = float(np.mean([r["mae_improvement_pct"] for r in rows]))

    wins_rmse = int(sum(1 for r in rows if r["rmse_pred"] < r["rmse_noisy"]))
    wins_mae = int(sum(1 for r in rows if r["mae_pred"] < r["mae_noisy"]))

    report = "\n".join(
        [
            f"samples={len(rows)}",
            f"rmse_noisy_mean={rmse_noisy_mean:.6f}",
            f"rmse_pred_mean={rmse_pred_mean:.6f}",
            f"rmse_improvement_pct_mean={improve_rmse_mean:.3f}",
            f"rmse_wins={wins_rmse}/{len(rows)}",
            f"mae_noisy_mean={mae_noisy_mean:.6f}",
            f"mae_pred_mean={mae_pred_mean:.6f}",
            f"mae_improvement_pct_mean={improve_mae_mean:.3f}",
            f"mae_wins={wins_mae}/{len(rows)}",
            f"summary_csv={csv_path}",
        ]
    )

    report_path = out_dir / "report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
