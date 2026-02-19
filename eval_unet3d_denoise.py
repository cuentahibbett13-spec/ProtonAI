#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch

from src.model_unet3d_clean import UNet3D

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def rmse_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    diff = a[mask] - b[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def mae_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(a[mask] - b[mask])))


def extract_profile(volume: np.ndarray, axis: str, center_a: int, center_b: int, window: int) -> np.ndarray:
    if axis == "z":
        y0, x0 = center_a, center_b
        y1, y2 = max(0, y0 - window), min(volume.shape[1], y0 + window + 1)
        x1, x2 = max(0, x0 - window), min(volume.shape[2], x0 + window + 1)
        return volume[:, y1:y2, x1:x2].mean(axis=(1, 2))
    if axis == "y":
        z0, x0 = center_a, center_b
        z1, z2 = max(0, z0 - window), min(volume.shape[0], z0 + window + 1)
        x1, x2 = max(0, x0 - window), min(volume.shape[2], x0 + window + 1)
        return volume[z1:z2, :, x1:x2].mean(axis=(0, 2))
    z0, y0 = center_a, center_b
    z1, z2 = max(0, z0 - window), min(volume.shape[0], z0 + window + 1)
    y1, y2 = max(0, y0 - window), min(volume.shape[1], y0 + window + 1)
    return volume[z1:z2, y1:y2, :].mean(axis=(0, 1))


def normalize_pdd(profile: np.ndarray) -> np.ndarray:
    peak = float(profile.max())
    if peak <= 0.0:
        return np.zeros_like(profile)
    return (profile / peak) * 100.0


def spacing_for_axis(spacing_xyz: np.ndarray, axis: str) -> float:
    if axis == "x":
        return float(spacing_xyz[0])
    if axis == "y":
        return float(spacing_xyz[1])
    return float(spacing_xyz[2])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate UNet denoising: noisy vs prediction against target")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/unet3d_denoise_eval")
    parser.add_argument("--max-samples", type=int, default=0, help="Evaluate only first N samples (0 = all)")
    parser.add_argument("--axis", type=str, choices=["z", "y", "x"], default="z")
    parser.add_argument("--window", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    configure_rocm_runtime_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

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
    if args.max_samples and args.max_samples > 0:
        files = files[: args.max_samples]

    rows = []
    with torch.no_grad():
        for f in files:
            d = np.load(f)
            noisy = d["noisy_dose"].astype(np.float32)
            target = d["target_dose"].astype(np.float32)
            density = d["density"].astype(np.float32)
            spacing = d.get("spacing", np.array([1.0, 1.0, 1.0], dtype=np.float32)).astype(np.float32)

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
            pred_clipped = np.clip(pred, 0.0, target_max)

            if args.axis == "z":
                c_a = target.shape[1] // 2
                c_b = target.shape[2] // 2
            elif args.axis == "y":
                c_a = target.shape[0] // 2
                c_b = target.shape[2] // 2
            else:
                c_a = target.shape[0] // 2
                c_b = target.shape[1] // 2

            pdd_noisy = normalize_pdd(extract_profile(noisy, args.axis, c_a, c_b, args.window))
            pdd_pred = normalize_pdd(extract_profile(pred, args.axis, c_a, c_b, args.window))
            pdd_pred_clipped = normalize_pdd(extract_profile(pred_clipped, args.axis, c_a, c_b, args.window))
            pdd_target = normalize_pdd(extract_profile(target, args.axis, c_a, c_b, args.window))

            step_mm = spacing_for_axis(spacing, args.axis)
            depth_mm = np.arange(pdd_target.shape[0], dtype=np.float32) * step_mm

            fig_path = plots_dir / f"{f.stem}_pdd.png"
            plt.figure(figsize=(8, 5))
            plt.plot(depth_mm, pdd_noisy, label="Noisy", linewidth=1.6)
            plt.plot(depth_mm, pdd_pred, label="Pred", linewidth=1.8)
            plt.plot(depth_mm, pdd_pred_clipped, label="Pred clipped", linewidth=1.8)
            plt.plot(depth_mm, pdd_target, label="Target", linewidth=1.8)
            plt.xlabel("Depth [mm]")
            plt.ylabel("PDD [%]")
            plt.title(f"{f.stem}")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()

            high_dose_mask = target >= (0.1 * target_max)

            rmse_noisy = rmse(noisy, target)
            rmse_pred = rmse(pred, target)
            rmse_pred_clipped = rmse(pred_clipped, target)
            mae_noisy = mae(noisy, target)
            mae_pred = mae(pred, target)
            mae_pred_clipped = mae(pred_clipped, target)

            rmse_noisy_hd = rmse_masked(noisy, target, high_dose_mask)
            rmse_pred_hd = rmse_masked(pred, target, high_dose_mask)
            rmse_pred_clipped_hd = rmse_masked(pred_clipped, target, high_dose_mask)
            mae_noisy_hd = mae_masked(noisy, target, high_dose_mask)
            mae_pred_hd = mae_masked(pred, target, high_dose_mask)
            mae_pred_clipped_hd = mae_masked(pred_clipped, target, high_dose_mask)

            improve_rmse = 100.0 * (rmse_noisy - rmse_pred) / max(rmse_noisy, 1e-12)
            improve_mae = 100.0 * (mae_noisy - mae_pred) / max(mae_noisy, 1e-12)
            improve_rmse_clipped = 100.0 * (rmse_noisy - rmse_pred_clipped) / max(rmse_noisy, 1e-12)
            improve_mae_clipped = 100.0 * (mae_noisy - mae_pred_clipped) / max(mae_noisy, 1e-12)

            improve_rmse_hd = 100.0 * (rmse_noisy_hd - rmse_pred_hd) / max(rmse_noisy_hd, 1e-12)
            improve_mae_hd = 100.0 * (mae_noisy_hd - mae_pred_hd) / max(mae_noisy_hd, 1e-12)
            improve_rmse_clipped_hd = 100.0 * (rmse_noisy_hd - rmse_pred_clipped_hd) / max(rmse_noisy_hd, 1e-12)
            improve_mae_clipped_hd = 100.0 * (mae_noisy_hd - mae_pred_clipped_hd) / max(mae_noisy_hd, 1e-12)

            rows.append(
                {
                    "sample": f.name,
                    "rmse_noisy": rmse_noisy,
                    "rmse_pred": rmse_pred,
                    "rmse_pred_clipped": rmse_pred_clipped,
                    "rmse_improvement_pct": improve_rmse,
                    "rmse_improvement_pct_clipped": improve_rmse_clipped,
                    "mae_noisy": mae_noisy,
                    "mae_pred": mae_pred,
                    "mae_pred_clipped": mae_pred_clipped,
                    "mae_improvement_pct": improve_mae,
                    "mae_improvement_pct_clipped": improve_mae_clipped,
                    "rmse_noisy_hd": rmse_noisy_hd,
                    "rmse_pred_hd": rmse_pred_hd,
                    "rmse_pred_clipped_hd": rmse_pred_clipped_hd,
                    "rmse_improvement_pct_hd": improve_rmse_hd,
                    "rmse_improvement_pct_clipped_hd": improve_rmse_clipped_hd,
                    "mae_noisy_hd": mae_noisy_hd,
                    "mae_pred_hd": mae_pred_hd,
                    "mae_pred_clipped_hd": mae_pred_clipped_hd,
                    "mae_improvement_pct_hd": improve_mae_hd,
                    "mae_improvement_pct_clipped_hd": improve_mae_clipped_hd,
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
                "rmse_pred_clipped",
                "rmse_improvement_pct",
                "rmse_improvement_pct_clipped",
                "mae_noisy",
                "mae_pred",
                "mae_pred_clipped",
                "mae_improvement_pct",
                "mae_improvement_pct_clipped",
                "rmse_noisy_hd",
                "rmse_pred_hd",
                "rmse_pred_clipped_hd",
                "rmse_improvement_pct_hd",
                "rmse_improvement_pct_clipped_hd",
                "mae_noisy_hd",
                "mae_pred_hd",
                "mae_pred_clipped_hd",
                "mae_improvement_pct_hd",
                "mae_improvement_pct_clipped_hd",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    rmse_noisy_mean = float(np.mean([r["rmse_noisy"] for r in rows]))
    rmse_pred_mean = float(np.mean([r["rmse_pred"] for r in rows]))
    rmse_pred_clipped_mean = float(np.mean([r["rmse_pred_clipped"] for r in rows]))
    mae_noisy_mean = float(np.mean([r["mae_noisy"] for r in rows]))
    mae_pred_mean = float(np.mean([r["mae_pred"] for r in rows]))
    mae_pred_clipped_mean = float(np.mean([r["mae_pred_clipped"] for r in rows]))
    improve_rmse_mean = float(np.mean([r["rmse_improvement_pct"] for r in rows]))
    improve_rmse_clipped_mean = float(np.mean([r["rmse_improvement_pct_clipped"] for r in rows]))
    improve_mae_mean = float(np.mean([r["mae_improvement_pct"] for r in rows]))
    improve_mae_clipped_mean = float(np.mean([r["mae_improvement_pct_clipped"] for r in rows]))

    rmse_noisy_hd_mean = float(np.mean([r["rmse_noisy_hd"] for r in rows]))
    rmse_pred_hd_mean = float(np.mean([r["rmse_pred_hd"] for r in rows]))
    rmse_pred_clipped_hd_mean = float(np.mean([r["rmse_pred_clipped_hd"] for r in rows]))
    mae_noisy_hd_mean = float(np.mean([r["mae_noisy_hd"] for r in rows]))
    mae_pred_hd_mean = float(np.mean([r["mae_pred_hd"] for r in rows]))
    mae_pred_clipped_hd_mean = float(np.mean([r["mae_pred_clipped_hd"] for r in rows]))
    improve_rmse_hd_mean = float(np.mean([r["rmse_improvement_pct_hd"] for r in rows]))
    improve_rmse_clipped_hd_mean = float(np.mean([r["rmse_improvement_pct_clipped_hd"] for r in rows]))
    improve_mae_hd_mean = float(np.mean([r["mae_improvement_pct_hd"] for r in rows]))
    improve_mae_clipped_hd_mean = float(np.mean([r["mae_improvement_pct_clipped_hd"] for r in rows]))

    wins_rmse = int(sum(1 for r in rows if r["rmse_pred"] < r["rmse_noisy"]))
    wins_rmse_clipped = int(sum(1 for r in rows if r["rmse_pred_clipped"] < r["rmse_noisy"]))
    wins_mae = int(sum(1 for r in rows if r["mae_pred"] < r["mae_noisy"]))
    wins_mae_clipped = int(sum(1 for r in rows if r["mae_pred_clipped"] < r["mae_noisy"]))

    wins_rmse_hd = int(sum(1 for r in rows if r["rmse_pred_hd"] < r["rmse_noisy_hd"]))
    wins_rmse_clipped_hd = int(sum(1 for r in rows if r["rmse_pred_clipped_hd"] < r["rmse_noisy_hd"]))
    wins_mae_hd = int(sum(1 for r in rows if r["mae_pred_hd"] < r["mae_noisy_hd"]))
    wins_mae_clipped_hd = int(sum(1 for r in rows if r["mae_pred_clipped_hd"] < r["mae_noisy_hd"]))

    report = "\n".join(
        [
            f"samples={len(rows)}",
            f"rmse_noisy_mean={rmse_noisy_mean:.6f}",
            f"rmse_pred_mean={rmse_pred_mean:.6f}",
            f"rmse_improvement_pct_mean={improve_rmse_mean:.3f}",
            f"rmse_wins={wins_rmse}/{len(rows)}",
            f"rmse_pred_clipped_mean={rmse_pred_clipped_mean:.6f}",
            f"rmse_improvement_pct_clipped_mean={improve_rmse_clipped_mean:.3f}",
            f"rmse_clipped_wins={wins_rmse_clipped}/{len(rows)}",
            f"mae_noisy_mean={mae_noisy_mean:.6f}",
            f"mae_pred_mean={mae_pred_mean:.6f}",
            f"mae_improvement_pct_mean={improve_mae_mean:.3f}",
            f"mae_wins={wins_mae}/{len(rows)}",
            f"mae_pred_clipped_mean={mae_pred_clipped_mean:.6f}",
            f"mae_improvement_pct_clipped_mean={improve_mae_clipped_mean:.3f}",
            f"mae_clipped_wins={wins_mae_clipped}/{len(rows)}",
            f"rmse_noisy_hd_mean={rmse_noisy_hd_mean:.6f}",
            f"rmse_pred_hd_mean={rmse_pred_hd_mean:.6f}",
            f"rmse_improvement_pct_hd_mean={improve_rmse_hd_mean:.3f}",
            f"rmse_hd_wins={wins_rmse_hd}/{len(rows)}",
            f"rmse_pred_clipped_hd_mean={rmse_pred_clipped_hd_mean:.6f}",
            f"rmse_improvement_pct_clipped_hd_mean={improve_rmse_clipped_hd_mean:.3f}",
            f"rmse_clipped_hd_wins={wins_rmse_clipped_hd}/{len(rows)}",
            f"mae_noisy_hd_mean={mae_noisy_hd_mean:.6f}",
            f"mae_pred_hd_mean={mae_pred_hd_mean:.6f}",
            f"mae_improvement_pct_hd_mean={improve_mae_hd_mean:.3f}",
            f"mae_hd_wins={wins_mae_hd}/{len(rows)}",
            f"mae_pred_clipped_hd_mean={mae_pred_clipped_hd_mean:.6f}",
            f"mae_improvement_pct_clipped_hd_mean={improve_mae_clipped_hd_mean:.3f}",
            f"mae_clipped_hd_wins={wins_mae_clipped_hd}/{len(rows)}",
            f"summary_csv={csv_path}",
            f"plots_dir={plots_dir}",
        ]
    )

    report_path = out_dir / "report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
