import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model_unet3d import PhysicsAwareUNet3D


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


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PDD prediction on validation NPZ files")
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/pdd_eval")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--axis", type=str, choices=["z", "y", "x"], default="z")
    parser.add_argument("--window", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    val_dir = Path(args.val_dir)
    files = sorted(val_dir.glob("*.npz"))
    if not files:
        raise ValueError(f"No .npz files found in {val_dir}")

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    base_channels = int(checkpoint.get("base_channels", args.base_channels))
    residual_learning = bool(checkpoint.get("residual_learning", False))

    model = PhysicsAwareUNet3D(in_channels=2, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []

    with torch.no_grad():
        for npz_path in files:
            data = np.load(npz_path)
            noisy = data["noisy_dose"].astype(np.float32)
            target = data["target_dose"].astype(np.float32)
            density = data["density"].astype(np.float32)
            spacing = data.get("spacing", np.array([1.0, 1.0, 1.0], dtype=np.float32)).astype(np.float32)

            max_target = float(target.max())
            target_norm = target / max_target if max_target > 0 else target
            noisy_norm = noisy / max_target if max_target > 0 else noisy
            density_norm = density / max(float(density.max()), 1.0)

            model_input = np.stack([noisy_norm, density_norm], axis=0)
            model_input_t = torch.from_numpy(model_input).unsqueeze(0).to(device)

            pred_norm_t = model(model_input_t)
            if residual_learning:
                pred_norm_t = pred_norm_t + model_input_t[:, :1]
            pred_norm_t = torch.clamp(pred_norm_t, min=0.0)
            pred_norm = pred_norm_t.squeeze(0).squeeze(0).cpu().numpy()
            pred_norm = np.clip(pred_norm, a_min=0.0, a_max=None)

            if args.axis == "z":
                c_a = target.shape[1] // 2
                c_b = target.shape[2] // 2
            elif args.axis == "y":
                c_a = target.shape[0] // 2
                c_b = target.shape[2] // 2
            else:
                c_a = target.shape[0] // 2
                c_b = target.shape[1] // 2

            pdd_noisy = normalize_pdd(extract_profile(noisy_norm, args.axis, c_a, c_b, args.window))
            pdd_pred = normalize_pdd(extract_profile(pred_norm, args.axis, c_a, c_b, args.window))
            pdd_target = normalize_pdd(extract_profile(target_norm, args.axis, c_a, c_b, args.window))

            step_mm = spacing_for_axis(spacing, args.axis)
            depth_mm = np.arange(pdd_target.shape[0], dtype=np.float32) * step_mm

            peak_noisy_idx = int(np.argmax(pdd_noisy))
            peak_pred_idx = int(np.argmax(pdd_pred))
            peak_target_idx = int(np.argmax(pdd_target))

            peak_noisy_mm = float(depth_mm[peak_noisy_idx])
            peak_pred_mm = float(depth_mm[peak_pred_idx])
            peak_target_mm = float(depth_mm[peak_target_idx])

            peak_error_pred_mm = abs(peak_pred_mm - peak_target_mm)
            peak_error_noisy_mm = abs(peak_noisy_mm - peak_target_mm)
            pdd_rmse_pred = float(np.sqrt(np.mean((pdd_pred - pdd_target) ** 2)))
            pdd_rmse_noisy = float(np.sqrt(np.mean((pdd_noisy - pdd_target) ** 2)))

            rows.append(
                [
                    npz_path.name,
                    peak_noisy_mm,
                    peak_pred_mm,
                    peak_target_mm,
                    peak_error_noisy_mm,
                    peak_error_pred_mm,
                    pdd_rmse_noisy,
                    pdd_rmse_pred,
                ]
            )

            fig_path = plots_dir / f"{npz_path.stem}_pdd.png"
            plt.figure(figsize=(8, 5))
            plt.plot(depth_mm, pdd_noisy, label="Noisy", linewidth=1.8)
            plt.plot(depth_mm, pdd_pred, label="Pred", linewidth=1.8)
            plt.plot(depth_mm, pdd_target, label="Target", linewidth=1.8)
            plt.xlabel("Depth [mm]")
            plt.ylabel("PDD [%]")
            plt.title(npz_path.stem)
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()

    summary_path = output_dir / "summary.csv"
    header = (
        "sample,peak_noisy_mm,peak_pred_mm,peak_target_mm,"
        "peak_error_noisy_mm,peak_error_pred_mm,pdd_rmse_noisy,pdd_rmse_pred"
    )
    np.savetxt(summary_path, np.array(rows, dtype=object), delimiter=",", fmt="%s", header=header, comments="")

    peak_noisy_mae = float(np.mean([float(r[4]) for r in rows]))
    peak_pred_mae = float(np.mean([float(r[5]) for r in rows]))
    rmse_noisy_mean = float(np.mean([float(r[6]) for r in rows]))
    rmse_pred_mean = float(np.mean([float(r[7]) for r in rows]))

    report_path = output_dir / "report.txt"
    report_path.write_text(
        "\n".join(
            [
                f"samples={len(rows)}",
                f"peak_mae_noisy_mm={peak_noisy_mae:.4f}",
                f"peak_mae_pred_mm={peak_pred_mae:.4f}",
                f"pdd_rmse_noisy_mean={rmse_noisy_mean:.4f}",
                f"pdd_rmse_pred_mean={rmse_pred_mean:.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved summary: {summary_path}")
    print(f"Saved report: {report_path}")
    print(f"Residual learning: {residual_learning}")
    print(f"Peak MAE noisy [mm]: {peak_noisy_mae:.4f}")
    print(f"Peak MAE pred  [mm]: {peak_pred_mae:.4f}")
    print(f"PDD RMSE noisy mean: {rmse_noisy_mean:.4f}")
    print(f"PDD RMSE pred  mean: {rmse_pred_mean:.4f}")


if __name__ == "__main__":
    main()
