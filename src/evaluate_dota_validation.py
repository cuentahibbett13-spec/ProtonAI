import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model_dota import DoTAModel


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


def parse_crop_shape(shape_value) -> Optional[Tuple[int, int, int]]:
    if shape_value is None:
        return None
    if isinstance(shape_value, (tuple, list)) and len(shape_value) == 3:
        return (int(shape_value[0]), int(shape_value[1]), int(shape_value[2]))
    return None


def center_crop_3d(volume: np.ndarray, crop_shape: Tuple[int, int, int]) -> np.ndarray:
    d, h, w = volume.shape
    cd, ch, cw = crop_shape

    if cd > d or ch > h or cw > w:
        raise ValueError(f"Crop {crop_shape} is larger than volume {volume.shape}")

    d0 = (d - cd) // 2
    h0 = (h - ch) // 2
    w0 = (w - cw) // 2

    return volume[d0:d0 + cd, h0:h0 + ch, w0:w0 + cw]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DoTA validation NPZ files")
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/dota_eval")
    parser.add_argument("--axis", type=str, choices=["z", "y", "x"], default="z")
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--default-energy-mev", type=float, default=150.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_rocm_runtime_dirs()

    val_dir = Path(args.val_dir)
    files = sorted(val_dir.glob("*.npz"))
    if not files:
        raise ValueError(f"No .npz files found in {val_dir}")

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = DoTAModel(
        in_channels=1,
        feat_channels=int(checkpoint.get("feat_channels", 32)),
        d_model=int(checkpoint.get("d_model", 128)),
        nhead=int(checkpoint.get("nhead", 8)),
        num_layers=int(checkpoint.get("num_layers", 4)),
        ff_dim=int(checkpoint.get("ff_dim", 256)),
        dropout=float(checkpoint.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    crop_shape = parse_crop_shape(checkpoint.get("crop_shape", None))
    default_energy = float(checkpoint.get("default_energy_mev", args.default_energy_mev))

    rows = []

    with torch.no_grad():
        for npz_path in files:
            data = np.load(npz_path)
            target = data["target_dose"].astype(np.float32)
            density = data["density"].astype(np.float32)
            spacing = data.get("spacing", np.array([1.0, 1.0, 1.0], dtype=np.float32)).astype(np.float32)

            if crop_shape is not None:
                target = center_crop_3d(target, crop_shape)
                density = center_crop_3d(density, crop_shape)

            max_target = float(target.max())
            target_norm = target / max_target if max_target > 0 else target

            ct_scale = max(float(np.max(np.abs(density))), 1.0)
            ct_like = density / ct_scale

            if "energy_mev" in data:
                energy_mev = float(np.array(data["energy_mev"]).reshape(-1)[0])
            else:
                energy_mev = default_energy

            ct_t = torch.from_numpy(ct_like).unsqueeze(0).unsqueeze(0).to(device)
            energy_t = torch.tensor([[energy_mev]], dtype=torch.float32, device=device)

            pred_norm = model(ct_t, energy_t).squeeze(0).squeeze(0).cpu().numpy()
            pred_norm = np.clip(pred_norm, a_min=0.0, a_max=None)

            if args.axis == "z":
                c_a = target_norm.shape[1] // 2
                c_b = target_norm.shape[2] // 2
            elif args.axis == "y":
                c_a = target_norm.shape[0] // 2
                c_b = target_norm.shape[2] // 2
            else:
                c_a = target_norm.shape[0] // 2
                c_b = target_norm.shape[1] // 2

            pdd_pred = normalize_pdd(extract_profile(pred_norm, args.axis, c_a, c_b, args.window))
            pdd_target = normalize_pdd(extract_profile(target_norm, args.axis, c_a, c_b, args.window))

            step_mm = spacing_for_axis(spacing, args.axis)
            depth_mm = np.arange(pdd_target.shape[0], dtype=np.float32) * step_mm

            peak_pred_idx = int(np.argmax(pdd_pred))
            peak_target_idx = int(np.argmax(pdd_target))

            peak_pred_mm = float(depth_mm[peak_pred_idx])
            peak_target_mm = float(depth_mm[peak_target_idx])

            peak_error_pred_mm = abs(peak_pred_mm - peak_target_mm)
            pdd_rmse_pred = float(np.sqrt(np.mean((pdd_pred - pdd_target) ** 2)))

            rows.append([
                npz_path.name,
                energy_mev,
                peak_pred_mm,
                peak_target_mm,
                peak_error_pred_mm,
                pdd_rmse_pred,
            ])

            fig_path = plots_dir / f"{npz_path.stem}_pdd.png"
            plt.figure(figsize=(8, 5))
            plt.plot(depth_mm, pdd_pred, label="Pred", linewidth=1.8)
            plt.plot(depth_mm, pdd_target, label="Target", linewidth=1.8)
            plt.xlabel("Depth [mm]")
            plt.ylabel("PDD [%]")
            plt.title(f"{npz_path.stem} | E={energy_mev:.1f} MeV")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()

    summary_path = output_dir / "summary.csv"
    header = "sample,energy_mev,peak_pred_mm,peak_target_mm,peak_error_pred_mm,pdd_rmse_pred"
    np.savetxt(summary_path, np.array(rows, dtype=object), delimiter=",", fmt="%s", header=header, comments="")

    peak_pred_mae = float(np.mean([float(r[4]) for r in rows]))
    rmse_pred_mean = float(np.mean([float(r[5]) for r in rows]))

    energy_values = sorted(set(float(r[1]) for r in rows))
    per_energy_rows = []
    for energy in energy_values:
        subset = [r for r in rows if abs(float(r[1]) - energy) < 1e-6]
        mae = float(np.mean([float(r[4]) for r in subset]))
        rmse = float(np.mean([float(r[5]) for r in subset]))
        per_energy_rows.append([energy, len(subset), mae, rmse])

    per_energy_path = output_dir / "per_energy.csv"
    per_energy_header = "energy_mev,count,peak_mae_pred_mm,pdd_rmse_pred_mean"
    np.savetxt(per_energy_path, np.array(per_energy_rows, dtype=object), delimiter=",", fmt="%s", header=per_energy_header, comments="")

    report_path = output_dir / "report.txt"
    report_path.write_text(
        "\n".join(
            [
                f"samples={len(rows)}",
                f"peak_mae_pred_mm={peak_pred_mae:.4f}",
                f"pdd_rmse_pred_mean={rmse_pred_mean:.4f}",
                f"per_energy_csv={per_energy_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved summary: {summary_path}")
    print(f"Saved per-energy summary: {per_energy_path}")
    print(f"Saved report: {report_path}")
    print(f"Peak MAE pred [mm]: {peak_pred_mae:.4f}")
    print(f"PDD RMSE pred mean: {rmse_pred_mean:.4f}")


if __name__ == "__main__":
    main()
