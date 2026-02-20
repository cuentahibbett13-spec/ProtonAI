#!/usr/bin/env python
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

from swin_dose_mc.data import crop_or_pad_3d
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


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def extract_central_profile(volume: np.ndarray, axis: str = "z") -> np.ndarray:
    if axis == "z":
        return volume[:, volume.shape[1] // 2, volume.shape[2] // 2]
    if axis == "y":
        return volume[volume.shape[0] // 2, :, volume.shape[2] // 2]
    return volume[volume.shape[0] // 2, volume.shape[1] // 2, :]


def gamma_1d_pass_rate(ref: np.ndarray, eva: np.ndarray, step_mm: float, dose_pct: float, dist_mm: float) -> float:
    ref_max = max(float(np.max(ref)), 1e-6)
    dose_crit = (dose_pct / 100.0) * ref_max
    n = ref.shape[0]
    passed = 0
    idx = np.arange(n, dtype=np.float32)
    for i in range(n):
        dd = np.abs(ref[i] - eva) / max(dose_crit, 1e-6)
        dt = np.abs(idx - i) * step_mm / max(dist_mm, 1e-6)
        gamma = np.sqrt(dd**2 + dt**2)
        if np.min(gamma) <= 1.0:
            passed += 1
    return 100.0 * passed / max(n, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Swin-Dose MC checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--val-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs/swin_dose_mc_eval")
    p.add_argument("--max-samples", type=int, default=10)
    p.add_argument("--axis", type=str, choices=["z", "y", "x"], default="z")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_rocm_runtime_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    crop_shape = tuple(ckpt.get("crop_shape", (96, 96, 96)))
    energy_min = float(ckpt.get("energy_min", 70.0))
    energy_max = float(ckpt.get("energy_max", 250.0))
    feature_size = int(ckpt.get("feature_size", 24))

    model = SwinDoseMC(
        img_size=crop_shape,
        in_channels=2,
        out_channels=1,
        feature_size=feature_size,
        use_energy_token=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.val_dir).glob("*.npz"))
    if args.max_samples > 0:
        files = files[: args.max_samples]
    if not files:
        raise ValueError(f"No NPZ files found in {args.val_dir}")

    rows = []
    with torch.no_grad():
        for f in files:
            d = np.load(f)
            noisy = d["noisy_dose"].astype(np.float32)
            target = d["target_dose"].astype(np.float32)
            density = d["density"].astype(np.float32)
            spacing = d.get("spacing", np.array([1.0, 1.0, 1.0], dtype=np.float32)).astype(np.float32)
            energy_mev = float(np.array(d.get("energy_mev", 150.0)).reshape(-1)[0])

            noisy = crop_or_pad_3d(noisy, crop_shape, random_crop=False)
            target = crop_or_pad_3d(target, crop_shape, random_crop=False)
            density = crop_or_pad_3d(density, crop_shape, random_crop=False)

            target_max = max(float(target.max()), 1e-6)
            density_max = max(float(density.max()), 1e-6)
            energy = np.clip((energy_mev - energy_min) / max(energy_max - energy_min, 1e-6), 0.0, 1.0)

            x = np.stack([noisy / target_max, density / density_max], axis=0)[None, ...].astype(np.float32)
            e = np.array([energy], dtype=np.float32)
            x_t = torch.from_numpy(x).to(device)
            e_t = torch.from_numpy(e).to(device)

            pred_norm = model(x_t, e_t).squeeze(0).squeeze(0).cpu().numpy()
            pred = pred_norm * target_max

            hd_mask = target >= (0.1 * target_max)
            rmse_noisy = rmse(noisy, target)
            rmse_pred = rmse(pred, target)
            mae_noisy = mae(noisy, target)
            mae_pred = mae(pred, target)
            rmse_hd = rmse(pred[hd_mask], target[hd_mask]) if np.any(hd_mask) else 0.0

            ref_prof = extract_central_profile(target, axis=args.axis)
            pred_prof = extract_central_profile(pred, axis=args.axis)
            if args.axis == "x":
                step_mm = float(spacing[0])
            elif args.axis == "y":
                step_mm = float(spacing[1])
            else:
                step_mm = float(spacing[2])

            gamma_3_3 = gamma_1d_pass_rate(ref_prof, pred_prof, step_mm=step_mm, dose_pct=3.0, dist_mm=3.0)
            gamma_2_2 = gamma_1d_pass_rate(ref_prof, pred_prof, step_mm=step_mm, dose_pct=2.0, dist_mm=2.0)

            rows.append(
                {
                    "sample": f.name,
                    "rmse_noisy": rmse_noisy,
                    "rmse_pred": rmse_pred,
                    "rmse_improve_pct": 100.0 * (rmse_noisy - rmse_pred) / max(rmse_noisy, 1e-6),
                    "mae_noisy": mae_noisy,
                    "mae_pred": mae_pred,
                    "rmse_hd_pred": rmse_hd,
                    "gamma_3_3_pass_rate": gamma_3_3,
                    "gamma_2_2_pass_rate": gamma_2_2,
                }
            )

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "samples": len(rows),
        "rmse_noisy_mean": float(np.mean([r["rmse_noisy"] for r in rows])),
        "rmse_pred_mean": float(np.mean([r["rmse_pred"] for r in rows])),
        "rmse_improve_pct_mean": float(np.mean([r["rmse_improve_pct"] for r in rows])),
        "mae_noisy_mean": float(np.mean([r["mae_noisy"] for r in rows])),
        "mae_pred_mean": float(np.mean([r["mae_pred"] for r in rows])),
        "gamma_3_3_pass_rate_mean": float(np.mean([r["gamma_3_3_pass_rate"] for r in rows])),
        "gamma_2_2_pass_rate_mean": float(np.mean([r["gamma_2_2_pass_rate"] for r in rows])),
    }

    report_path = out_dir / "report.json"
    report_path.write_text(str(report), encoding="utf-8")
    print(report)
    print(f"Saved: {csv_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
