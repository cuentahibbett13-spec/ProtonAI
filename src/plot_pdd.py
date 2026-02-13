import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_profile(volume: np.ndarray, axis: str, center_y: int, center_x: int, window: int) -> np.ndarray:
    if axis == "z":
        y0, x0 = center_y, center_x
        y1, y2 = max(0, y0 - window), min(volume.shape[1], y0 + window + 1)
        x1, x2 = max(0, x0 - window), min(volume.shape[2], x0 + window + 1)
        return volume[:, y1:y2, x1:x2].mean(axis=(1, 2))

    if axis == "y":
        z0, x0 = center_y, center_x
        z1, z2 = max(0, z0 - window), min(volume.shape[0], z0 + window + 1)
        x1, x2 = max(0, x0 - window), min(volume.shape[2], x0 + window + 1)
        return volume[z1:z2, :, x1:x2].mean(axis=(0, 2))

    z0, y0 = center_y, center_x
    z1, z2 = max(0, z0 - window), min(volume.shape[0], z0 + window + 1)
    y1, y2 = max(0, y0 - window), min(volume.shape[1], y0 + window + 1)
    return volume[z1:z2, y1:y2, :].mean(axis=(0, 1))


def normalize_to_pdd(profile: np.ndarray) -> np.ndarray:
    peak = float(profile.max())
    if peak <= 0.0:
        return np.zeros_like(profile)
    return (profile / peak) * 100.0


def axis_spacing_mm(spacing_xyz: np.ndarray, axis: str) -> float:
    if axis == "x":
        return float(spacing_xyz[0])
    if axis == "y":
        return float(spacing_xyz[1])
    return float(spacing_xyz[2])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot PDD from NPZ sample (easy mode)")
    parser.add_argument("--npz", type=str, required=True, help="Path to sample .npz")
    parser.add_argument("--output-dir", type=str, default="outputs/pdd", help="Output directory")
    parser.add_argument("--axis", type=str, choices=["z", "y", "x"], default="z", help="Depth axis")
    parser.add_argument("--window", type=int, default=1, help="Averaging half-window around central axis")
    parser.add_argument("--center-y", type=int, default=None, help="Center index (auto if omitted)")
    parser.add_argument("--center-x", type=int, default=None, help="Center index (auto if omitted)")
    parser.add_argument("--noisy-key", type=str, default="noisy_dose")
    parser.add_argument("--target-key", type=str, default="target_dose")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    data = np.load(args.npz)
    noisy = data[args.noisy_key].astype(np.float32)
    target = data[args.target_key].astype(np.float32)
    spacing = data.get("spacing", np.array([1.0, 1.0, 1.0], dtype=np.float32)).astype(np.float32)

    if noisy.shape != target.shape:
        raise ValueError(f"Shape mismatch: noisy={noisy.shape}, target={target.shape}")

    if args.axis == "z":
        cy = noisy.shape[1] // 2 if args.center_y is None else args.center_y
        cx = noisy.shape[2] // 2 if args.center_x is None else args.center_x
    elif args.axis == "y":
        cy = noisy.shape[0] // 2 if args.center_y is None else args.center_y
        cx = noisy.shape[2] // 2 if args.center_x is None else args.center_x
    else:
        cy = noisy.shape[0] // 2 if args.center_y is None else args.center_y
        cx = noisy.shape[1] // 2 if args.center_x is None else args.center_x

    noisy_profile = extract_profile(noisy, args.axis, cy, cx, args.window)
    target_profile = extract_profile(target, args.axis, cy, cx, args.window)

    noisy_pdd = normalize_to_pdd(noisy_profile)
    target_pdd = normalize_to_pdd(target_profile)

    step_mm = axis_spacing_mm(spacing, args.axis)
    depth_mm = np.arange(noisy_pdd.shape[0], dtype=np.float32) * step_mm

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "pdd.csv"
    np.savetxt(
        csv_path,
        np.column_stack([depth_mm, noisy_pdd, target_pdd]),
        delimiter=",",
        header="depth_mm,noisy_pdd_percent,target_pdd_percent",
        comments="",
    )

    png_path = output_dir / "pdd.png"
    plt.figure(figsize=(8, 5))
    plt.plot(depth_mm, noisy_pdd, label="Noisy (normalized)", linewidth=2)
    plt.plot(depth_mm, target_pdd, label="Target (normalized)", linewidth=2)
    plt.xlabel("Depth [mm]")
    plt.ylabel("PDD [%]")
    plt.title("Percent Depth Dose (PDD)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    noisy_peak_mm = float(depth_mm[int(np.argmax(noisy_pdd))])
    target_peak_mm = float(depth_mm[int(np.argmax(target_pdd))])

    print(f"Saved CSV: {csv_path}")
    print(f"Saved PNG: {png_path}")
    print(f"Noisy peak depth [mm]: {noisy_peak_mm:.2f}")
    print(f"Target peak depth [mm]: {target_peak_mm:.2f}")


if __name__ == "__main__":
    main()
