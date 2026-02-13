import argparse
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_center_slices(volume: np.ndarray):
    zc = volume.shape[0] // 2
    yc = volume.shape[1] // 2
    xc = volume.shape[2] // 2
    return volume[zc, :, :], volume[:, yc, :], volume[:, :, xc]


def save_three_views(volume: np.ndarray, title_prefix: str, cmap: str, output_path: Path) -> None:
    axial, coronal, sagittal = get_center_slices(volume)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (img, name) in zip(
        axes,
        [(axial, "Axial"), (coronal, "Coronal"), (sagittal, "Sagittal")],
    ):
        im = ax.imshow(img, cmap=cmap, origin="lower")
        ax.set_title(f"{title_prefix} - {name}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize one NPZ sample (noisy, target, density)")
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/figures/npz")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    data = np.load(args.npz)

    noisy = data["noisy_dose"].astype(np.float32)
    target = data["target_dose"].astype(np.float32)
    density = data["density"].astype(np.float32)

    output_dir = Path(args.output_dir)
    save_three_views(noisy, "Noisy Dose", "inferno", output_dir / "noisy_views.png")
    save_three_views(target, "Target Dose", "inferno", output_dir / "target_views.png")
    save_three_views(density, "Density", "viridis", output_dir / "density_views.png")

    print(f"Saved: {output_dir / 'noisy_views.png'}")
    print(f"Saved: {output_dir / 'target_views.png'}")
    print(f"Saved: {output_dir / 'density_views.png'}")


if __name__ == "__main__":
    main()
