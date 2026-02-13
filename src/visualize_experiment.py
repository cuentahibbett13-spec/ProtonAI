import argparse
from pathlib import Path

import matplotlib
import numpy as np

from .convert_mhd_to_npz import read_mhd_volume


def get_center_slices(volume: np.ndarray):
    zc = volume.shape[0] // 2
    yc = volume.shape[1] // 2
    xc = volume.shape[2] // 2

    axial = volume[zc, :, :]
    coronal = volume[:, yc, :]
    sagittal = volume[:, :, xc]
    return axial, coronal, sagittal


def plot_three_views(volume: np.ndarray, title_prefix: str, cmap: str, output_png: Path):
    import matplotlib.pyplot as plt

    axial, coronal, sagittal = get_center_slices(volume)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    views = [
        (axial, f"{title_prefix} - Axial"),
        (coronal, f"{title_prefix} - Coronal"),
        (sagittal, f"{title_prefix} - Sagittal"),
    ]

    for ax, (img, name) in zip(axes, views):
        im = ax.imshow(img, cmap=cmap, origin="lower")
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize MHD experiment outputs")
    parser.add_argument("--dose-mhd", type=str, required=True)
    parser.add_argument("--density-mhd", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    parser.add_argument("--show", action="store_true", help="Show interactive windows")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    dose, _ = read_mhd_volume(args.dose_mhd)
    density, _ = read_mhd_volume(args.density_mhd)

    output_dir = Path(args.output_dir)
    dose_png = output_dir / "dose_views.png"
    density_png = output_dir / "density_views.png"

    plot_three_views(dose, "Dose", "inferno", dose_png)
    plot_three_views(density, "Density", "viridis", density_png)

    print(f"Saved: {dose_png}")
    print(f"Saved: {density_png}")

    if args.show:
        dose_axial, dose_coronal, dose_sagittal = get_center_slices(dose)
        density_axial, density_coronal, density_sagittal = get_center_slices(density)

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        dose_views = [dose_axial, dose_coronal, dose_sagittal]
        density_views = [density_axial, density_coronal, density_sagittal]
        names = ["Axial", "Coronal", "Sagittal"]

        for i in range(3):
            im1 = axes[0, i].imshow(dose_views[i], cmap="inferno", origin="lower")
            axes[0, i].set_title(f"Dose {names[i]}")
            plt.colorbar(im1, ax=axes[0, i], shrink=0.8)

            im2 = axes[1, i].imshow(density_views[i], cmap="viridis", origin="lower")
            axes[1, i].set_title(f"Density {names[i]}")
            plt.colorbar(im2, ax=axes[1, i], shrink=0.8)

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
