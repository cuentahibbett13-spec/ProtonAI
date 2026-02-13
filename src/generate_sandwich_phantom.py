import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def write_mhd_raw(volume: np.ndarray, spacing_xyz: Tuple[float, float, float], mhd_path: Path) -> None:
    raw_path = mhd_path.with_suffix(".raw")
    mhd_path.parent.mkdir(parents=True, exist_ok=True)

    volume.tofile(raw_path)

    if volume.dtype == np.float32:
        element_type = "MET_FLOAT"
    elif volume.dtype == np.uint16:
        element_type = "MET_USHORT"
    else:
        raise ValueError(f"Unsupported dtype for MHD writing: {volume.dtype}")

    depth, height, width = volume.shape
    spacing_x, spacing_y, spacing_z = spacing_xyz

    header = "\n".join(
        [
            "ObjectType = Image",
            "NDims = 3",
            "BinaryData = True",
            "BinaryDataByteOrderMSB = False",
            "CompressedData = False",
            "TransformMatrix = 1 0 0 0 1 0 0 0 1",
            "Offset = 0 0 0",
            "CenterOfRotation = 0 0 0",
            "AnatomicalOrientation = RAI",
            f"ElementSpacing = {spacing_x} {spacing_y} {spacing_z}",
            f"DimSize = {width} {height} {depth}",
            f"ElementType = {element_type}",
            f"ElementDataFile = {raw_path.name}",
            "",
        ]
    )

    mhd_path.write_text(header, encoding="utf-8")


def generate_sandwich(size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((size, size, size), dtype=np.uint16)
    density = np.zeros((size, size, size), dtype=np.float32)

    z1 = size // 3
    z2 = 2 * size // 3

    labels[:z1, :, :] = 0
    labels[z1:z2, :, :] = 1
    labels[z2:, :, :] = 2

    density[:z1, :, :] = 1.0
    density[z1:z2, :, :] = 1.85
    density[z2:, :, :] = 0.2

    return labels, density


def generate_homogeneous(size: int = 128, density_value: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((size, size, size), dtype=np.uint16)
    density = np.full((size, size, size), fill_value=density_value, dtype=np.float32)
    return labels, density


def write_labels_to_materials(path: Path) -> None:
    content = "\n".join(
        [
            "0 0 G4_WATER",
            "1 1 CORTICAL_BONE_1850",
            "2 2 LUNG_0200",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def write_labels_to_materials_homogeneous(path: Path) -> None:
    content = "\n".join(
        [
            "0 0 G4_WATER",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 128^3 sandwich phantom as MHD/RAW")
    parser.add_argument("--output-dir", type=str, default="data/phantom")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--voxel-mm", type=float, default=1.0)
    parser.add_argument("--mode", type=str, choices=["sandwich", "homogeneous"], default="sandwich")
    parser.add_argument("--homogeneous-density", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "homogeneous":
        labels, density = generate_homogeneous(size=args.size, density_value=args.homogeneous_density)
        write_mhd_raw(
            labels,
            (args.voxel_mm, args.voxel_mm, args.voxel_mm),
            output_dir / "homogeneous_labels.mhd",
        )
        write_mhd_raw(
            density,
            (args.voxel_mm, args.voxel_mm, args.voxel_mm),
            output_dir / "homogeneous_density.mhd",
        )
        write_labels_to_materials_homogeneous(output_dir / "labels_to_materials_homogeneous.txt")
    else:
        labels, density = generate_sandwich(size=args.size)
        write_mhd_raw(labels, (args.voxel_mm, args.voxel_mm, args.voxel_mm), output_dir / "sandwich_labels.mhd")
        write_mhd_raw(density, (args.voxel_mm, args.voxel_mm, args.voxel_mm), output_dir / "sandwich_density.mhd")
        write_labels_to_materials(output_dir / "labels_to_materials.txt")

    print(f"Generated phantom in {output_dir}")


if __name__ == "__main__":
    main()
