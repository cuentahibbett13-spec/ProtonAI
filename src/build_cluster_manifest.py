import argparse
import csv
from pathlib import Path
import random

from .generate_pdd_bootstrap_dataset import (
    create_homogeneous_phantom,
    create_single_change_phantom,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build manifest for parallel cluster simulations")
    parser.add_argument("--train-hom-cases", type=int, default=8)
    parser.add_argument("--val-hom-cases", type=int, default=2)
    parser.add_argument("--train-change-cases", type=int, default=12)
    parser.add_argument("--val-change-cases", type=int, default=3)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--voxel-mm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--change-min-ratio", type=float, default=0.35)
    parser.add_argument("--change-max-ratio", type=float, default=0.65)
    parser.add_argument("--change-materials", type=str, default="CORTICAL_BONE_1850,LUNG_0200")
    parser.add_argument("--phantom-root", type=str, default="data/phantoms_pdd_bootstrap")
    parser.add_argument("--manifest", type=str, default="data/manifests/pdd_bootstrap_manifest.csv")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)

    phantom_root = Path(args.phantom_root)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    hom_root = phantom_root / "homogeneous"
    hom_root.mkdir(parents=True, exist_ok=True)
    hom_phantom, hom_map, hom_density = create_homogeneous_phantom(
        hom_root,
        size=args.size,
        voxel_mm=args.voxel_mm,
    )

    for i in range(args.train_hom_cases):
        rows.append([
            "train",
            f"hom_{i:04d}",
            hom_phantom,
            hom_map,
            hom_density,
        ])

    for i in range(args.val_hom_cases):
        rows.append([
            "val",
            f"hom_{i:04d}",
            hom_phantom,
            hom_map,
            hom_density,
        ])

    materials = [m.strip() for m in args.change_materials.split(",") if m.strip()]
    if not materials:
        raise ValueError("At least one material must be provided in --change-materials")

    min_idx = int(args.size * args.change_min_ratio)
    max_idx = int(args.size * args.change_max_ratio)
    if min_idx >= max_idx:
        raise ValueError("Invalid change ratio interval")

    for i in range(args.train_change_cases):
        material = materials[i % len(materials)]
        z_change = random.randint(min_idx, max_idx)
        case_phantom_dir = phantom_root / "train" / f"change_{i:04d}"
        case_phantom_dir.mkdir(parents=True, exist_ok=True)
        labels_mhd, labels_map, density_mhd = create_single_change_phantom(
            case_phantom_dir,
            size=args.size,
            voxel_mm=args.voxel_mm,
            z_change=z_change,
            second_material=material,
        )
        rows.append([
            "train",
            f"chg_{i:04d}",
            labels_mhd,
            labels_map,
            density_mhd,
        ])

    for i in range(args.val_change_cases):
        material = materials[i % len(materials)]
        z_change = random.randint(min_idx, max_idx)
        case_phantom_dir = phantom_root / "val" / f"change_{i:04d}"
        case_phantom_dir.mkdir(parents=True, exist_ok=True)
        labels_mhd, labels_map, density_mhd = create_single_change_phantom(
            case_phantom_dir,
            size=args.size,
            voxel_mm=args.voxel_mm,
            z_change=z_change,
            second_material=material,
        )
        rows.append([
            "val",
            f"chg_{i:04d}",
            labels_mhd,
            labels_map,
            density_mhd,
        ])

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "case_name", "phantom_mhd", "labels_to_materials", "density_map_mhd"])
        writer.writerows(rows)

    print(f"Manifest saved: {manifest_path}")
    print(f"Total cases: {len(rows)}")


if __name__ == "__main__":
    main()
