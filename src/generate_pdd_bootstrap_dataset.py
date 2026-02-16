import argparse
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

from .convert_mhd_to_npz import convert_to_npz
from .generate_sandwich_phantom import generate_homogeneous, write_mhd_raw


MATERIAL_DENSITY = {
    "G4_WATER": 1.0,
    "CORTICAL_BONE_1850": 1.85,
    "LUNG_0200": 0.2,
}


def set_geant4_data_env(geant4_data_root: str) -> None:
    root = Path(geant4_data_root)
    mapping = {
        "G4NEUTRONHPDATA": "G4NDL4.7",
        "G4LEDATA": "G4EMLOW8.4",
        "G4LEVELGAMMADATA": "PhotonEvaporation5.7",
        "G4RADIOACTIVEDATA": "RadioactiveDecay5.6",
        "G4SAIDXSDATA": "G4SAIDDATA2.0",
        "G4PARTICLEXSDATA": "G4PARTICLEXS4.0",
        "G4ABLADATA": "G4ABLA3.3",
        "G4INCLDATA": "G4INCL1.1",
        "G4PIIDATA": "G4PII1.3",
        "G4ENSDFSTATEDATA": "G4ENSDFSTATE2.3",
        "G4REALSURFACEDATA": "RealSurface2.2",
    }
    for var_name, folder in mapping.items():
        candidate = root / folder
        if candidate.exists() and var_name not in os.environ:
            os.environ[var_name] = str(candidate)


def run_simulation_subprocess(
    phantom_mhd: str,
    labels_to_materials: str,
    material_db: str,
    output_dose: str,
    output_density: str,
    density_map_mhd: str,
    primaries: int,
    energy_mev: float,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.simulate_gate10",
        "--phantom-mhd",
        phantom_mhd,
        "--labels-to-materials",
        labels_to_materials,
        "--material-db",
        material_db,
        "--output-dose",
        output_dose,
        "--output-density",
        output_density,
        "--density-map-mhd",
        density_map_mhd,
        "--primaries",
        str(primaries),
        "--energy-mev",
        str(energy_mev),
    ]
    subprocess.run(cmd, check=True)


def write_labels_materials(path: Path, material1: str, material2: str | None = None) -> None:
    if material2 is None:
        text = "0 0 G4_WATER\n"
    else:
        text = f"0 0 {material1}\n1 1 {material2}\n"
    path.write_text(text, encoding="utf-8")


def create_homogeneous_phantom(phantom_dir: Path, size: int, voxel_mm: float) -> tuple[str, str, str]:
    labels, density = generate_homogeneous(size=size, density_value=1.0)
    labels_mhd = phantom_dir / "homogeneous_labels.mhd"
    density_mhd = phantom_dir / "homogeneous_density.mhd"
    labels_to_materials = phantom_dir / "labels_to_materials_homogeneous.txt"

    write_mhd_raw(labels, (voxel_mm, voxel_mm, voxel_mm), labels_mhd)
    write_mhd_raw(density, (voxel_mm, voxel_mm, voxel_mm), density_mhd)
    write_labels_materials(labels_to_materials, material1="G4_WATER")
    return str(labels_mhd), str(labels_to_materials), str(density_mhd)


def create_single_change_phantom(
    phantom_dir: Path,
    size: int,
    voxel_mm: float,
    z_change: int,
    second_material: str,
) -> tuple[str, str, str]:
    labels = np.zeros((size, size, size), dtype=np.uint16)
    density = np.full((size, size, size), fill_value=MATERIAL_DENSITY["G4_WATER"], dtype=np.float32)

    labels[z_change:, :, :] = 1
    density[z_change:, :, :] = MATERIAL_DENSITY[second_material]

    labels_mhd = phantom_dir / "single_change_labels.mhd"
    density_mhd = phantom_dir / "single_change_density.mhd"
    labels_to_materials = phantom_dir / "labels_to_materials_single_change.txt"

    write_mhd_raw(labels, (voxel_mm, voxel_mm, voxel_mm), labels_mhd)
    write_mhd_raw(density, (voxel_mm, voxel_mm, voxel_mm), density_mhd)
    write_labels_materials(labels_to_materials, material1="G4_WATER", material2=second_material)
    return str(labels_mhd), str(labels_to_materials), str(density_mhd)


def run_case(
    split: str,
    case_name: str,
    phantom_mhd: str,
    labels_to_materials: str,
    density_map_mhd: str,
    noisy_primaries: int,
    target_primaries: int,
    energy_mev: float,
    material_db: str,
    gate_output_root: Path,
    dataset_root: Path,
) -> None:
    gate_split = gate_output_root / split
    gate_split.mkdir(parents=True, exist_ok=True)

    noisy_mhd = gate_split / f"{case_name}_noisy.mhd"
    target_mhd = gate_split / f"{case_name}_target.mhd"
    density_mhd = gate_split / f"{case_name}_density.mhd"

    run_simulation_subprocess(
        phantom_mhd=phantom_mhd,
        labels_to_materials=labels_to_materials,
        material_db=material_db,
        output_dose=str(noisy_mhd),
        output_density=str(density_mhd),
        density_map_mhd=density_map_mhd,
        primaries=noisy_primaries,
        energy_mev=energy_mev,
    )

    run_simulation_subprocess(
        phantom_mhd=phantom_mhd,
        labels_to_materials=labels_to_materials,
        material_db=material_db,
        output_dose=str(target_mhd),
        output_density=str(density_mhd),
        density_map_mhd=density_map_mhd,
        primaries=target_primaries,
        energy_mev=energy_mev,
    )

    npz_split = dataset_root / split
    npz_split.mkdir(parents=True, exist_ok=True)
    out_npz = npz_split / f"{case_name}.npz"
    convert_to_npz(
        noisy_mhd=str(noisy_mhd),
        target_mhd=str(target_mhd),
        density_mhd=str(density_mhd),
        output_npz=str(out_npz),
        energy_mev=energy_mev,
    )
    print(f"[{split}] {case_name} (energy={energy_mev:.1f} MeV) -> {out_npz}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate mixed dataset: homogeneous + single material change")
    parser.add_argument("--train-hom-cases", type=int, default=8)
    parser.add_argument("--val-hom-cases", type=int, default=2)
    parser.add_argument("--train-change-cases", type=int, default=12)
    parser.add_argument("--val-change-cases", type=int, default=3)
    parser.add_argument("--noisy-primaries", type=int, default=20_000)
    parser.add_argument("--target-primaries", type=int, default=200_000)
    parser.add_argument("--energy-mev", type=float, default=150.0)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--voxel-mm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--change-min-ratio", type=float, default=0.35)
    parser.add_argument("--change-max-ratio", type=float, default=0.65)
    parser.add_argument("--change-materials", type=str, default="CORTICAL_BONE_1850,LUNG_0200")
    parser.add_argument("--material-db", type=str, default="gate/materials/sandwich_materials.db")
    parser.add_argument("--phantom-root", type=str, default="data/phantoms_pdd_bootstrap")
    parser.add_argument("--gate-output-root", type=str, default="data/gate/pdd_bootstrap")
    parser.add_argument("--dataset-root", type=str, default="data/dataset_pdd_bootstrap")
    parser.add_argument(
        "--geant4-data-root",
        type=str,
        default="/home/fer/geant4_install/geant4-install/share/Geant4/data",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)

    set_geant4_data_env(args.geant4_data_root)

    phantom_root = Path(args.phantom_root)
    hom_root = phantom_root / "homogeneous"
    hom_root.mkdir(parents=True, exist_ok=True)
    hom_phantom, hom_map, hom_density = create_homogeneous_phantom(
        hom_root,
        size=args.size,
        voxel_mm=args.voxel_mm,
    )

    for i in range(args.train_hom_cases):
        run_case(
            split="train",
            case_name=f"hom_{i:04d}",
            phantom_mhd=hom_phantom,
            labels_to_materials=hom_map,
            density_map_mhd=hom_density,
            noisy_primaries=args.noisy_primaries,
            target_primaries=args.target_primaries,
            energy_mev=args.energy_mev,
            material_db=args.material_db,
            gate_output_root=Path(args.gate_output_root),
            dataset_root=Path(args.dataset_root),
        )

    for i in range(args.val_hom_cases):
        run_case(
            split="val",
            case_name=f"hom_{i:04d}",
            phantom_mhd=hom_phantom,
            labels_to_materials=hom_map,
            density_map_mhd=hom_density,
            noisy_primaries=args.noisy_primaries,
            target_primaries=args.target_primaries,
            energy_mev=args.energy_mev,
            material_db=args.material_db,
            gate_output_root=Path(args.gate_output_root),
            dataset_root=Path(args.dataset_root),
        )

    materials = [m.strip() for m in args.change_materials.split(",") if m.strip()]
    if not materials:
        raise ValueError("At least one material must be provided in --change-materials")

    min_idx = int(args.size * args.change_min_ratio)
    max_idx = int(args.size * args.change_max_ratio)
    if min_idx >= max_idx:
        raise ValueError("Invalid change ratio interval")

    total_change_train = args.train_change_cases
    total_change_val = args.val_change_cases

    for i in range(total_change_train):
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

        run_case(
            split="train",
            case_name=f"chg_{i:04d}",
            phantom_mhd=labels_mhd,
            labels_to_materials=labels_map,
            density_map_mhd=density_mhd,
            noisy_primaries=args.noisy_primaries,
            target_primaries=args.target_primaries,
            energy_mev=args.energy_mev,
            material_db=args.material_db,
            gate_output_root=Path(args.gate_output_root),
            dataset_root=Path(args.dataset_root),
        )

    for i in range(total_change_val):
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

        run_case(
            split="val",
            case_name=f"chg_{i:04d}",
            phantom_mhd=labels_mhd,
            labels_to_materials=labels_map,
            density_map_mhd=density_mhd,
            noisy_primaries=args.noisy_primaries,
            target_primaries=args.target_primaries,
            energy_mev=args.energy_mev,
            material_db=args.material_db,
            gate_output_root=Path(args.gate_output_root),
            dataset_root=Path(args.dataset_root),
        )

    print("PDD bootstrap dataset generation complete.")


if __name__ == "__main__":
    main()
