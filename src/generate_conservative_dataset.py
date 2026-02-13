import argparse
import os
from pathlib import Path
import subprocess
import sys

from .convert_mhd_to_npz import convert_to_npz


def run_simulation_subprocess(
    phantom_mhd: str,
    labels_to_materials: str,
    material_db: str,
    output_dose: str,
    output_density: str,
    density_map_mhd: str,
    primaries: int,
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
    ]
    subprocess.run(cmd, check=True)


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


def run_case(
    split: str,
    case_idx: int,
    noisy_primaries: int,
    target_primaries: int,
    gate_output_root: Path,
    dataset_root: Path,
    phantom_mhd: str,
    labels_to_materials: str,
    material_db: str,
    density_map_mhd: str,
) -> None:
    case_name = f"{split}_{case_idx:04d}"
    split_gate_dir = gate_output_root / split
    split_gate_dir.mkdir(parents=True, exist_ok=True)

    noisy_mhd = split_gate_dir / f"{case_name}_noisy.mhd"
    target_mhd = split_gate_dir / f"{case_name}_target.mhd"
    density_mhd = split_gate_dir / f"{case_name}_density.mhd"

    run_simulation_subprocess(
        phantom_mhd=phantom_mhd,
        labels_to_materials=labels_to_materials,
        material_db=material_db,
        output_dose=str(noisy_mhd),
        output_density=str(density_mhd),
        density_map_mhd=density_map_mhd,
        primaries=noisy_primaries,
    )

    run_simulation_subprocess(
        phantom_mhd=phantom_mhd,
        labels_to_materials=labels_to_materials,
        material_db=material_db,
        output_dose=str(target_mhd),
        output_density=str(density_mhd),
        density_map_mhd=density_map_mhd,
        primaries=target_primaries,
    )

    split_npz_dir = dataset_root / split
    split_npz_dir.mkdir(parents=True, exist_ok=True)
    output_npz = split_npz_dir / f"{case_name}.npz"

    convert_to_npz(
        noisy_mhd=str(noisy_mhd),
        target_mhd=str(target_mhd),
        density_mhd=str(density_mhd),
        output_npz=str(output_npz),
    )

    print(f"[{split}] case {case_idx} done -> {output_npz}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate conservative train/val dataset for fast iteration")
    parser.add_argument("--train-cases", type=int, default=3)
    parser.add_argument("--val-cases", type=int, default=1)
    parser.add_argument("--noisy-primaries", type=int, default=20_000)
    parser.add_argument("--target-primaries", type=int, default=200_000)

    parser.add_argument("--phantom-mhd", type=str, default="data/phantom/sandwich_labels.mhd")
    parser.add_argument("--labels-to-materials", type=str, default="data/phantom/labels_to_materials.txt")
    parser.add_argument("--material-db", type=str, default="gate/materials/sandwich_materials.db")
    parser.add_argument("--density-map-mhd", type=str, default="data/phantom/sandwich_density.mhd")

    parser.add_argument("--gate-output-root", type=str, default="data/gate/conservative")
    parser.add_argument("--dataset-root", type=str, default="data/dataset")
    parser.add_argument(
        "--geant4-data-root",
        type=str,
        default="/home/fer/geant4_install/geant4-install/share/Geant4/data",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    set_geant4_data_env(args.geant4_data_root)

    for i in range(args.train_cases):
        run_case(
            split="train",
            case_idx=i,
            noisy_primaries=args.noisy_primaries,
            target_primaries=args.target_primaries,
            gate_output_root=Path(args.gate_output_root),
            dataset_root=Path(args.dataset_root),
            phantom_mhd=args.phantom_mhd,
            labels_to_materials=args.labels_to_materials,
            material_db=args.material_db,
            density_map_mhd=args.density_map_mhd,
        )

    for i in range(args.val_cases):
        run_case(
            split="val",
            case_idx=i,
            noisy_primaries=args.noisy_primaries,
            target_primaries=args.target_primaries,
            gate_output_root=Path(args.gate_output_root),
            dataset_root=Path(args.dataset_root),
            phantom_mhd=args.phantom_mhd,
            labels_to_materials=args.labels_to_materials,
            material_db=args.material_db,
            density_map_mhd=args.density_map_mhd,
        )

    print("Conservative dataset generation complete.")


if __name__ == "__main__":
    main()
