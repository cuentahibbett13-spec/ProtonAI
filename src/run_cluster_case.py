import argparse
import csv
from pathlib import Path

from .generate_pdd_bootstrap_dataset import run_case, set_geant4_data_env


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one simulation case from a manifest (for cluster arrays)")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--task-id", type=int, required=True, help="1-based row index excluding header")
    parser.add_argument("--noisy-primaries", type=int, required=True)
    parser.add_argument("--target-primaries", type=int, required=True)
    parser.add_argument("--material-db", type=str, default="gate/materials/sandwich_materials.db")
    parser.add_argument("--gate-output-root", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument(
        "--geant4-data-root",
        type=str,
        default="",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.geant4_data_root:
        set_geant4_data_env(args.geant4_data_root)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.task_id < 1 or args.task_id > len(rows):
        raise ValueError(f"task-id out of range: {args.task_id}, expected 1..{len(rows)}")

    row = rows[args.task_id - 1]
    energy_mev = float(row.get("energy_mev", 150.0))

    run_case(
        split=row["split"],
        case_name=row["case_name"],
        phantom_mhd=row["phantom_mhd"],
        labels_to_materials=row["labels_to_materials"],
        density_map_mhd=row["density_map_mhd"],
        noisy_primaries=args.noisy_primaries,
        target_primaries=args.target_primaries,
        energy_mev=energy_mev,
        material_db=args.material_db,
        gate_output_root=Path(args.gate_output_root),
        dataset_root=Path(args.dataset_root),
    )

    print(
        f"Completed task-id={args.task_id}, case={row['case_name']}, split={row['split']}, energy_mev={energy_mev}"
    )


if __name__ == "__main__":
    main()
