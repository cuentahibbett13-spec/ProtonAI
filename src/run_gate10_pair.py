import argparse

from .simulate_gate10 import run_simulation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run noisy (1M) and target (50M) Gate10 simulations")
    parser.add_argument("--phantom-mhd", type=str, default="data/phantom/sandwich_labels.mhd")
    parser.add_argument("--labels-to-materials", type=str, default="data/phantom/labels_to_materials.txt")
    parser.add_argument("--material-db", type=str, default="gate/materials/sandwich_materials.db")
    parser.add_argument("--density-map-mhd", type=str, default="data/phantom/sandwich_density.mhd")
    parser.add_argument("--noisy-primaries", type=int, default=1_000_000)
    parser.add_argument("--target-primaries", type=int, default=50_000_000)
    parser.add_argument("--noisy-dose", type=str, default="data/gate/noisy_dose.mhd")
    parser.add_argument("--target-dose", type=str, default="data/gate/target_dose.mhd")
    parser.add_argument("--density", type=str, default="data/gate/density_map.mhd")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    run_simulation(
        phantom_mhd=args.phantom_mhd,
        labels_to_materials=args.labels_to_materials,
        material_db=args.material_db,
        output_dose=args.noisy_dose,
        output_density=args.density,
        density_map_mhd=args.density_map_mhd,
        primaries=args.noisy_primaries,
    )

    run_simulation(
        phantom_mhd=args.phantom_mhd,
        labels_to_materials=args.labels_to_materials,
        material_db=args.material_db,
        output_dose=args.target_dose,
        output_density=args.density,
        density_map_mhd=args.density_map_mhd,
        primaries=args.target_primaries,
    )

    print("Both simulations completed.")


if __name__ == "__main__":
    main()
