import argparse
import subprocess
import sys


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 1 homogeneous curriculum: phantom + conservative dataset")
    parser.add_argument("--train-cases", type=int, default=3)
    parser.add_argument("--val-cases", type=int, default=1)
    parser.add_argument("--noisy-primaries", type=int, default=20_000)
    parser.add_argument("--target-primaries", type=int, default=200_000)
    parser.add_argument("--phantom-dir", type=str, default="data/phantom")
    parser.add_argument("--gate-output-root", type=str, default="data/gate/stage1_homogeneous")
    parser.add_argument("--dataset-root", type=str, default="data/dataset_stage1_homogeneous")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    run_command(
        [
            sys.executable,
            "-m",
            "src.generate_sandwich_phantom",
            "--mode",
            "homogeneous",
            "--output-dir",
            args.phantom_dir,
        ]
    )

    run_command(
        [
            sys.executable,
            "-m",
            "src.generate_conservative_dataset",
            "--train-cases",
            str(args.train_cases),
            "--val-cases",
            str(args.val_cases),
            "--noisy-primaries",
            str(args.noisy_primaries),
            "--target-primaries",
            str(args.target_primaries),
            "--phantom-mhd",
            f"{args.phantom_dir}/homogeneous_labels.mhd",
            "--labels-to-materials",
            f"{args.phantom_dir}/labels_to_materials_homogeneous.txt",
            "--density-map-mhd",
            f"{args.phantom_dir}/homogeneous_density.mhd",
            "--gate-output-root",
            args.gate_output_root,
            "--dataset-root",
            args.dataset_root,
        ]
    )

    print("Stage 1 homogeneous dataset ready.")


if __name__ == "__main__":
    main()
