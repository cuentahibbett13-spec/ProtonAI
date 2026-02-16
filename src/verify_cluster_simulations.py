import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np


def parse_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_case_files(split: str, case_name: str, gate_root: Path, dataset_root: Path) -> dict[str, bool]:
    gate_split = gate_root / split
    ds_split = dataset_root / split

    paths = {
        "noisy_mhd": gate_split / f"{case_name}_noisy.mhd",
        "noisy_raw": gate_split / f"{case_name}_noisy.raw",
        "target_mhd": gate_split / f"{case_name}_target.mhd",
        "target_raw": gate_split / f"{case_name}_target.raw",
        "density_mhd": gate_split / f"{case_name}_density.mhd",
        "density_raw": gate_split / f"{case_name}_density.raw",
        "npz": ds_split / f"{case_name}.npz",
    }

    return {k: p.exists() for k, p in paths.items()}


def validate_npz(npz_path: Path, expected_energy_mev: Optional[float]) -> tuple[bool, str]:
    required_keys = {"noisy_dose", "target_dose", "density", "spacing"}
    try:
        data = np.load(npz_path)
    except Exception as exc:
        return False, f"load_error={exc}"

    missing = sorted(required_keys - set(data.files))
    if missing:
        return False, f"missing_keys={missing}"

    noisy = data["noisy_dose"]
    target = data["target_dose"]
    density = data["density"]
    spacing = data["spacing"]

    if noisy.shape != target.shape or noisy.shape != density.shape:
        return False, f"shape_mismatch noisy={noisy.shape} target={target.shape} density={density.shape}"

    if len(noisy.shape) != 3:
        return False, f"invalid_dim={noisy.shape}"

    if np.any(~np.isfinite(noisy)) or np.any(~np.isfinite(target)) or np.any(~np.isfinite(density)):
        return False, "non_finite_values"

    if spacing.shape[0] != 3:
        return False, f"invalid_spacing={spacing}"

    if "energy_mev" in data.files and expected_energy_mev is not None:
        got_energy = float(np.array(data["energy_mev"]).reshape(-1)[0])
        if abs(got_energy - expected_energy_mev) > 1e-3:
            return False, f"energy_mismatch expected={expected_energy_mev} got={got_energy}"

    return True, "ok"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify cluster simulation outputs against manifest")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--gate-output-root", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--strict-energy", action="store_true", help="Fail if energy mismatch or missing")
    parser.add_argument("--max-print-failures", type=int, default=20)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    manifest_path = Path(args.manifest)
    gate_root = Path(args.gate_output_root)
    dataset_root = Path(args.dataset_root)

    rows = parse_manifest(manifest_path)
    if not rows:
        raise ValueError(f"Empty manifest: {manifest_path}")

    total = len(rows)
    complete = 0
    npz_valid = 0
    failures: list[str] = []

    for row in rows:
        split = row["split"]
        case_name = row["case_name"]
        expected_energy = float(row["energy_mev"]) if "energy_mev" in row and row["energy_mev"] else None

        file_status = check_case_files(split, case_name, gate_root, dataset_root)
        if all(file_status.values()):
            complete += 1
        else:
            missing = [k for k, ok in file_status.items() if not ok]
            failures.append(f"{split}/{case_name}: missing={missing}")
            continue

        npz_path = dataset_root / split / f"{case_name}.npz"
        ok, reason = validate_npz(npz_path, expected_energy if args.strict_energy else None)
        if ok:
            npz_valid += 1
        else:
            failures.append(f"{split}/{case_name}: npz_invalid {reason}")

    print(f"Manifest: {manifest_path}")
    print(f"Total cases: {total}")
    print(f"Complete cases (all files present): {complete}/{total}")
    print(f"Valid NPZ: {npz_valid}/{total}")

    if failures:
        print(f"Failures: {len(failures)}")
        for line in failures[: args.max_print_failures]:
            print(f" - {line}")
        if len(failures) > args.max_print_failures:
            print(f" ... and {len(failures) - args.max_print_failures} more")

    if complete == total and npz_valid == total:
        print("Status: OK")
    else:
        print("Status: INCOMPLETE")


if __name__ == "__main__":
    main()
