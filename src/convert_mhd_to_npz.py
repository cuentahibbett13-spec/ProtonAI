import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def parse_mhd_header(mhd_path: Path) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    with mhd_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def get_numpy_dtype(element_type: str) -> np.dtype:
    mapping = {
        "MET_FLOAT": np.float32,
        "MET_DOUBLE": np.float64,
        "MET_SHORT": np.int16,
        "MET_USHORT": np.uint16,
        "MET_CHAR": np.int8,
        "MET_UCHAR": np.uint8,
        "MET_INT": np.int32,
        "MET_UINT": np.uint32,
    }
    if element_type not in mapping:
        raise ValueError(f"Unsupported ElementType: {element_type}")
    return mapping[element_type]


def read_mhd_volume(mhd_path: str) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(mhd_path)
    metadata = parse_mhd_header(path)

    dim_size = [int(v) for v in metadata["DimSize"].split()]
    if len(dim_size) != 3:
        raise ValueError(f"Only 3D volumes are supported. DimSize={dim_size}")

    spacing_values = metadata.get("ElementSpacing", "1 1 1").split()
    spacing = np.array([float(v) for v in spacing_values], dtype=np.float32)

    dtype = get_numpy_dtype(metadata["ElementType"])
    raw_file = metadata["ElementDataFile"]
    raw_path = path.parent / raw_file

    expected_size = dim_size[0] * dim_size[1] * dim_size[2]
    buffer = np.fromfile(raw_path, dtype=dtype)
    if buffer.size != expected_size:
        raise ValueError(
            f"Raw size mismatch for {raw_path}: expected {expected_size}, got {buffer.size}"
        )

    volume = buffer.reshape((dim_size[2], dim_size[1], dim_size[0]))
    return volume.astype(np.float32), spacing


def convert_to_npz(
    noisy_mhd: str,
    target_mhd: str,
    density_mhd: str,
    output_npz: str,
    energy_mev: Optional[float] = None,
) -> None:
    noisy_dose, spacing_noisy = read_mhd_volume(noisy_mhd)
    target_dose, spacing_target = read_mhd_volume(target_mhd)
    density, spacing_density = read_mhd_volume(density_mhd)

    if noisy_dose.shape != target_dose.shape or noisy_dose.shape != density.shape:
        raise ValueError(
            f"Shape mismatch: noisy={noisy_dose.shape}, target={target_dose.shape}, density={density.shape}"
        )

    if not (np.allclose(spacing_noisy, spacing_target) and np.allclose(spacing_noisy, spacing_density)):
        raise ValueError(
            f"Spacing mismatch: noisy={spacing_noisy}, target={spacing_target}, density={spacing_density}"
        )

    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "noisy_dose": noisy_dose,
        "target_dose": target_dose,
        "density": density,
        "spacing": spacing_noisy,
    }
    if energy_mev is not None:
        payload["energy_mev"] = np.array([float(energy_mev)], dtype=np.float32)

    np.savez_compressed(output_path, **payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert GATE MHD/RAW outputs to NPZ")
    parser.add_argument("--noisy-dose-mhd", required=True, type=str)
    parser.add_argument("--target-dose-mhd", required=True, type=str)
    parser.add_argument("--density-mhd", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--energy-mev", type=float, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    convert_to_npz(
        noisy_mhd=args.noisy_dose_mhd,
        target_mhd=args.target_dose_mhd,
        density_mhd=args.density_mhd,
        output_npz=args.output,
        energy_mev=args.energy_mev,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
