import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def hu_to_spr(ct_hu: np.ndarray) -> np.ndarray:
    hu = ct_hu.astype(np.float32)
    spr = np.zeros_like(hu, dtype=np.float32)

    air = hu <= -950
    lung = (hu > -950) & (hu <= -300)
    soft = (hu > -300) & (hu <= 200)
    bone = hu > 200

    spr[air] = 0.001
    spr[lung] = 0.20 + (hu[lung] + 950.0) * (0.75 / 650.0)
    spr[soft] = 0.90 + (hu[soft] + 300.0) * (0.25 / 500.0)
    spr[bone] = 1.15 + np.minimum(hu[bone], 2000.0) * (0.85 / 2000.0)
    return np.clip(spr, 0.001, 2.0)


def extract_bev_patch(volume: np.ndarray, center_zyx: Tuple[int, int, int], patch_shape: Tuple[int, int, int]) -> np.ndarray:
    z, y, x = volume.shape
    cz, cy, cx = patch_shape
    zc, yc, xc = center_zyx

    z0 = max(0, min(z - cz, zc - cz // 2))
    y0 = max(0, min(y - cy, yc - cy // 2))
    x0 = max(0, min(x - cx, xc - cx // 2))
    return volume[z0 : z0 + cz, y0 : y0 + cy, x0 : x0 + cx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CT HU NPZ to SPR NPZ and optional BEV patch")
    parser.add_argument("--input-npz", type=str, required=True)
    parser.add_argument("--output-npz", type=str, required=True)
    parser.add_argument("--hu-key", type=str, default="ct_hu")
    parser.add_argument("--patch-shape", type=str, default="")
    parser.add_argument("--center-zyx", type=str, default="")
    args = parser.parse_args()

    data = np.load(args.input_npz)
    if args.hu_key not in data.files:
        raise KeyError(f"Missing key '{args.hu_key}' in {args.input_npz}")

    ct_hu = data[args.hu_key].astype(np.float32)
    spr = hu_to_spr(ct_hu)

    if args.patch_shape and args.center_zyx:
        patch_shape = tuple(int(v) for v in args.patch_shape.split(","))
        center_zyx = tuple(int(v) for v in args.center_zyx.split(","))
        spr = extract_bev_patch(spr, center_zyx, patch_shape)

    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"spr": spr}
    if "spacing_zyx" in data.files:
        payload["spacing_zyx"] = data["spacing_zyx"].astype(np.float32)
    np.savez_compressed(out, **payload)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
