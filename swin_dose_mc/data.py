from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def crop_or_pad_3d(volume: np.ndarray, crop_shape: Tuple[int, int, int], random_crop: bool) -> np.ndarray:
    z, y, x = volume.shape
    cz, cy, cx = crop_shape

    pad_z = max(0, cz - z)
    pad_y = max(0, cy - y)
    pad_x = max(0, cx - x)
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        volume = np.pad(volume, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant", constant_values=0)
        z, y, x = volume.shape

    if random_crop:
        z0 = np.random.randint(0, z - cz + 1)
        y0 = np.random.randint(0, y - cy + 1)
        x0 = np.random.randint(0, x - cx + 1)
    else:
        z0 = (z - cz) // 2
        y0 = (y - cy) // 2
        x0 = (x - cx) // 2

    return volume[z0 : z0 + cz, y0 : y0 + cy, x0 : x0 + cx]


class DoseNPZDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        crop_shape: Tuple[int, int, int],
        random_crop: bool,
        include_energy: bool = True,
        energy_min_mev: float = 70.0,
        energy_max_mev: float = 250.0,
        max_samples: int = 0,
    ):
        files = sorted(data_dir.glob("*.npz"))
        if max_samples > 0:
            files = files[:max_samples]
        if not files:
            raise ValueError(f"No NPZ files found in {data_dir}")

        self.files: List[Path] = files
        self.crop_shape = crop_shape
        self.random_crop = random_crop
        self.include_energy = include_energy
        self.energy_min_mev = energy_min_mev
        self.energy_max_mev = energy_max_mev

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        npz = np.load(self.files[idx])

        noisy = npz["noisy_dose"].astype(np.float32)
        target = npz["target_dose"].astype(np.float32)
        density = npz["density"].astype(np.float32)

        noisy = crop_or_pad_3d(noisy, self.crop_shape, self.random_crop)
        target = crop_or_pad_3d(target, self.crop_shape, self.random_crop)
        density = crop_or_pad_3d(density, self.crop_shape, self.random_crop)

        target_max = max(float(target.max()), 1e-6)
        density_max = max(float(density.max()), 1e-6)

        noisy = noisy / target_max
        target = target / target_max
        density = density / density_max

        x = np.stack([noisy, density], axis=0).astype(np.float32)
        y = target[np.newaxis, ...].astype(np.float32)

        energy_mev = float(np.array(npz.get("energy_mev", 150.0)).reshape(-1)[0])
        energy_norm = (energy_mev - self.energy_min_mev) / max(self.energy_max_mev - self.energy_min_mev, 1e-6)
        energy_norm = float(np.clip(energy_norm, 0.0, 1.0))

        batch: Dict[str, torch.Tensor] = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "target_max": torch.tensor(target_max, dtype=torch.float32),
            "energy": torch.tensor(energy_norm if self.include_energy else 0.0, dtype=torch.float32),
        }
        return batch
