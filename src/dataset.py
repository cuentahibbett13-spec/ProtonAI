from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class ProtonDoseDataset(Dataset):
    def __init__(self, npz_dir: str):
        self.root = Path(npz_dir)
        self.files: List[Path] = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise ValueError(f"No .npz files found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        sample_path = self.files[index]
        data = np.load(sample_path)

        noisy_dose = data["noisy_dose"].astype(np.float32)
        target_dose = data["target_dose"].astype(np.float32)
        density = data["density"].astype(np.float32)

        max_target = float(target_dose.max())
        if max_target > 0.0:
            noisy_dose = noisy_dose / max_target
            target_dose = target_dose / max_target

        density_scale = max(float(density.max()), 1.0)
        density = density / density_scale

        input_tensor = np.stack([noisy_dose, density], axis=0)
        target_tensor = np.expand_dims(target_dose, axis=0)

        return (
            torch.from_numpy(input_tensor),
            torch.from_numpy(target_tensor),
        )
