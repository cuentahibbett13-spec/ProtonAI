from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDoseDataset(Dataset):
    def __init__(
        self,
        npz_dir: str,
        use_density_as_ct: bool = True,
        crop_shape: Optional[Tuple[int, int, int]] = None,
        default_energy_mev: float = 150.0,
    ):
        self.root = Path(npz_dir)
        self.files: List[Path] = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise ValueError(f"No .npz files found in {self.root}")

        self.use_density_as_ct = use_density_as_ct
        self.crop_shape = crop_shape
        self.default_energy_mev = float(default_energy_mev)

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _center_crop_3d(volume: np.ndarray, crop_shape: Tuple[int, int, int]) -> np.ndarray:
        d, h, w = volume.shape
        cd, ch, cw = crop_shape

        if cd > d or ch > h or cw > w:
            raise ValueError(f"Crop {crop_shape} is larger than volume {volume.shape}")

        d0 = (d - cd) // 2
        h0 = (h - ch) // 2
        w0 = (w - cw) // 2

        return volume[d0:d0 + cd, h0:h0 + ch, w0:w0 + cw]

    def __getitem__(self, index: int):
        sample_path = self.files[index]
        data = np.load(sample_path)

        noisy_dose = data["noisy_dose"].astype(np.float32)
        target_dose = data["target_dose"].astype(np.float32)
        density = data["density"].astype(np.float32)

        if self.use_density_as_ct:
            ct_like = density
        else:
            ct_like = density

        if self.crop_shape is not None:
            noisy_dose = self._center_crop_3d(noisy_dose, self.crop_shape)
            target_dose = self._center_crop_3d(target_dose, self.crop_shape)
            ct_like = self._center_crop_3d(ct_like, self.crop_shape)

        max_target = float(target_dose.max())
        if max_target > 0.0:
            noisy_dose = noisy_dose / max_target
            target_dose = target_dose / max_target

        ct_scale = max(float(np.max(np.abs(ct_like))), 1.0)
        ct_like = ct_like / ct_scale

        if "energy_mev" in data:
            energy_mev = float(data["energy_mev"])
        else:
            energy_mev = self.default_energy_mev

        input_tensor = torch.from_numpy(np.stack([noisy_dose, ct_like], axis=0))
        target_tensor = torch.from_numpy(target_dose).unsqueeze(0)
        energy_tensor = torch.tensor([energy_mev], dtype=torch.float32)

        return input_tensor, energy_tensor, target_tensor
