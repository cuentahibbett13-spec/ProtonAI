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
        augment: bool = False,
        aug_flip_prob: float = 0.0,
        aug_density_jitter_std: float = 0.0,
        aug_noisy_noise_std: float = 0.0,
        aug_max_shift_vox: int = 0,
    ):
        self.root = Path(npz_dir)
        self.files: List[Path] = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise ValueError(f"No .npz files found in {self.root}")

        self.use_density_as_ct = use_density_as_ct
        self.crop_shape = crop_shape
        self.default_energy_mev = float(default_energy_mev)
        self.augment = bool(augment)
        self.aug_flip_prob = float(max(0.0, aug_flip_prob))
        self.aug_density_jitter_std = float(max(0.0, aug_density_jitter_std))
        self.aug_noisy_noise_std = float(max(0.0, aug_noisy_noise_std))
        self.aug_max_shift_vox = int(max(0, aug_max_shift_vox))

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

    @staticmethod
    def _roll_with_zero_fill(volume: np.ndarray, shift: int, axis: int) -> np.ndarray:
        if shift == 0:
            return volume

        rolled = np.roll(volume, shift=shift, axis=axis)
        slicer = [slice(None), slice(None), slice(None)]
        if shift > 0:
            slicer[axis] = slice(0, shift)
        else:
            slicer[axis] = slice(shift, None)
        rolled[tuple(slicer)] = 0.0
        return rolled

    def _apply_geometric_augment(
        self,
        noisy_dose: np.ndarray,
        target_dose: np.ndarray,
        ct_like: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.aug_flip_prob > 0.0:
            if np.random.rand() < self.aug_flip_prob:
                noisy_dose = np.flip(noisy_dose, axis=1).copy()
                target_dose = np.flip(target_dose, axis=1).copy()
                ct_like = np.flip(ct_like, axis=1).copy()
            if np.random.rand() < self.aug_flip_prob:
                noisy_dose = np.flip(noisy_dose, axis=2).copy()
                target_dose = np.flip(target_dose, axis=2).copy()
                ct_like = np.flip(ct_like, axis=2).copy()

        if self.aug_max_shift_vox > 0:
            shift_h = int(np.random.randint(-self.aug_max_shift_vox, self.aug_max_shift_vox + 1))
            shift_w = int(np.random.randint(-self.aug_max_shift_vox, self.aug_max_shift_vox + 1))

            noisy_dose = self._roll_with_zero_fill(noisy_dose, shift_h, axis=1)
            target_dose = self._roll_with_zero_fill(target_dose, shift_h, axis=1)
            ct_like = self._roll_with_zero_fill(ct_like, shift_h, axis=1)

            noisy_dose = self._roll_with_zero_fill(noisy_dose, shift_w, axis=2)
            target_dose = self._roll_with_zero_fill(target_dose, shift_w, axis=2)
            ct_like = self._roll_with_zero_fill(ct_like, shift_w, axis=2)

        return noisy_dose, target_dose, ct_like

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

        if self.augment:
            noisy_dose, target_dose, ct_like = self._apply_geometric_augment(noisy_dose, target_dose, ct_like)

            if self.aug_density_jitter_std > 0.0:
                density_scale = 1.0 + float(np.random.normal(loc=0.0, scale=self.aug_density_jitter_std))
                density_scale = max(0.8, min(1.2, density_scale))
                ct_like = np.clip(ct_like * density_scale, a_min=0.0, a_max=None)

        max_target = float(target_dose.max())
        if max_target > 0.0:
            noisy_dose = noisy_dose / max_target
            target_dose = target_dose / max_target

        if self.augment and self.aug_noisy_noise_std > 0.0:
            noisy_dose = noisy_dose + np.random.normal(
                loc=0.0,
                scale=self.aug_noisy_noise_std,
                size=noisy_dose.shape,
            ).astype(np.float32)
            noisy_dose = np.clip(noisy_dose, a_min=0.0, a_max=None)

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
