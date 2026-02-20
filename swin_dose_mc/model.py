import inspect
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from monai.networks.nets import SwinUNETR
except Exception as exc:  # pragma: no cover
    SwinUNETR = None
    _MONAI_IMPORT_ERROR = exc
else:
    _MONAI_IMPORT_ERROR = None


class SwinDoseMC(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 2,
        out_channels: int = 1,
        feature_size: int = 24,
        use_energy_token: bool = True,
    ):
        super().__init__()
        if SwinUNETR is None:
            raise ImportError(
                "MONAI is required for SwinDoseMC. Install with: pip install monai"
            ) from _MONAI_IMPORT_ERROR

        self.use_energy_token = use_energy_token
        total_in_channels = in_channels + (1 if use_energy_token else 0)
        kwargs = {
            "in_channels": total_in_channels,
            "out_channels": out_channels,
            "feature_size": feature_size,
            "use_checkpoint": False,
        }
        sig = inspect.signature(SwinUNETR.__init__)
        if "img_size" in sig.parameters:
            kwargs["img_size"] = img_size
        if "spatial_dims" in sig.parameters:
            kwargs["spatial_dims"] = 3
        self.backbone = SwinUNETR(**kwargs)

    def forward(self, x: torch.Tensor, energy: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_energy_token:
            if energy is None:
                raise ValueError("energy tensor is required when use_energy_token=True")
            if energy.ndim == 1:
                energy = energy.unsqueeze(1)
            b, _, d, h, w = x.shape
            e = energy.view(b, 1, 1, 1, 1).expand(b, 1, d, h, w)
            x = torch.cat([x, e], dim=1)
        return self.backbone(x)
