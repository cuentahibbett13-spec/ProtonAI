import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSEBraggLoss(nn.Module):
    def __init__(self, threshold_ratio: float = 0.5, high_dose_weight: float = 4.0):
        super().__init__()
        self.threshold_ratio = threshold_ratio
        self.high_dose_weight = high_dose_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch: prediction={prediction.shape}, target={target.shape}")

        max_per_sample = torch.amax(target, dim=(2, 3, 4), keepdim=True)
        threshold = self.threshold_ratio * max_per_sample
        high_dose_mask = (target > threshold).float()
        weights = 1.0 + (self.high_dose_weight - 1.0) * high_dose_mask

        squared_error = (prediction - target) ** 2
        weighted_error = squared_error * weights
        return weighted_error.mean()


class WeightedMSEWithPDDLoss(nn.Module):
    def __init__(
        self,
        threshold_ratio: float = 0.5,
        high_dose_weight: float = 4.0,
        pdd_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.base = WeightedMSEBraggLoss(
            threshold_ratio=threshold_ratio,
            high_dose_weight=high_dose_weight,
        )
        self.pdd_loss_weight = pdd_loss_weight

    @staticmethod
    def _normalize_profile(profile: torch.Tensor) -> torch.Tensor:
        peak = torch.amax(profile, dim=1, keepdim=True)
        peak = torch.clamp(peak, min=1e-8)
        return profile / peak

    def _pdd_profile_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = prediction[:, 0]
        targ = target[:, 0]

        pred_z = pred.mean(dim=(2, 3))
        targ_z = targ.mean(dim=(2, 3))

        pred_y = pred.mean(dim=(1, 3))
        targ_y = targ.mean(dim=(1, 3))

        pred_x = pred.mean(dim=(1, 2))
        targ_x = targ.mean(dim=(1, 2))

        loss_z = F.mse_loss(self._normalize_profile(pred_z), self._normalize_profile(targ_z))
        loss_y = F.mse_loss(self._normalize_profile(pred_y), self._normalize_profile(targ_y))
        loss_x = F.mse_loss(self._normalize_profile(pred_x), self._normalize_profile(targ_x))

        return (loss_z + loss_y + loss_x) / 3.0

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_loss = self.base(prediction, target)
        if self.pdd_loss_weight <= 0.0:
            return base_loss
        pdd_loss = self._pdd_profile_loss(prediction, target)
        return base_loss + self.pdd_loss_weight * pdd_loss
