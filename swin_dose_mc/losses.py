from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def gradient_loss_3d(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dz = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    pred_dy = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    pred_dx = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]

    targ_dz = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
    targ_dy = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
    targ_dx = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]

    return (F.l1_loss(pred_dz, targ_dz) + F.l1_loss(pred_dy, targ_dy) + F.l1_loss(pred_dx, targ_dx)) / 3.0


def physics_informed_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 0.5,
    target_threshold: float = 0.7,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    l_global = F.mse_loss(pred, target)

    target_peak = target.amax(dim=(2, 3, 4), keepdim=True)
    mask = target >= (target_threshold * target_peak)
    if torch.any(mask):
        l_target = torch.mean((pred[mask] - target[mask]) ** 2)
    else:
        l_target = l_global

    l_grad = gradient_loss_3d(pred, target)

    total = l_global + alpha * l_target + beta * l_grad
    metrics = {
        "loss": float(total.detach().item()),
        "loss_global": float(l_global.detach().item()),
        "loss_target": float(l_target.detach().item()),
        "loss_grad": float(l_grad.detach().item()),
    }
    return total, metrics
