from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_dice_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    preds_prob = torch.sigmoid(predictions)
    bce = F.binary_cross_entropy_with_logits(predictions, targets)

    preds_flat = preds_prob.reshape(preds_prob.size(0), -1)
    targets_flat = targets.reshape(targets.size(0), -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    dice = 1 - ((2.0 * intersection + smooth) / (preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + smooth))

    return bce_weight * bce + (1 - bce_weight) * dice.mean()


def calculate_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def resize_logits_to_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        logits,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
