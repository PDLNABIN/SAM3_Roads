from __future__ import annotations

import torch
import torch.nn.functional as F
import cv2
import numpy as np


def bce_dice_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
    pos_weight: float | None = None,
    smooth: float = 1e-6,
) -> torch.Tensor:
    preds_prob = torch.sigmoid(predictions)
    if pos_weight is not None:
        pos_weight_tensor = torch.as_tensor([pos_weight], device=predictions.device, dtype=predictions.dtype)
        bce = F.binary_cross_entropy_with_logits(predictions, targets, pos_weight=pos_weight_tensor)
    else:
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


def evaluate_pixel_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Return cumulative pixel metrics for a batch of logits or probabilities."""
    probs = torch.sigmoid(predictions) if predictions.dtype.is_floating_point else predictions.float()
    predicted_mask = probs > threshold
    target_mask = targets > 0.5
    tp = (predicted_mask & target_mask).sum().item()
    fp = (predicted_mask & ~target_mask).sum().item()
    fn = (~predicted_mask & target_mask).sum().item()
    tn = (~predicted_mask & ~target_mask).sum().item()
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    return {
        "road_iou": tp / (tp + fp + fn + eps),
        "f1_score": dice,
        "dice": dice,
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "pixel_accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }


def _boundary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel_size = radius * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    foreground = mask.astype(np.uint8)
    eroded = cv2.erode(foreground, kernel, iterations=1)
    return foreground - eroded


def calculate_boundary_iou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    radius: int = 3,
    eps: float = 1e-6,
) -> float:
    """Calculate boundary IoU with a dilation tolerance, following CVPR 2021."""
    pred = (torch.sigmoid(prediction) > 0.5 if prediction.dtype.is_floating_point else prediction > 0.5)
    truth = target > 0.5
    values = []
    for pred_np, truth_np in zip(pred.detach().cpu().numpy(), truth.detach().cpu().numpy()):
        pred_boundary = _boundary_mask(np.squeeze(pred_np), radius)
        truth_boundary = _boundary_mask(np.squeeze(truth_np), radius)
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
        pred_region = cv2.dilate(pred_boundary, kernel)
        truth_region = cv2.dilate(truth_boundary, kernel)
        intersection = np.logical_and(pred_region, truth_region).sum()
        union = np.logical_or(pred_region, truth_region).sum()
        values.append(float((intersection + eps) / (union + eps)))
    return float(np.mean(values)) if values else 0.0


def calculate_relaxed_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    slack: int = 3,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Measure road precision/recall after allowing a pixel buffer around labels."""
    pred = (torch.sigmoid(prediction) > 0.5 if prediction.dtype.is_floating_point else prediction > 0.5)
    truth = target > 0.5
    kernel = np.ones((slack * 2 + 1, slack * 2 + 1), dtype=np.uint8)
    relaxed_precision = []
    relaxed_recall = []
    for pred_np, truth_np in zip(pred.detach().cpu().numpy(), truth.detach().cpu().numpy()):
        pred_np = np.squeeze(pred_np).astype(np.uint8)
        truth_np = np.squeeze(truth_np).astype(np.uint8)
        pred_buffer = cv2.dilate(pred_np, kernel)
        truth_buffer = cv2.dilate(truth_np, kernel)
        relaxed_precision.append(float((pred_np * truth_buffer).sum() / (pred_np.sum() + eps)))
        relaxed_recall.append(float((truth_np * pred_buffer).sum() / (truth_np.sum() + eps)))
    return {
        "relaxed_precision": float(np.mean(relaxed_precision)) if relaxed_precision else 0.0,
        "relaxed_recall": float(np.mean(relaxed_recall)) if relaxed_recall else 0.0,
    }


class MetricAccumulator:
    """Accumulate confusion counts and geometric metrics over a full dataset."""

    def __init__(self, threshold: float = 0.5, boundary_radius: int = 3, relaxed_slack: int = 3) -> None:
        self.threshold = threshold
        self.boundary_radius = boundary_radius
        self.relaxed_slack = relaxed_slack
        self.counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
        self.boundary_ious: list[float] = []
        self.relaxed_precisions: list[float] = []
        self.relaxed_recalls: list[float] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = torch.sigmoid(logits) > self.threshold
        labels = targets > 0.5
        self.counts["tp"] += (predictions & labels).sum().item()
        self.counts["fp"] += (predictions & ~labels).sum().item()
        self.counts["fn"] += (~predictions & labels).sum().item()
        self.counts["tn"] += (~predictions & ~labels).sum().item()
        self.boundary_ious.append(calculate_boundary_iou(predictions, labels, self.boundary_radius))
        relaxed = calculate_relaxed_metrics(predictions, labels, self.relaxed_slack)
        self.relaxed_precisions.append(relaxed["relaxed_precision"])
        self.relaxed_recalls.append(relaxed["relaxed_recall"])

    def compute(self) -> dict[str, float]:
        tp, fp, fn, tn = (self.counts[key] for key in ("tp", "fp", "fn", "tn"))
        eps = 1e-6
        dice = (2 * tp) / (2 * tp + fp + fn + eps)
        return {
            "road_iou": tp / (tp + fp + fn + eps),
            "f1_score": dice,
            "dice": dice,
            "precision": tp / (tp + fp + eps),
            "recall": tp / (tp + fn + eps),
            "pixel_accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
            "boundary_iou": float(np.mean(self.boundary_ious)) if self.boundary_ious else 0.0,
            "relaxed_precision": float(np.mean(self.relaxed_precisions)) if self.relaxed_precisions else 0.0,
            "relaxed_recall": float(np.mean(self.relaxed_recalls)) if self.relaxed_recalls else 0.0,
        }


def calculate_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    predictions = probs > threshold
    labels = targets > 0.5

    true_positive = (predictions & labels).sum().item()
    false_positive = (predictions & ~labels).sum().item()
    false_negative = (~predictions & labels).sum().item()
    true_negative = (~predictions & ~labels).sum().item()

    iou = true_positive / (true_positive + false_positive + false_negative + eps)
    dice = (2 * true_positive) / (
        2 * true_positive + false_positive + false_negative + eps
    )
    precision = true_positive / (true_positive + false_positive + eps)
    recall = true_positive / (true_positive + false_negative + eps)
    accuracy = (true_positive + true_negative) / (
        true_positive + true_negative + false_positive + false_negative + eps
    )

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": dice,
        "pixel_accuracy": accuracy,
    }


def resize_logits_to_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        logits,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
