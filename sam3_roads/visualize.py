from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sam3_roads.losses import resize_logits_to_target


def save_training_curves(history: dict[str, list[float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["val_iou"], marker="o", color="green", label="Val IoU")
    axes[1].set_title("Validation IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_prediction_preview(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    output_path: str | Path,
    num_samples: int = 4,
    threshold: float = 0.5,
) -> None:
    output_path = Path(output_path)
    model.eval()
    shown = 0
    fig = plt.figure(figsize=(12, 4 * num_samples))

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(pixel_values=images)
            logits = resize_logits_to_target(logits, masks)
            pred_mask = (torch.sigmoid(logits) > threshold).float()

            img_np = images[0].float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
            p2, p98 = np.percentile(img_np, (2, 98))
            img_np = np.clip((img_np - p2) / (p98 - p2 + 1e-6), 0, 1)

            gt_np = masks[0, 0].float().cpu().numpy()
            pred_np = pred_mask[0, 0].float().cpu().numpy()

            plt.subplot(num_samples, 2, 2 * shown + 1)
            plt.imshow(img_np)
            plt.imshow(gt_np, alpha=0.45, cmap="Reds")
            plt.title("Ground Truth Mask + Image")
            plt.axis("off")

            plt.subplot(num_samples, 2, 2 * shown + 2)
            plt.imshow(img_np)
            plt.imshow(pred_np, alpha=0.45, cmap="Blues")
            plt.title("Predicted Mask + Image")
            plt.axis("off")

            shown += 1
            if shown >= num_samples:
                break

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
