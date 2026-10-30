from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from sam3_roads.config import AppConfig
from sam3_roads.model import build_model
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


def save_split_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    logits: torch.Tensor,
    dataset,
    batch_index: int,
    output_dir: Path,
    threshold: float,
) -> list[dict[str, object]]:
    """Save aligned RGB, ground-truth, prediction, and triplet files per image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (torch.sigmoid(logits) > threshold).float()
    records = []
    batch_size = images.shape[0]
    for item_index in range(batch_size):
        dataset_index = batch_index * batch_size + item_index
        stem = Path(dataset.image_filenames[dataset_index]).stem
        rgb = _stretch(images[item_index].detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy())
        truth = masks[item_index, 0].detach().cpu().numpy()
        prediction = predictions[item_index, 0].detach().cpu().numpy()
        rgb_path = output_dir / f"{stem}_image.png"
        truth_path = output_dir / f"{stem}_ground_truth.png"
        prediction_path = output_dir / f"{stem}_prediction.png"
        triplet_path = output_dir / f"{stem}_triplet.png"
        plt.imsave(rgb_path, rgb)
        plt.imsave(truth_path, truth, cmap="gray", vmin=0, vmax=1)
        plt.imsave(prediction_path, prediction, cmap="gray", vmin=0, vmax=1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb); axes[0].set_title("Image")
        axes[1].imshow(truth, cmap="gray"); axes[1].set_title("Ground Truth")
        axes[2].imshow(prediction, cmap="gray"); axes[2].set_title("Prediction")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(triplet_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        records.append({
            "image": stem,
            "image_path": rgb_path.name,
            "ground_truth_path": truth_path.name,
            "prediction_path": prediction_path.name,
            "triplet_path": triplet_path.name,
        })
    return records


def _stretch(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (2, 98))
    return np.clip((image - low) / (high - low + 1e-6), 0, 1)


def _skeleton(mask: np.ndarray) -> np.ndarray:
    skeleton = np.zeros_like(mask, dtype=np.uint8)
    work = (mask > 0).astype(np.uint8) * 255
    while cv2.countNonZero(work):
        eroded = cv2.erode(work, np.ones((3, 3), np.uint8))
        opened = cv2.dilate(eroded, np.ones((3, 3), np.uint8))
        skeleton |= ((work - opened) > 0).astype(np.uint8)
        work = eroded
    return skeleton


def export_portfolio_assets(config: AppConfig, checkpoint_path: Path, output_dir: Path) -> None:
    """Export web-ready qualitative panels from a trained checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model, config.augmentation.image_size, device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    _, _, test_loader = __import__("sam3_roads.train", fromlist=["build_loaders"]).build_loaders(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_batches = []
    mask_batches = []
    for batch_images, batch_masks in test_loader:
        image_batches.append(batch_images)
        mask_batches.append(batch_masks)
        if sum(batch.shape[0] for batch in image_batches) >= 4:
            break
    images = torch.cat(image_batches, dim=0)[:4].to(device)
    masks = torch.cat(mask_batches, dim=0)[:4].to(device)
    with torch.no_grad():
        predictions = (torch.sigmoid(resize_logits_to_target(model(pixel_values=images), masks)) > 0.5).float()

    for index in range(min(4, images.shape[0])):
        rgb = _stretch(images[index].cpu().permute(1, 2, 0).numpy())
        truth = masks[index, 0].cpu().numpy()
        prediction = predictions[index, 0].cpu().numpy()
        skeleton = _skeleton(prediction)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(rgb); axes[0].set_title("Drone RGB")
        axes[1].imshow(truth, cmap="gray"); axes[1].set_title("Ground Truth")
        axes[2].imshow(prediction, cmap="magma"); axes[2].set_title("SAM3 + LoRA")
        axes[3].imshow(rgb)
        skeleton_overlay = np.ma.masked_where(skeleton == 0, skeleton)
        axes[3].imshow(skeleton_overlay, cmap="spring", alpha=0.9)
        axes[3].set_title("Skeleton Overlay")
        for axis in axes: axis.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"triplet_{index + 1}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        overlay = rgb.copy()
        overlay[prediction > 0.5] = [0.1, 0.95, 1.0]
        cv2.imwrite(
            str(output_dir / f"slider_{index + 1}_overlay.png"),
            cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(output_dir / f"slider_{index + 1}_rgb.png"),
            cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, index, title in zip(axes, range(3), ["Canopy shadows", "Unpaved roads", "Sharp turns"]):
        rgb = _stretch(images[index].cpu().permute(1, 2, 0).numpy())
        axis.imshow(rgb)
        axis.imshow(predictions[index, 0].cpu(), cmap="cool", alpha=0.45)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "failure_case_analysis.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
