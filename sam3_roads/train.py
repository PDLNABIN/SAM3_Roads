from __future__ import annotations

import csv
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sam3_roads.config import AppConfig
from sam3_roads.dataset import ImageSegmentationDataset
from sam3_roads.losses import bce_dice_loss, calculate_iou, resize_logits_to_target
from sam3_roads.model import build_model
from sam3_roads.transforms import build_eval_transform, build_train_transform
from sam3_roads.visualize import save_prediction_preview, save_training_curves


def run_training(config: AppConfig) -> dict[str, Path | float]:
    set_seed(config.training.seed)
    configure_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch: {torch.__version__} | Device: {device}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu} | VRAM: {vram_gb:.1f} GB")

    train_loader, val_loader, test_loader = build_loaders(config)
    model = build_model(config.model, config.augmentation.image_size, device)

    if config.training.data_parallel and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    optimizer = build_optimizer(model, config)
    output_dir = config.output.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / config.output.checkpoint_name

    history = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        config=config,
        checkpoint_path=checkpoint_path,
    )

    history_path = output_dir / config.output.history_name
    write_history(history, history_path)
    curves_path = output_dir / config.output.curves_name
    save_training_curves(history, curves_path)

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        state_dict = state["model_state_dict"]
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    preview_path = output_dir / config.output.prediction_preview_name
    save_prediction_preview(
        model=model,
        dataloader=test_loader,
        device=device,
        output_path=preview_path,
        num_samples=config.output.preview_samples,
        threshold=config.training.iou_threshold,
    )

    return {
        "best_val_iou": max(history["val_iou"]) if history["val_iou"] else 0.0,
        "checkpoint_path": checkpoint_path,
        "history_path": history_path,
        "curves_path": curves_path,
        "preview_path": preview_path,
    }


def build_loaders(config: AppConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = build_train_transform(config.augmentation)
    eval_transform = build_eval_transform(config.augmentation)
    dataset_config = config.dataset

    train_dataset = ImageSegmentationDataset(
        base_dir=dataset_config.data_dir,
        split=dataset_config.train_split,
        transform=train_transform,
        image_dir_name=dataset_config.image_dir_name,
        mask_dir_name=dataset_config.mask_dir_name,
    )
    val_dataset = ImageSegmentationDataset(
        base_dir=dataset_config.data_dir,
        split=dataset_config.val_split,
        transform=eval_transform,
        image_dir_name=dataset_config.image_dir_name,
        mask_dir_name=dataset_config.mask_dir_name,
    )
    test_dataset = ImageSegmentationDataset(
        base_dir=dataset_config.data_dir,
        split=dataset_config.test_split,
        transform=eval_transform,
        image_dir_name=dataset_config.image_dir_name,
        mask_dir_name=dataset_config.mask_dir_name,
    )

    kwargs = {
        "num_workers": config.training.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if config.training.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    print(f"Training Dataset: {len(train_dataset)} Images")
    print(f"Validation Dataset: {len(val_dataset)} Images")
    print(f"Testing Dataset: {len(test_dataset)} Images")
    print(f"SAM3 input size: {config.augmentation.image_size}x{config.augmentation.image_size}")

    return (
        DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, **kwargs),
        DataLoader(val_dataset, batch_size=config.training.val_batch_size, shuffle=False, **kwargs),
        DataLoader(test_dataset, batch_size=config.training.test_batch_size, shuffle=False, **kwargs),
    )


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: AppConfig,
    checkpoint_path: Path,
) -> dict[str, list[float]]:
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_iou": []}
    best_val_iou = -1.0

    use_amp = config.training.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    scaler = GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    print("Starting training...")
    for epoch in range(config.training.epochs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            config=config,
            epoch=epoch,
        )
        avg_val_loss, avg_val_iou = validate_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            config=config,
            epoch=epoch,
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_iou"].append(avg_val_iou)

        print(
            f"Epoch {epoch + 1:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}"
        )

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            save_checkpoint(model, optimizer, history, config, epoch, best_val_iou, checkpoint_path)
            print(f"Saved new best model to: {checkpoint_path} (Val IoU={best_val_iou:.4f})")

    print(f"Training finished. Best Val IoU: {best_val_iou:.4f}")
    return history


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    config: AppConfig,
    epoch: int,
) -> float:
    model.train()
    train_loss_sum = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.training.epochs} [TRAIN]")

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        amp_ctx = autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if use_amp else nullcontext()
        with amp_ctx:
            logits = model(pixel_values=images)
            logits_resized = resize_logits_to_target(logits, masks)
            loss = bce_dice_loss(logits_resized, masks, bce_weight=config.training.bce_weight)
            loss = loss / config.training.accumulation_steps

        if loss.dim() > 0:
            loss = loss.mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss detected.")

        scaler.scale(loss).backward()

        if ((batch_idx + 1) % config.training.accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        step_loss = loss.item() * config.training.accumulation_steps
        train_loss_sum += step_loss
        pbar.set_postfix(loss=f"{step_loss:.4f}")

    return train_loss_sum / max(len(train_loader), 1)


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    config: AppConfig,
    epoch: int,
) -> tuple[float, float]:
    model.eval()
    val_loss_sum = 0.0
    val_iou_sum = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.training.epochs} [VALID]")
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            amp_ctx = autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if use_amp else nullcontext()
            with amp_ctx:
                logits = model(pixel_values=images)
                logits_resized = resize_logits_to_target(logits, masks)
                loss = bce_dice_loss(logits_resized, masks, bce_weight=config.training.bce_weight)

            if loss.dim() > 0:
                loss = loss.mean()

            val_loss_sum += loss.item()
            val_iou_sum += calculate_iou(logits_resized, masks, threshold=config.training.iou_threshold)

    return val_loss_sum / max(len(val_loader), 1), val_iou_sum / max(len(val_loader), 1)


def build_optimizer(model: nn.Module, config: AppConfig) -> torch.optim.Optimizer:
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    lora_params = [param for param in base_model.backbone.parameters() if param.requires_grad]
    head_params = [param for param in base_model.mask_head.parameters() if param.requires_grad]
    print(f"Trainable params in backbone: {sum(param.numel() for param in lora_params):,}")
    print(f"Trainable params in head: {sum(param.numel() for param in head_params):,}")

    return torch.optim.AdamW(
        [
            {"params": lora_params, "lr": config.training.backbone_lr},
            {"params": head_params, "lr": config.training.head_lr},
        ],
        weight_decay=config.training.weight_decay,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: dict[str, list[float]],
    config: AppConfig,
    epoch: int,
    best_val_iou: float,
    checkpoint_path: Path,
) -> None:
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_iou": best_val_iou,
            "history": history,
            "config": {
                "num_epochs": config.training.epochs,
                "accumulation_steps": config.training.accumulation_steps,
                "image_size": config.augmentation.image_size,
                "model_name": config.model.model_name,
                "seed": config.training.seed,
            },
        },
        checkpoint_path,
    )


def write_history(history: dict[str, list[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss", "val_iou"])
        writer.writeheader()
        for idx, train_loss in enumerate(history["train_loss"]):
            writer.writerow(
                {
                    "epoch": idx + 1,
                    "train_loss": train_loss,
                    "val_loss": history["val_loss"][idx],
                    "val_iou": history["val_iou"][idx],
                }
            )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
