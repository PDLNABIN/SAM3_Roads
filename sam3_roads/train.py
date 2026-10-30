from __future__ import annotations

import csv
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sam3_roads.benchmark import build_portfolio_stats
from sam3_roads.config import AppConfig
from sam3_roads.dataset import ImageSegmentationDataset
from sam3_roads.losses import bce_dice_loss, resize_logits_to_target, MetricAccumulator
from sam3_roads.model import build_model
from sam3_roads.transforms import build_eval_transform, build_train_transform
from sam3_roads.visualize import save_prediction_preview, save_split_predictions, save_training_curves


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

    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))

    history = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        checkpoint_path=checkpoint_path,
    )

    history_path = output_dir / config.output.history_name
    write_history(history, history_path)
    (output_dir / "training_metrics.json").write_text(
        json.dumps(
            {
                "epochs": len(history["train_loss"]),
                "best_val_iou": max(history["val_iou"]) if history["val_iou"] else 0.0,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    curves_path = output_dir / config.output.curves_name
    save_training_curves(history, curves_path)

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        state_dict = state["model_state_dict"]
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    validation_metrics = evaluate_split(
        model, val_loader, device, config, output_dir / config.output.validation_predictions_dir
    )
    validation_metrics["split"] = "validation"
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2) + "\n", encoding="utf-8"
    )
    preview_path = output_dir / config.output.prediction_preview_name
    save_prediction_preview(
        model=model,
        dataloader=test_loader,
        device=device,
        output_path=preview_path,
        num_samples=config.output.preview_samples,
        threshold=config.training.iou_threshold,
    )
    test_metrics = evaluate_split(
        model, test_loader, device, config, output_dir / config.output.test_predictions_dir
    )
    test_metrics["split"] = "test"
    portfolio_stats = build_portfolio_stats(model, next(iter(test_loader))[0], device)
    test_metrics["portfolio_stats"] = portfolio_stats
    metrics_path = output_dir / config.output.metrics_name
    metrics_path.write_text(json.dumps(test_metrics, indent=2) + "\n", encoding="utf-8")
    (output_dir / "portfolio_stats.json").write_text(
        json.dumps({"metrics": test_metrics, **portfolio_stats}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Test metrics:")
    for name, value in test_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.4f}")

    return {
        "best_val_iou": max(history["val_iou"]) if history["val_iou"] else 0.0,
        "test_metrics": test_metrics,
        "checkpoint_path": checkpoint_path,
        "history_path": history_path,
        "curves_path": curves_path,
        "preview_path": preview_path,
        "metrics_path": metrics_path,
    }


def evaluate_checkpoint(config: AppConfig, checkpoint_path: Path) -> dict[str, object]:
    set_seed(config.training.seed)
    configure_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, test_loader = build_loaders(config)
    model = build_model(config.model, config.augmentation.image_size, device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    validation_metrics = evaluate_split(
        model, val_loader, device, config, config.output.output_dir / config.output.validation_predictions_dir
    )
    test_metrics = evaluate_split(
        model, test_loader, device, config, config.output.output_dir / config.output.test_predictions_dir
    )
    metrics = {"validation": validation_metrics, "test": test_metrics}
    stats = build_portfolio_stats(model, next(iter(test_loader))[0], device)
    result = {"metrics": metrics, **stats}
    output_dir = config.output.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "portfolio_stats.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: AppConfig,
    prediction_dir: Path,
) -> dict[str, object]:
    model.eval()
    accumulator = MetricAccumulator(threshold=config.training.iou_threshold)
    records: list[dict[str, object]] = []
    for batch_index, (images, masks) in enumerate(tqdm(dataloader, desc=f"Evaluating {prediction_dir.name}")):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = resize_logits_to_target(model(pixel_values=images), masks)
        accumulator.update(logits, masks)
        records.extend(
            save_split_predictions(
                images=images,
                masks=masks,
                logits=logits,
                dataset=dataloader.dataset,
                batch_index=batch_index,
                output_dir=prediction_dir,
                threshold=config.training.iou_threshold,
            )
        )
    metrics: dict[str, object] = accumulator.compute()
    metrics["images"] = len(dataloader.dataset)
    metrics["threshold"] = config.training.iou_threshold
    (prediction_dir / "predictions.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return metrics


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
    scheduler: LambdaLR,
    device: torch.device,
    config: AppConfig,
    checkpoint_path: Path,
) -> dict[str, list[float]]:
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_f1_score": [],
        "val_precision": [],
        "val_recall": [],
        "val_boundary_iou": [],
        "val_relaxed_precision": [],
        "val_relaxed_recall": [],
    }
    best_val_iou = -1.0
    epochs_without_improvement = 0

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
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            config=config,
            epoch=epoch,
        )
        avg_val_loss, val_metrics = validate_epoch(
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
        history["val_iou"].append(val_metrics["road_iou"])
        for name in (
            "f1_score",
            "precision",
            "recall",
            "boundary_iou",
            "relaxed_precision",
            "relaxed_recall",
        ):
            history[f"val_{name}"].append(val_metrics[name])

        print(
            f"Epoch {epoch + 1:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val IoU: {val_metrics['road_iou']:.4f} | "
            f"Val F1: {val_metrics['f1_score']:.4f} | "
            f"LR backbone: {optimizer.param_groups[0]['lr']:.2e} | "
            f"LR head: {optimizer.param_groups[1]['lr']:.2e}"
        )

        if val_metrics["road_iou"] > best_val_iou + config.training.early_stopping_min_delta:
            best_val_iou = val_metrics["road_iou"]
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, history, config, epoch, best_val_iou, checkpoint_path)
            print(f"Saved new best model to: {checkpoint_path} (Val IoU={best_val_iou:.4f})")
        else:
            epochs_without_improvement += 1
            patience = config.training.early_stopping_patience
            if patience > 0 and epochs_without_improvement >= patience:
                print(
                    f"Early stopping after {epoch + 1} epochs: validation IoU did not improve "
                    f"by at least {config.training.early_stopping_min_delta:.4f} "
                    f"for {patience} consecutive epochs."
                )
                break

    print(f"Training finished. Best Val IoU: {best_val_iou:.4f}")
    return history


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
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
            loss = bce_dice_loss(
                logits_resized,
                masks,
                bce_weight=config.training.bce_weight,
                pos_weight=config.training.pos_weight,
            )
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
            scheduler.step()
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
) -> tuple[float, dict[str, float]]:
    model.eval()
    val_loss_sum = 0.0
    accumulator = MetricAccumulator(threshold=config.training.iou_threshold)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.training.epochs} [VALID]")
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            amp_ctx = autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if use_amp else nullcontext()
            with amp_ctx:
                logits = model(pixel_values=images)
                logits_resized = resize_logits_to_target(logits, masks)
                loss = bce_dice_loss(
                logits_resized,
                masks,
                bce_weight=config.training.bce_weight,
                pos_weight=config.training.pos_weight,
            )

            if loss.dim() > 0:
                loss = loss.mean()

            val_loss_sum += loss.item()
            accumulator.update(logits_resized, masks)

    return val_loss_sum / max(len(val_loader), 1), accumulator.compute()


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: AppConfig,
) -> dict[str, object]:
    model.eval()
    from sam3_roads.losses import MetricAccumulator

    accumulator = MetricAccumulator(threshold=config.training.iou_threshold)

    for images, masks in tqdm(test_loader, desc="Evaluating test set"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = resize_logits_to_target(model(pixel_values=images), masks)
        accumulator.update(logits, masks)
    metrics: dict[str, object] = accumulator.compute()
    metrics["test_images"] = len(test_loader.dataset)
    metrics["threshold"] = config.training.iou_threshold
    return metrics


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
        fieldnames = ["epoch", *history.keys()]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for idx, train_loss in enumerate(history["train_loss"]):
            row = {"epoch": idx + 1}
            row.update({name: values[idx] for name, values in history.items()})
            writer.writerow(row)


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


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: AppConfig,
    steps_per_epoch: int,
) -> LambdaLR:
    """Cosine annealing schedule with a linear warmup phase.

    During the warmup phase (first ``warmup_epochs`` epochs) the LR ramps from
    ``min_lr_factor`` → 1.0 linearly.  After warmup it follows a cosine decay
    from 1.0 → ``min_lr_factor`` over the remaining steps.
    """
    total_steps = config.training.epochs * steps_per_epoch
    # Guard: accumulation means we step the scheduler once per *effective* step
    effective_steps_per_epoch = max(steps_per_epoch // config.training.accumulation_steps, 1)
    total_effective = config.training.epochs * effective_steps_per_epoch
    warmup_steps = int(config.training.warmup_epochs * effective_steps_per_epoch)
    min_factor = config.training.min_lr_factor

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear ramp: min_factor → 1.0
            return min_factor + (1.0 - min_factor) * current_step / max(warmup_steps, 1)
        # Cosine decay: 1.0 → min_factor
        progress = (current_step - warmup_steps) / max(total_effective - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
