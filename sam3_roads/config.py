from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


@dataclass(slots=True)
class DatasetConfig:
    data_dir: Path
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    image_dir_name: str = "images"
    mask_dir_name: str = "masks"


@dataclass(slots=True)
class AugmentationConfig:
    image_size: int = 1024
    random_brightness_contrast_p: float = 0.5
    clahe_p: float = 0.2
    random_rotate90_p: float = 0.75
    rotate_limit: int = 45
    rotate_p: float = 0.4
    shift_limit: float = 0.05
    scale_limit: float = 0.15
    shift_scale_rotate_p: float = 0.4
    blur_p: float = 0.3
    noise_p: float = 0.3
    downscale_p: float = 0.3
    grid_distortion_p: float = 0.2
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5


@dataclass(slots=True)
class ModelConfig:
    model_name: str = "facebook/sam3"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    num_classes: int = 1
    attention_implementation: str | None = "eager"
    gradient_checkpointing: bool = True
    head_type: str = "upsampling"


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 1
    val_batch_size: int = 1
    test_batch_size: int = 1
    num_workers: int = 0
    epochs: int = 15
    accumulation_steps: int = 8
    backbone_lr: float = 1e-4
    head_lr: float = 5e-4
    weight_decay: float = 1e-4
    clip_grad_norm: float = 1.0
    bce_weight: float = 0.5
    iou_threshold: float = 0.5
    data_parallel: bool = True
    amp: bool = True
    warmup_epochs: float = 1.0
    min_lr_factor: float = 0.01
    pos_weight: float | None = None
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0


@dataclass(slots=True)
class OutputConfig:
    output_dir: Path = Path("outputs/sam3_roads")
    checkpoint_name: str = "sam3_lora_best.pth"
    history_name: str = "history.csv"
    curves_name: str = "training_curves.png"
    prediction_preview_name: str = "prediction_preview.png"
    metrics_name: str = "test_metrics.json"
    validation_predictions_dir: str = "validation_predictions"
    test_predictions_dir: str = "test_predictions"
    preview_samples: int = 4


@dataclass(slots=True)
class AppConfig:
    dataset: DatasetConfig
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section [{name}] must be a table.")
    return value


def _path(value: str | Path) -> Path:
    return Path(value).expanduser()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = _read_toml(config_path)

    dataset_raw = _section(raw, "dataset")
    if "data_dir" not in dataset_raw:
        raise ValueError("Config must define dataset.data_dir.")

    dataset = DatasetConfig(
        data_dir=_path(dataset_raw["data_dir"]),
        train_split=dataset_raw.get("train_split", "train"),
        val_split=dataset_raw.get("val_split", "val"),
        test_split=dataset_raw.get("test_split", "test"),
        image_dir_name=dataset_raw.get("image_dir_name", "images"),
        mask_dir_name=dataset_raw.get("mask_dir_name", "masks"),
    )

    augmentation = AugmentationConfig(**_section(raw, "augmentation"))
    model = ModelConfig(**_section(raw, "model"))
    training = TrainingConfig(**_section(raw, "training"))

    output_raw = _section(raw, "output")
    if "output_dir" in output_raw:
        output_raw = {**output_raw, "output_dir": _path(output_raw["output_dir"])}
    output = OutputConfig(**output_raw)

    return AppConfig(
        dataset=dataset,
        augmentation=augmentation,
        model=model,
        training=training,
        output=output,
    )
