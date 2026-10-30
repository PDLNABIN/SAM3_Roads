from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import Sam3Model

from sam3_roads.config import ModelConfig


class LegacyMaskHead(nn.Sequential):
    def __init__(self, embed_dim: int, num_classes: int = 1) -> None:
        super().__init__(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )


class ProgressiveMaskHead(nn.Sequential):
    """Multi-stage progressive transpose convolution decoder.

    Upsamples features 8x (e.g. 36x36 -> 72x72 -> 144x144 -> 288x288)
    using learnable transposed convolutions paired with GroupNorm (safe for batch_size=1)
    and GELU activations, enabling crisp delineation of thin linear road boundaries.
    """

    def __init__(self, embed_dim: int, num_classes: int = 1) -> None:
        super().__init__(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            # Stage 1: 2x upsampling (36 -> 72)
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            # Stage 2: 2x upsampling (72 -> 144)
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            # Stage 3: 2x upsampling (144 -> 288)
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            # Final refinement and projection to classes
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )


class UnpromptedSAMSegmenter(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_classes: int = 1,
        head_type: str = "upsampling",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise RuntimeError("embed_dim must be a positive integer for SAM3 backbone outputs.")

        if head_type in {"legacy", "linear"}:
            self.mask_head: nn.Module = LegacyMaskHead(embed_dim=embed_dim, num_classes=num_classes)
        else:
            self.mask_head = ProgressiveMaskHead(embed_dim=embed_dim, num_classes=num_classes)

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False):
        # Backwards compatibility: gracefully adapt if an older 3-layer mask head checkpoint is loaded
        if (
            "mask_head.0.weight" in state_dict
            and state_dict["mask_head.0.weight"].shape == torch.Size([128, self.embed_dim, 3, 3])
        ):
            if not isinstance(self.mask_head, LegacyMaskHead):
                target_device = next(self.parameters()).device
                self.mask_head = LegacyMaskHead(self.embed_dim, self.num_classes).to(target_device)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "base_model"):
            backbone_outputs = self.backbone.base_model(pixel_values=pixel_values)
        else:
            backbone_outputs = self.backbone(pixel_values=pixel_values)

        features = self._reshape_features(backbone_outputs.last_hidden_state)
        features = features.to(self.mask_head[0].weight.dtype)
        return self.mask_head(features)

    def _reshape_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            return features
        if features.ndim == 3:
            batch, sequence_length, channels = features.shape
            side = int(np.sqrt(sequence_length))
            if side * side != sequence_length:
                raise RuntimeError(f"Cannot reshape sequence length {sequence_length} into square feature map")
            return features.transpose(1, 2).contiguous().view(batch, channels, side, side)
        raise RuntimeError(f"Unexpected backbone feature shape: {tuple(features.shape)}")


def login_from_env() -> None:
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)


def build_model(config: ModelConfig, image_size: int, device: torch.device) -> UnpromptedSAMSegmenter:
    login_from_env()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model_kwargs = {"torch_dtype": dtype}
    if config.attention_implementation:
        model_kwargs["attn_implementation"] = config.attention_implementation

    try:
        sam_model = Sam3Model.from_pretrained(config.model_name, image_size=image_size, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        sam_model = Sam3Model.from_pretrained(config.model_name, image_size=image_size, **model_kwargs)

    vision_backbone = sam_model.vision_encoder
    if config.gradient_checkpointing:
        enable_gradient_checkpointing(vision_backbone)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    peft_backbone = get_peft_model(vision_backbone, lora_config)
    segmenter = UnpromptedSAMSegmenter(
        backbone=peft_backbone,
        embed_dim=resolve_backbone_embed_dim(sam_model, vision_backbone),
        num_classes=config.num_classes,
        head_type=getattr(config, "head_type", "upsampling"),
    ).to(device)

    if hasattr(segmenter.backbone, "enable_input_require_grads"):
        segmenter.backbone.enable_input_require_grads()

    return segmenter


def enable_gradient_checkpointing(module: nn.Module) -> bool:
    if hasattr(module, "gradient_checkpointing_enable"):
        try:
            module.gradient_checkpointing_enable()
            return True
        except (ValueError, TypeError, AttributeError):
            return False
    return False


def resolve_backbone_embed_dim(sam_model: nn.Module, vision_backbone: nn.Module) -> int:
    candidates = [
        getattr(getattr(vision_backbone, "config", None), "hidden_size", None),
        getattr(getattr(sam_model, "config", None), "hidden_size", None),
        getattr(getattr(getattr(sam_model, "config", None), "vision_config", None), "hidden_size", None),
    ]
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    return 1024
