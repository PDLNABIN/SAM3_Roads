from __future__ import annotations

import albumentations as A

from sam3_roads.config import AugmentationConfig

# SAM3 expects images normalized with mean=0.5, std=0.5 (scaling [0, 1] -> [-1, 1])
_SAM3_MEAN = (0.5, 0.5, 0.5)
_SAM3_STD = (0.5, 0.5, 0.5)


def build_train_transform(config: AugmentationConfig) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=config.image_size, width=config.image_size),
            A.RandomBrightnessContrast(p=config.random_brightness_contrast_p),
            A.CLAHE(p=config.clahe_p),
            A.RandomRotate90(p=config.random_rotate90_p),
            A.Rotate(limit=config.rotate_limit, p=config.rotate_p),
            A.ShiftScaleRotate(
                shift_limit=config.shift_limit,
                scale_limit=config.scale_limit,
                rotate_limit=0,
                p=config.shift_scale_rotate_p,
            ),
            A.OneOf(
                [
                    A.MotionBlur(p=1.0),
                    A.GaussianBlur(p=1.0),
                ],
                p=config.blur_p,
            ),
            A.GaussNoise(std_range=(0.02, 0.08), p=config.noise_p),
            A.Downscale(scale_range=(0.5, 0.9), p=config.downscale_p),
            A.GridDistortion(p=config.grid_distortion_p),
            A.HorizontalFlip(p=config.horizontal_flip_p),
            A.VerticalFlip(p=config.vertical_flip_p),
            A.Normalize(mean=_SAM3_MEAN, std=_SAM3_STD),
        ]
    )


def build_eval_transform(config: AugmentationConfig) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=config.image_size, width=config.image_size),
            A.Normalize(mean=_SAM3_MEAN, std=_SAM3_STD),
        ]
    )
