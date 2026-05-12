from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageSegmentationDataset(Dataset):
    SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    SUPPORTED_MASK_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        base_dir: str | Path,
        split: str,
        transform=None,
        image_dir_name: str = "images",
        mask_dir_name: str = "masks",
    ) -> None:
        super().__init__()
        self.split_dir = Path(base_dir) / split
        self.image_dir = self.split_dir / image_dir_name
        self.mask_dir = self.split_dir / mask_dir_name
        self.transform = transform

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing mask directory: {self.mask_dir}")

        self.mask_paths = {
            path.stem: path
            for path in self.mask_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_MASK_EXTS
        }
        self.image_filenames = sorted(
            [
                path.name
                for path in self.image_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in self.SUPPORTED_IMAGE_EXTS
                and path.stem in self.mask_paths
            ]
        )

        if not self.image_filenames:
            raise RuntimeError(f"No valid image-mask pairs found in split: {split}")

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_name = self.image_filenames[idx]
        image_path = self.image_dir / image_name
        mask_path = self.mask_paths[image_path.stem]

        image_rgb = self._read_image(image_path)
        mask = self._read_mask(mask_path)

        if self.transform is not None:
            transformed = self.transform(image=image_rgb, mask=mask)
            image_rgb = transformed["image"]
            mask = transformed["mask"]

        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor

    def _read_image(self, path: Path) -> np.ndarray:
        if path.suffix.lower() in {".tif", ".tiff"}:
            import rasterio

            with rasterio.open(path) as src:
                if src.count >= 3:
                    arr = src.read([1, 2, 3]).transpose(1, 2, 0)
                else:
                    band = src.read(1)
                    arr = np.stack([band, band, band], axis=-1)
            return _to_uint8(arr)

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _read_mask(self, path: Path) -> np.ndarray:
        if path.suffix.lower() in {".tif", ".tiff"}:
            import rasterio

            with rasterio.open(path) as src:
                mask = src.read(1)
            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8) * 255
            return mask

        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        return mask


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    return ((arr - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
