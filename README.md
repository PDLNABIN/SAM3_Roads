# Road Segmentation from Drone Imagery using SAM3 + LoRA

Fine-tuning **SAM3** (`facebook/sam3`) with **LoRA (PEFT)** for binary road segmentation from drone/aerial imagery. The original notebook has been converted into a reusable Poetry project with a config-driven CLI.

## Project Structure

```text
E:/SAM3_Roads/
|-- sam3-peft.ipynb
|-- pyproject.toml
|-- README.md
|-- configs/
|   `-- sam3_roads.example.toml
`-- sam3_roads/
    |-- cli.py
    |-- config.py
    |-- dataset.py
    |-- losses.py
    |-- model.py
    |-- train.py
    |-- transforms.py
    `-- visualize.py
```

## How It Works

SAM3 is a prompted multimodal model. This pipeline repurposes its **vision encoder only** as a pretrained feature extractor. The DETR decoder and mask decoder are bypassed because they require text and bounding box prompts. A lightweight CNN head is trained on top of the LoRA-adapted features.

```text
Drone Image (504x504)
      |
      v
SAM3 Vision Encoder <- LoRA adapters on q_proj, v_proj
      | last_hidden_state
      v
Reshape tokens -> spatial feature map
      |
      v
CNN Mask Head: Conv(C->128) -> Conv(128->64) -> Conv(64->1)
      |
      v
Bilinear upsample -> Binary road mask (504x504)
```

Only the LoRA adapter weights and mask head are trained. The rest of the SAM3 vision backbone stays frozen.

## Dataset Structure

```text
data/
|-- train/
|   |-- images/
|   `-- masks/
|-- val/
|   |-- images/
|   `-- masks/
`-- test/
    |-- images/
    `-- masks/
```

Masks are paired to images by filename stem. Mixed image/mask extensions are supported, and GeoTIFF images are handled through `rasterio`.

## Setup

```powershell
cd E:\SAM3_Roads
poetry install
```

If your SAM3 model download requires Hugging Face authentication:

```powershell
$env:HF_TOKEN = "your_huggingface_token"
```

Edit [configs/sam3_roads.example.toml](configs/sam3_roads.example.toml) so `dataset.data_dir` points to the folder containing the `train`, `val`, and `test` splits.

## Training

```powershell
poetry run sam3-roads train --config configs\sam3_roads.example.toml
```

The default notebook-equivalent hyperparameters are stored in the config:

```text
image_size = 504
batch_size = 1
accumulation_steps = 8
epochs = 15
lora_r = 32
lora_alpha = 64
lora_target_modules = ["q_proj", "v_proj"]
backbone_lr = 0.0001
head_lr = 0.0005
```

Training uses AMP, gradient accumulation, gradient clipping, AdamW, validation IoU tracking, and best-checkpoint saving.

## Outputs

Outputs are written to `outputs/sam3_roads/` by default:

```text
outputs/sam3_roads/
|-- sam3_lora_best.pth
|-- history.csv
|-- training_curves.png
`-- prediction_preview.png
```

When you run training locally, these README image links will show the generated artifacts:

![Training curves](outputs/sam3_roads/training_curves.png)

![Prediction preview](outputs/sam3_roads/prediction_preview.png)

Checkpoint contents:

- `model_state_dict`
- `optimizer_state_dict`
- `best_val_iou`
- `history`
- training `config`

## Key Design Decisions

**`image_size = 504`**

SAM3 uses 2D RoPE positional embeddings baked at load time. The resize transform, `from_pretrained(image_size=...)`, and config image size must match.

**LoRA on vision encoder only**

SAM3's full decoder path needs prompts. The vision encoder alone produces spatial features that work well with a compact CNN mask head.

**PEFT routing in `forward`**

```python
if hasattr(self.backbone, "base_model"):
    out = self.backbone.base_model(pixel_values=pixel_values)
```

This avoids PEFT passing keyword arguments that SAM3 does not accept.

**Aerial display fix**

The preview visualization applies a 2-98 percentile contrast stretch so drone imagery displays clearly.

## Augmentations

| Transform | Purpose |
|---|---|
| `RandomRotate90` + `Rotate(+/-45)` | Orientation invariance |
| `HorizontalFlip` + `VerticalFlip` | Standard aerial augmentation |
| `CLAHE` | Improve low-contrast roads in shadow |
| `Downscale(0.5-0.9)` | Simulate GSD variation |
| `GridDistortion` | Simulate orthorectification residuals |
| `ShiftScaleRotate` | Viewpoint variation |
| `GaussNoise` + `MotionBlur` | Sensor noise and motion blur |

## Some Results

<img width="761" height="650" alt="Road segmentation result 1" src="https://github.com/user-attachments/assets/d3885f55-5365-4bba-a63e-ca79981b234f" />

<img width="769" height="675" alt="Road segmentation result 2" src="https://github.com/user-attachments/assets/a384adef-9b72-4604-b16c-d0d527f4a041" />

<img width="727" height="648" alt="Road segmentation result 3" src="https://github.com/user-attachments/assets/2f7ca85d-8a08-4e52-abc2-964a45227cff" />

<img width="1635" height="632" alt="Training result overview" src="https://github.com/user-attachments/assets/182d6976-e5bf-4a58-b852-e1c090879c15" />

## Limitations

- Label boundaries are approximate human-drawn buffers, so visual inspection can be more informative than raw IoU.
- SAM3's pretrained mask decoder is not used because this workflow is prompt-free.
- The current project is binary road / no-road segmentation only.
- At 504x504 resolution, batch size is memory-limited on 16 GB GPUs.

## References

- [Segment Anything Model 3 (SAM3)](https://huggingface.co/facebook/sam3)
- [PEFT - Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Albumentations](https://albumentations.ai/)
- [OpenStreetMap](https://www.openstreetmap.org/)
