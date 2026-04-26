#  Road Segmentation from Drone Imagery using SAM3 + LoRA

Fine-tuning **SAM3** (`facebook/sam3`) with **LoRA (PEFT)** for binary road segmentation from drone/aerial imagery. Trained on manually labelled Kathmandu road tiles. Runs on Kaggle with dual T4 16 GB GPUs.

---

## How It Works

SAM3 is a prompted multimodal model. This pipeline repurposes its **vision encoder only** as a pretrained feature extractor — the DETR decoder and mask decoder are bypassed entirely since they require text + bounding box prompts. A lightweight CNN head is trained on top of the LoRA-adapted features.

```
Drone Image (504×504)
      │
      ▼
SAM3 Vision Encoder  ← LoRA adapters on q_proj, v_proj
      │  last_hidden_state
      ▼
Reshape tokens → spatial feature map (B, C, 36, 36)
      │
      ▼
CNN Mask Head: Conv(C→128) → Conv(128→64) → Conv(64→1)
      │
      ▼
Bilinear upsample → Binary road mask (504×504)
```

Only **~1–3% of SAM3's parameters** are trained (LoRA A/B matrices + mask head). The rest stays frozen.

---

## Dataset Structure

```
data/
├── train/
│   ├── images/    # aerial tiles (.jpg / .png / .tif)
│   └── masks/     # binary road masks (paired by filename stem)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Masks are paired to images by **filename stem** — mixed extensions within a split are supported (e.g. `.jpg` image with `.png` mask). GeoTIFF (`.tif`) images including 16-bit and multi-band are supported via `rasterio`.

---

## Setup

```bash
pip install -q -U ultralytics peft bitsandbytes transformers accelerate huggingface_hub rasterio
```

You will need a HuggingFace token with access to `facebook/sam3`. On Kaggle, store it as a secret named `HF_TOKEN`.

---

## Key Design Decisions

**`SAM3_INPUT_SIZE = 504`**
SAM3 uses 2D RoPE positional embeddings baked at load time. The resize transform, `from_pretrained(image_size=...)`, and `SAM3_INPUT_SIZE` must all match. Mismatching causes a tensor size error at the RoPE layer during validation.

**LoRA on vision encoder only**
SAM3's DETR pipeline (`detr_encoder → detr_decoder → mask_decoder`) requires text tokens and bounding box prompts — there is no prompt-free path unlike SAM1/SAM2. The vision encoder alone produces strong spatial features suitable for a CNN head.

**PEFT routing in `forward`**
```python
# Routes through base_model to avoid PEFT injecting kwargs SAM3 doesn't accept
if hasattr(self.backbone, "base_model"):
    out = self.backbone.base_model(pixel_values=pixel_values)
```

**Label noise handling**
Labels are human-buffered approximations. The loss uses boundary pixel weighting via morphological erosion (`F.conv2d`) to down-weight ambiguous buffer edge pixels during training.

**Aerial display fix**
Drone imagery has compressed dynamic range. Visualization applies 2–98 percentile contrast stretch (equivalent to "stretch to min/max" in QGIS) before display.

---

## Training

```python
# Core hyperparameters
SAM3_INPUT_SIZE   = 504
BATCH_SIZE        = 1 per GPU          # T4 16 GB limit at this resolution
accumulation_steps = 8                 # effective batch size = 8 (single GPU) / 16 (dual)
num_epochs        = 15
optimizer         = AdamW(lora_params lr=1e-4, head_params lr=5e-4, weight_decay=1e-4)

# LoRA
r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"]

# Mixed precision
amp_dtype = bfloat16  # on Ampere+ (A100)
amp_dtype = float16   # on T4
```

Training uses gradient accumulation, AMP, gradient clipping (`max_norm=1.0`), and saves the best checkpoint by validation IoU.

---

## Augmentations

Designed for aerial imagery where camera orientation is arbitrary and GSD varies with flight altitude:

| Transform | Purpose |
|---|---|
| `RandomRotate90` + `Rotate(±45°)` | Full orientation invariance (lossless 90° first) |
| `HorizontalFlip` + `VerticalFlip` | Standard aerial augmentation |
| `CLAHE` | Enhance low-contrast roads in shadow |
| `Downscale(0.5–0.9)` | Simulate GSD variation across altitudes |
| `GridDistortion` | Simulate orthorectification residuals |
| `ShiftScaleRotate` | Viewpoint variation |
| `GaussNoise` + `MotionBlur` | Sensor noise and motion blur |

---

## Results

Metric is standard IoU against human-buffered labels. Note that label boundaries are imprecise (buffer edge ≠ actual road edge), so IoU against these labels has an inherent ceiling of approximately 0.55–0.65 regardless of prediction quality. Visual inspection is a better indicator of real performance.

---

## File Overview

```
sam3-peft.ipynb          # main training notebook
README.md                # this file
```

Checkpoint contains: `model_state_dict`, `optimizer_state_dict`, `best_val_iou`, `history`, and training `config`.

---

## Some Results
<img width="761" height="650" alt="image" src="https://github.com/user-attachments/assets/d3885f55-5365-4bba-a63e-ca79981b234f" />
<img width="769" height="675" alt="image" src="https://github.com/user-attachments/assets/a384adef-9b72-4604-b16c-d0d527f4a041" />
<img width="727" height="648" alt="image" src="https://github.com/user-attachments/assets/2f7ca85d-8a08-4e52-abc2-964a45227cff" />
<img width="1635" height="632" alt="image" src="https://github.com/user-attachments/assets/182d6976-e5bf-4a58-b852-e1c090879c15" />






## Limitations

- **Label quality ceiling** — masks are human-drawn buffers over OSM road centrelines. Buffer width is inconsistent and boundaries do not align precisely with road edges in imagery.
- **No decoder** — SAM3's pretrained mask decoder cannot be used without prompt tokens (text + boxes). The CNN head is a reasonable substitute but produces lower-resolution boundaries.
- **Single class** — binary road / no-road only. Lane-level or road type classification would require a different label scheme.
- **Memory** — at 504² resolution, batch size is limited to 1 per T4 GPU. Scaling to 1024² requires gradient checkpointing and `batch_size=1` with at least 24 GB VRAM.

---

## References

- [Segment Anything Model 3 (SAM3)](https://huggingface.co/facebook/sam3)
- [PEFT — Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Albumentations](https://albumentations.ai/)
- [OpenStreetMap](https://www.openstreetmap.org/)
