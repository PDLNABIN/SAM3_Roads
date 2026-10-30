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
Drone Image (1024x1024)
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
Bilinear upsample -> Binary road mask (1024x1024)
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
image_size = 1024
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
Training also supports early stopping based on validation IoU. Set
`training.early_stopping_patience` to the number of consecutive non-improving
epochs to allow, or `0` to disable it. `training.early_stopping_min_delta`
controls the minimum IoU improvement required.

## Outputs

Outputs are written to `outputs/sam3_roads/` by default:

```text
outputs/sam3_roads/
|-- sam3_lora_best.pth
|-- history.csv
|-- training_curves.png
|-- prediction_preview.png
|-- test_metrics.json
|-- validation_metrics.json
|-- training_metrics.json
|-- validation_predictions/
|   |-- <sample>_image.png
|   |-- <sample>_ground_truth.png
|   |-- <sample>_prediction.png
|   |-- <sample>_triplet.png
|   `-- predictions.json
|-- test_predictions/
|   |-- <sample>_image.png
|   |-- <sample>_ground_truth.png
|   |-- <sample>_prediction.png
|   |-- <sample>_triplet.png
|   `-- predictions.json
`-- portfolio_stats.json
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

## Evaluation and portfolio exports

The Kaggle configuration uses the published Kathmandu road dataset with 575
training, 191 validation, and 225 held-out test tiles:

```bash
poetry run sam3-roads evaluate \
  --config configs/sam3_roads.kaggle.toml \
  --checkpoint outputs/kaggle_sam3_roads/sam3_lora_best.pth
poetry run sam3-roads export-portfolio-assets \
  --config configs/sam3_roads.kaggle.toml \
  --checkpoint outputs/kaggle_sam3_roads/sam3_lora_best.pth \
  --output portfolio/assets
```

Evaluation reports cumulative Road IoU, F1/Dice, precision, recall, pixel
accuracy, Boundary IoU with a three-pixel dilation radius, and relaxed
precision/recall with a three-pixel buffer. The profiler also records total
and trainable parameters, synchronized CUDA latency, throughput, and peak
memory.

Every training run also preserves epoch-level training loss and validation
loss, IoU, F1/Dice, precision, recall, Boundary IoU, and relaxed metrics in
`history.csv` and `training_metrics.json`. The validation and test prediction
directories preserve the source RGB image, ground-truth mask, predicted mask,
and a three-panel visual triplet for every sample, plus a JSON manifest.

## Key Design Decisions

**`image_size = 1024`**

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
- At 1024x1024 resolution, batch size is memory-limited. If a GPU runs out
  of memory, reduce `training.batch_size` to `1` and increase
  `training.accumulation_steps` to `8` to preserve the same effective batch
  size.
- In environments with a small `/dev/shm` (shared-memory) mount, use
  `training.num_workers = 0` to avoid DataLoader worker allocation errors.

## References

- [Segment Anything Model 3 (SAM3)](https://huggingface.co/facebook/sam3)
- [PEFT - Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Albumentations](https://albumentations.ai/)
- [OpenStreetMap](https://www.openstreetmap.org/)
