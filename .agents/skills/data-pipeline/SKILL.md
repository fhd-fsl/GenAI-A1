---
name: data-pipeline
description: >
  How to build and test the data loading pipelines for this project.
  Covers Oxford-IIIT Pet corruption pipeline (Tasks 1-3) and FS2K paired loading (Task 4).
---

# Data Pipeline Skill

## Oxford-IIIT Pet Corruption Pipeline (Tasks 1–3)

### Dataset
- Download via `torchvision.datasets.OxfordIIITPet` or manual download
- All images → RGB → resize 128×128
- Split: 80/20 train/val with `sklearn.model_selection.train_test_split(seed=42)`

### Runtime Corruption (Training)
For each image loaded during training, randomly select ONE of 4 conditions with equal probability:

1. **Clean** — return image unchanged, label = 0
2. **Salt-and-pepper** — sample p ~ U(0.02, 0.15), flip pixels to 0 or 1 with prob p/2 each, label = 1
3. **Gaussian blur** — kernel ∈ {3,5,7}, σ ~ U(0.5, 2.5), label = 2
4. **Rectangular occlusion** — 1–3 black rectangles, total 10–35% of area, label = 3

### Deterministic Manifests (Validation & Test)
Store as JSON files in `manifests/`:
```json
{
  "image_path": "path/to/image.jpg",
  "corruption_type": "salt_and_pepper",
  "severity": "medium",
  "params": {"probability": 0.08},
  "seed": 12345
}
```

**Test manifest** must have 3 severity levels per corruption:
| Corruption | Low | Medium | High |
|-----------|-----|--------|------|
| S&P | p=0.03 | p=0.08 | p=0.15 |
| Blur | (3, 0.7) | (5, 1.5) | (7, 2.5) |
| Occlusion | ~10%, 1 rect | ~20%, 2 rects | ~35%, 3 rects |

### Implementation Location
- `src/data/corruptions.py` — corruption functions
- `src/data/pet_dataset.py` — PyTorch Dataset class
- `src/data/manifest.py` — manifest generation & loading
- `scripts/generate_manifests.py` — CLI to create val/test manifests

## FS2K Pipeline (Task 4)

### Dataset
- 2,104 paired photos + sketches, 3 style categories
- Use official train/test splits
- Reserve 15% of train as validation (stratified by style, seed 42)

### Paired Augmentation Rule
**CRITICAL**: Any spatial augmentation (flip, rotate, crop) MUST be applied identically to BOTH the photo and sketch. Use the same random state for both transforms.

### Implementation Location
- `src/data/fs2k_dataset.py` — PyTorch Dataset class
- Style labels loaded from dataset metadata
