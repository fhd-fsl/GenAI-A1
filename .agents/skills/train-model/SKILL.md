---
name: train-model
description: >
  How to train models in this project. Covers architecture patterns, loss functions,
  training loops, checkpointing, and experiment tracking for all 4 tasks.
---

# Model Training Skill

## General Training Pattern

All training scripts follow this structure:
```python
# scripts/train_task{N}.py
1. Load config from configs/task{N}.yaml
2. Set seeds (torch, numpy, random) = 42
3. Create dataset & dataloaders
4. Instantiate model, optimizer, scheduler
5. Initialize experiment tracker (MLflow/W&B)
6. Training loop with:
   - Train epoch → log train loss
   - Validation epoch → log val loss + metrics
   - Checkpoint best model
   - Early stopping (patience configurable)
7. Save final model to checkpoints/task{N}/
```

## Task 1 — Universal Denoising Autoencoder
- **Architecture**: Conv Encoder → Bottleneck → Conv Decoder
- **Input**: corrupted 128×128 RGB (any of 4 conditions)
- **Target**: clean 128×128 RGB
- **Loss**: `L = α·L1(x, x̂) + (1-α)·(1-SSIM(x, x̂))`
- **Key**: Model must NOT know which corruption was applied
- **Skip connections**: allowed but must be limited and justified

## Task 2 — Classifier + Specialist AEs

### Classifier
- **Architecture**: Conv layers → global pool → FC → 4 classes
- **Loss**: Cross-entropy
- **IMPORTANT**: Balanced batches (equal samples per corruption class)
- **Metrics**: accuracy, macro P/R/F1, per-class, confusion matrix

### Specialist AEs (×3)
- Salt-and-pepper specialist: trained ONLY on S&P corruptions
- Blur specialist: trained ONLY on blur corruptions
- Occlusion specialist: trained ONLY on occlusion corruptions
- Same base architecture, independent weights
- Clean images → identity bypass (no specialist needed)

## Task 3 — Soft Mixture-of-Experts

### Two-Stage Training
1. **Warm-up**: Freeze experts (from T2), train gating network only
2. **Joint fine-tune**: Unfreeze all, use smaller LR

### Components
- **Gating network**: initialized from T2 classifier
- **Experts**: initialized from T2 specialists
- **Identity branch**: passes input through unchanged

### Joint Loss
```
L_joint = λ₁·L1 + λ₂·(1-SSIM) + λ₃·L_CE + λ₄·L_balance
L_balance = Σ(w̄_k - 1/4)²  (over K=4 branches)
```
Initial: λ₁=0.8, λ₂=0.2, λ₃=0.1, λ₄=0.01

## Task 4 — Conditional GAN

### Generator (U-Net)
- Input: photo (128×128) + style embedding
- Style: learned categorical embedding (dim tuned by Optuna)
- Output: sketch (128×128)

### Discriminator (PatchGAN)
- Input: (photo, sketch, style_embedding)
- Output: patch-level real/fake scores

### Losses
- **G_loss** = L_adv + λ_L1 · L1(y, G(x,s))  — initial λ_L1 = 100
- **D_loss** = BCE(D(x,y_real,s), 1) + BCE(D(x,G(x,s),s), 0)

### Logging Requirements
- D_real_loss, D_fake_loss, G_adv_loss, G_recon_loss
- Validation metrics per epoch
- Sample generations at fixed intervals (same val images)

## Experiment Tracking
All training runs MUST log:
- Hyperparameters
- Train/val losses per epoch
- Best metrics
- Model checkpoint path
- Visual samples (periodic)
