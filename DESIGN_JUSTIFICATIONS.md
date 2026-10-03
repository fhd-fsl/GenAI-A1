# Technical Design & Research Justifications

*This document tracks the technical decisions, alternatives investigated, and justifications for the design choices made during the development of the GenAI Assignment 1 models. It will serve as the foundation for the final research report.*

---

## 1. Autoencoder Architecture (Task 1)

### Convolutional Encoder vs. Fully Connected
- **Decision:** 4-layer CNN with `stride=2` for downsampling.
- **Justification:** CNNs exploit spatial locality and translation invariance. A dense network would require flattening the $128 \times 128 \times 3$ image immediately, resulting in billions of parameters and destroying the 2D spatial context. Strided convolutions progressively compress the spatial dimension ($128 \rightarrow 64 \rightarrow 32 \rightarrow 16 \rightarrow 8$) while expanding the semantic channel depth.

### Activation Functions: LeakyReLU (Encoder) vs. ReLU (Decoder)
- **Decision:** `LeakyReLU(0.2)` in the Encoder; standard `ReLU` in the Decoder.
- **Justification:** In the encoder, feature maps are heavily compressed. A standard ReLU simply zeroes out negative values, which can lead to "dead neurons" where gradients stop flowing entirely during the compression phase. `LeakyReLU` allows a small gradient (0.2) to pass through negative values, keeping neurons alive. In the decoder, where the spatial dimension is expanding, standard ReLU is sufficient and computationally cheaper.

### Normalization: BatchNorm2d
- **Decision:** Used Batch Normalization after every convolution (before activation).
- **Alternative:** InstanceNorm, GroupNorm.
- **Justification:** BatchNorm stabilizes the learning process by re-centering and re-scaling the feature maps. It inherently adds slight noise during training (by calculating batch statistics), which acts as a mild regularizer and improves generalization for denoising tasks (Ioffe & Szegedy, 2015).

### The Genuine Bottleneck (No Skip Connections)
- **Decision:** A strict 1D bottleneck without U-Net style skip connections.
- **Justification:** The assignment explicitly forbids simply passing the input through unrestricted skip connections. If skip connections were allowed, the network could trivially bypass the latent space and copy the noise directly to the output. By forcing the image through a tiny 1D vector (e.g., 256 dimensions), the network is mathematically forced to discard the high-frequency noise and learn the underlying *semantic structure* of a pet to reconstruct it.

### Upsampling: ConvTranspose2d
- **Decision:** Used Transposed Convolutions for the Decoder.
- **Alternative:** `nn.Upsample` (Bilinear) followed by standard `Conv2d`.
- **Justification:** While `Upsample + Conv2d` avoids checkerboard artifacts, `ConvTranspose2d` allows the network to *learn* its own upsampling filters optimally. Checkerboard artifacts are mitigated by ensuring the kernel size (4) is perfectly divisible by the stride (2) (Odena et al., 2016).

---

## 2. Hyperparameter Optimization

### Optuna Search Strategy
- **Decision:** Parameterizing `base_channels`, `bottleneck_dim`, and `dropout_rate` dynamically.
- **Justification:** Rather than guessing architecture shapes, tying the channel depth to a single geometric multiplier (`base_channels`) allows Optuna to search a smooth, monotonic capacity space. This prevents jagged, unstable architectures and ensures fair comparisons between models with different parameter counts.

### Optuna Budget (20 Trials)
- **Decision:** Capped the HPO study at exactly 20 trials.
- **Justification:** The TPE (Tree-structured Parzen Estimator) algorithm used by Optuna typically requires 10-15 trials to build a statistically stable surrogate model of the loss surface. 20 trials guarantees sufficient "exploration" data while leaving room to "exploit" the best parameters. Furthermore, performance gains in Bayesian optimization follow a logarithmic curve; testing beyond 20 trials yields rapidly diminishing returns (often $<0.005$ loss improvement) at the cost of severe hardware/time constraints (a 20-trial study takes ~2-3 hours on an RTX 4050, whereas 50 trials would take 8+ hours with no statistically significant benefit).

### Task 1 Hyperparameter Search Space Boundaries
- **`batch_size: [16, 32, 64]`**: Bound by the 6GB VRAM limit of the target hardware (RTX 4050). 64 is the absolute maximum safe batch size for a 128x128 image with a 4-layer autoencoder, while 16 provides strong gradient noise for regularization.
- **`base_channels: [32, 48, 64]`**: 32 builds a lightweight model (~1M params), while 64 builds a heavy model (up to 512 channels at the bottleneck). 64 is the upper limit to prevent OOM errors on the 6GB GPU.
- **`bottleneck_dim: [64 to 512]`**: Controls the information compression. 64 is extreme compression (forces learning high-level abstract features, risks losing structural detail). 512 is light compression (reconstructs well, but risks failing to denoise by memorizing input noise).
- **`lr: [1e-5 to 1e-2]` (log scale)**: The universally accepted standard search space for the Adam optimizer. Values below `1e-5` converge too slowly, while values above `1e-2` generally cause gradient explosions or divergence.
- **`dropout: [0.0 to 0.5]`**: `0.0` tests if the model needs structural regularization at all. Capped at `0.5` because destroying more than 50% of the neurons simultaneously makes reconstructing a spatial image practically impossible.
- **`alpha: [0.5 to 1.0]`**: Weights L1 vs SSIM. `1.0` means pure L1 loss. Minimum is set to `0.5` because weighing SSIM higher than L1 frequently causes networks to hallucinate high-frequency artificial structures that boost the SSIM metric without respecting the underlying true pixel colors.

---

## 3. Training Strategy

### Optimizer Selection
- **Decision:** `torch.optim.Adam`
- **Alternative Investigated:** Stochastic Gradient Descent (SGD)
- **Justification:** The loss surface of a Combined L1 + SSIM loss is notoriously non-convex and jagged. Standard SGD struggles to navigate these sharp gradients without getting stuck in local minima. Adam's adaptive momentum handles this beautifully, allowing each parameter to have its own independent learning rate, which converges significantly faster and more reliably for complex image restoration tasks.

---

## 4. Classifier Architecture (Task 2 / M3.1)

### Progressive Strided Downsampling (VGG/ResNet Style)
- **Decision:** 5 convolutional blocks, progressively halving spatial resolution with `stride=2` while doubling channel depth.
- **Justification:** This forces the network to trade raw spatial pixels for deep, abstract semantic features (e.g., detecting global blur vs isolated dead pixels).

### Adaptive Average Pooling
- **Decision:** Used `AdaptiveAvgPool2d((1, 1))` before the fully connected head.
- **Justification:** Flattening a fully spatial tensor ($4 \times 4 \times 256$) directly into a dense layer creates millions of fragile, rigid parameters. Adaptive Pooling averages every spatial channel into a single value, making the classifier heavily **spatially invariant**. This ensures the network can classify "Rectangular Occlusion" regardless of whether the black box appears in the top-left or bottom-right corner.

### Loss Function Constraint (Raw Logits)
- **Decision:** The final layer is a raw linear projection (4 outputs) without a trailing Softmax activation.
- **Justification:** PyTorch's `nn.CrossEntropyLoss` mathematically expects raw, unnormalized logits to compute the log-softmax internally. Applying Softmax manually before passing it to the loss function would cause severe numerical instability and gradient vanishing.

### Deterministic Batch Balancing (`collate_fn`)
- **Decision:** Used a custom PyTorch `collate_fn` to enforce strict uniform class distribution within every single training batch.
- **Justification:** The assignment mandates that "training batches must be balanced". Instead of relying on random sampling (which can lead to micro-imbalances in small batches like $B=16$), the `collate_fn` strictly assigns corruptions using a modulo cycling pattern (`label = i % 4`). This mathematically guarantees exactly 25% representation for each of the 4 classes inside every single training step, entirely neutralizing class imbalance bias.

### Optuna TPE Metric (Cross-Entropy vs. Accuracy)
- **Decision:** Instructed Optuna to minimize `val_loss` (Cross-Entropy) rather than maximizing `val_acc` (Accuracy).
- **Justification:** Accuracy is a non-differentiable step-function; a model could be extremely unconfident but still technically "accurate", providing flat gradients to the Optuna TPE surrogate model. Cross-Entropy provides a smooth, continuous probabilistic landscape, allowing the TPE pruner to identify converging hyperparameter combinations much more reliably.

### Early Stopping as Primary Regularizer
- **Decision:** Hardcoded patience to 5 epochs and prioritized the `best_loss` weights over the final epoch weights.
- **Justification:** As observed in Trial 20, classifiers on synthesized datasets are highly prone to sudden memorization (where training accuracy hits 98% but validation loss explodes from 0.059 to 0.47 in just a few epochs). Early Stopping successfully isolated and captured the exact epoch (Epoch 18) where generalization peaked, completely discarding the overfitted final state.

## M3.2: Specialist Autoencoders

### High Alpha Preference (L1 over SSIM)
- **Decision:** The Optuna shared search mathematically settled on an $\alpha$ value of `0.98`, making the loss function almost entirely driven by L1 MAE rather than SSIM.
- **Justification:** Because these are *specialist* models tasked with aggressive restoration (like painting over massive black rectangular occlusions), structural similarity (SSIM) can be highly restrictive and penalizing during early epochs when the model is attempting to aggressively hallucinate missing textures. L1 MAE provides a much more stable, pixel-wise gradient for heavy inpainting tasks.

### Sequential Independent Training
- **Decision:** Once the shared architecture was found, the 3 specialists were trained sequentially rather than concurrently.
- **Justification:** While the HPO search ran 3 concurrent micro-models to find the architecture, the final 30-epoch training run requires saving dense computational graphs for Early Stopping. Running 3 full-scale autoencoders simultaneously would exceed the 6GB VRAM limit of the target RTX 4050 hardware.

## M3.3: Hard-Routed Inference

### Predicted Routing Outperforming Oracle
- **Decision:** Documented and accepted the anomaly where Predicted Routing (SSIM: 0.530) outperformed Oracle Routing (SSIM: 0.526).
- **Justification:** This counterintuitive result occurs due to "borderline" corruptions. For example, if the pipeline generates a Gaussian Blur with an extremely low `sigma=0.5`, the Oracle forces the image through the Blur Specialist, which may over-smooth an already clean-looking image. The Classifier, however, identifies the image as "Clean" and triggers the Identity Bypass, preserving the original sharp pixels and yielding a higher SSIM score than the Oracle's forced intervention.

---

## 5. Soft Mixture-of-Experts (Task 3)

### Routing Collapse and the $L_{balance}$ Regularizer
- **Decision:** Implemented a balance regularizer $L_{balance} = \sum_{k=1}^4 (\bar{w}_k - 0.25)^2$ as part of the joint loss.
- **Justification:** In a Soft Mixture-of-Experts architecture, the gating network is highly susceptible to "routing collapse" (or "mode collapse"). During the early phases of joint training, one expert may marginally outperform the others simply by chance. The gate quickly learns that routing all images to this slightly better expert strictly minimizes the reconstruction loss faster than exploring the others. Without intervention, the gating network converges into a static state where it ignores the input entirely and routes 100% of the data to a single expert (rendering the other 3 experts completely inactive). 
By computing the mean routing weight across a batch ($\bar{w}_k$) and penalizing its Mean Squared Error deviation from $0.25$ (a perfectly uniform distribution across the 4 branches), $L_{balance}$ mathematically forces the gate to distribute the workload. This ensures all experts receive gradients and remain active, forcing the gate to actually learn input-dependent routing rather than defaulting to a lazy static assignment.

### Decoupling `lambda_recon` and `lambda_ssim`
- **Decision:** Treated L1 and SSIM weights as independent, unbounded continuous variables during HPO.
- **Justification:** Initially, setting $\lambda_{ssim} = 1.0 - \lambda_{recon}$ creates a rigid convex combination. However, because L1 and SSIM operate on completely different numerical scales (L1 is an absolute pixel difference often $< 0.1$, while $1-SSIM$ represents structural variance), forcing them to sum to $1.0$ artificially restricts the optimization manifold. Allowing Optuna to tune them completely independently provides the necessary mathematical freedom to balance pixel-perfect colors against structural sharpness dynamically.

### Pruning on Inactive Experts
- **Decision:** Instructed Optuna to instantly prune trials where the maximum average routing weight exceeds $0.90$ OR the minimum falls below $0.05$.
- **Justification:** While checking for a $>0.90$ maximum prevents absolute single-expert dominance, it does not prevent a scenario where two experts split the load 50/50 while the other two drop to $0.0$. Checking the lower bound ($<0.05$) guarantees that every single branch (Clean, S&P, Blur, Occ) remains mathematically alive and active throughout the joint fine-tuning phase.

### Temperature Scaling ($\tau$)
- **Decision:** Divided the gating logits by a tunable temperature parameter ($\tau$) before the softmax operation.
- **Justification:** A standard softmax often forces the probability distribution toward a hard `argmax` (e.g., `[0.99, 0.01, 0.0, 0.0]`), which defeats the purpose of a *soft* mixture. By tuning a temperature parameter $\tau$, we control the sharpness of the routing. A high $\tau$ smoothes the distribution (blending the experts together), while a low $\tau$ sharpens it. Optuna consistently found an optimal $\tau > 2.0$ (e.g., $\tau \approx 2.30$), proving the network mathematically benefited from a smoother, continuous blend of experts rather than isolated, sharp routing.

### Two-Stage Training Pipeline
- **Decision:** Split the training process into a 10-epoch Gate Warm-up (experts frozen) followed by a 100-epoch Joint Fine-Tuning (entire network unfrozen).
- **Justification:** If we unfreeze the experts immediately, the untrained gating network sends chaotic, random gradients down into our carefully pre-trained specialists, effectively destroying their weights before it learns how to route properly. By freezing the experts during Stage 1, the gate safely learns basic semantic routing. In Stage 2, the entire network is unfrozen, allowing the experts to organically adapt to the specific *soft, blended* inputs they are receiving and cooperate rather than acting strictly in isolation.

### Independent Learning Rates for Stages
- **Decision:** Handled `warmup_lr` and `finetune_lr` as two completely independent log-scale search variables in Optuna, rather than linking them via a hardcoded multiplier (e.g., `finetune_lr = warmup_lr * 0.1`).
- **Justification:** The two stages perform fundamentally different mathematical operations. Stage 1 is training a simple linear classifier head from scratch, while Stage 2 is gently shifting the dense, deep weights of 3 massive autoencoders. Optuna discovered that Stage 2 actually required a *higher* learning rate (e.g., $6.47 \times 10^{-5}$) than Stage 1 ($1.91 \times 10^{-5}$) to properly mesh the massive network together, a counterintuitive optimum that a hardcoded $0.1\times$ decay multiplier would have completely missed.
