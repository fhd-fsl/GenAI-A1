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
