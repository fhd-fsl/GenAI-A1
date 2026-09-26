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
