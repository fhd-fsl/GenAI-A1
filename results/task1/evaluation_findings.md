# Task 1: Universal Autoencoder Evaluation Findings

## Metrics Summary (Best Trial: #14)

### 📊 Per Corruption Type
| Corruption Type | L1 Loss (Lower = Better) | SSIM (Higher = Better) |
| :--- | :--- | :--- |
| **Clean (Baseline)** | 0.0701 | 0.5137 |
| **Gaussian Blur** | **0.0699** (Best) | **0.5126** (Best) |
| **Salt & Pepper** | 0.0712 | 0.5077 |
| **Rectangular Occlusion** | 0.0836 (Worst) | 0.4715 (Worst) |

### 📈 Per Severity Level
| Severity Level | L1 Loss | SSIM |
| :--- | :--- | :--- |
| **Low** | 0.0721 | 0.5061 |
| **Medium** | 0.0744 | 0.4984 |
| **High** | 0.0781 | 0.4872 |

## Key Findings & Observations
1. **Performance Disparity by Corruption Type**: The Universal Autoencoder performs almost as well as the baseline (clean images) on `Gaussian Blur` and `Salt & Pepper` noise. This demonstrates that the 4-layer CNN architecture and the bottleneck dimension of 512 are fully capable of resolving local, high-frequency spatial noise. 
2. **Struggle with Occlusion**: The model struggles significantly more with `Rectangular Occlusion`. This represents a fundamental limitation of universal generic autoencoders: blurring and noise can be "smoothed" or "filtered" using surrounding local pixels, but a large missing black square requires the model to semantically "hallucinate" entirely missing structural information (e.g., imagining a dog's eye that was completely deleted), which this model lacks the deep generative capacity to do effectively.
3. **Severity Degradation**: As expected mathematically, the loss increases and the Structural Similarity Index Measure (SSIM) monotonically decreases as the severity of the noise increases from low to high.

## Visual Grid Outputs
*Please reference the generated image files in this directory for the visual components of the final report.*
- **`representative_examples.png`**: Contains 12 randomly selected test samples (using reservoir sampling) with a combined loss below 0.15, demonstrating typical performance characteristics.
- **`failure_cases.png`**: Contains the 4 worst-performing test instances evaluated according to the model's exact Combined Loss function (Alpha-weighted L1 + SSIM) dynamically extracted from the best hyperparameter trial (`alpha ≈ 0.958`).

## Hyperparameter Optimization (Optuna)
As required by the assignment, an Optuna study was conducted to find the optimal architecture and training parameters. 

**Search Space:**
- **Learning Rate (`lr`)**: `[1e-5, 1e-2]` (Log Scale)
- **Batch Size (`batch_size`)**: `[16, 32, 64]`
- **Bottleneck Dimension (`bottleneck_dim`)**: `[64, 128, 256, 320, 512]`
- **Base Channels (`base_channels`)**: `[32, 48, 64]`
- **Dropout Rate (`dropout`)**: `[0.0, 0.5]`
- **L1/SSIM Loss Weight (`alpha`)**: `[0.5, 1.0]`

**Results:**
- **Total Trials Executed**: 20
- **Best Trial**: Trial #14 (Validation Loss: `0.0900`)
- **Selected Configuration**: 
  - `lr`: 0.000113
  - `batch_size`: 32
  - `bottleneck_dim`: 512
  - `base_channels`: 48
  - `dropout`: 0.0256
  - `alpha`: 0.9582 (Heavily weighting L1 over SSIM to prevent structural hallucinations)
