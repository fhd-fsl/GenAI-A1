# Task 2: Hard-Routed Inference Evaluation

## Setup
This report evaluates the complete Task 2 pipeline, which consists of:
- **Corruption Classifier:** Dynamically routes corrupted inputs to the appropriate restoration expert.
- **Specialist Autoencoders:** Three independent models (Salt & Pepper, Gaussian Blur, Rectangular Occlusion) tuned via a shared Optuna search.
- **Identity Bypass:** Clean images are mathematically bypassed directly to the output without passing through any autoencoder.

## Final Test Set Metrics (N=36,690)
| Routing Mode | L1 Loss ↓ | SSIM ↑ |
|--------------|-----------|--------|
| **Oracle** (Ground Truth Labels) | 0.06814 | 0.53327 |
| **Predicted** (Classifier Driven) | **0.06774** | **0.53699** |

## Findings & Anomalies
Interestingly, the **Predicted Routing outperformed the Oracle Routing** by a marginal fraction. 
This counterintuitive result implies that in borderline cases where the synthetic corruption was extremely subtle (e.g., a Gaussian blur with `sigma=0.5`), the Classifier's "mistake" to route the image to the Identity Bypass (Clean) or another specialist actually resulted in a mathematically superior reconstruction than forcing it through the dedicated expert, which may have over-smoothed or over-corrected the barely-corrupted image.

## Visual Failure Cases
The generated grid at `results/task2/routing_failures.png` explicitly highlights instances where the Classifier's misrouting led to a distinct visual output compared to the Oracle's mathematically mandated routing.
