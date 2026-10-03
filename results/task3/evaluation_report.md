# Task 3: Soft Mixture-of-Experts Evaluation

## Test Set Global Metrics (N=36690)
| Metric | Value |
|--------|-------|
| L1 Loss | 0.05281 |
| SSIM | 0.70292 |
| Avg Inference Time | 0.14 ms/image |

## Per-Corruption Metrics
| Corruption | Count | L1 Loss | SSIM |
|------------|-------|---------|------|
| Clean | 3669 | 0.00873 | 0.99043 |
| Salt & Pepper | 11007 | 0.06947 | 0.52070 |
| Gaussian Blur | 11007 | 0.04000 | 0.74703 |
| Rect. Occlusion | 11007 | 0.06364 | 0.74518 |

## Routing Weight Analysis (Type & Severity)
| Condition | Identity | S&P Expert | Blur Expert | Occ Expert |
|-----------|----------|------------|-------------|------------|
| Clean | 0.8719 | 0.0073 | 0.0382 | 0.0826 |
| Salt & Pepper (low) | 0.0934 | 0.7525 | 0.0993 | 0.0547 |
| Salt & Pepper (medium) | 0.0016 | 0.9683 | 0.0057 | 0.0245 |
| Salt & Pepper (high) | 0.0000 | 0.9796 | 0.0001 | 0.0203 |
| Gaussian Blur (low) | 0.6579 | 0.0043 | 0.3180 | 0.0197 |
| Gaussian Blur (medium) | 0.5669 | 0.0045 | 0.4121 | 0.0164 |
| Gaussian Blur (high) | 0.5600 | 0.0056 | 0.4145 | 0.0199 |
| Rect. Occlusion (low) | 0.7600 | 0.0101 | 0.0173 | 0.2126 |
| Rect. Occlusion (medium) | 0.6074 | 0.0107 | 0.0100 | 0.3719 |
| Rect. Occlusion (high) | 0.4274 | 0.0098 | 0.0066 | 0.5563 |

## Expert Activity & Dominance Check
### 1. Inactivity Check
- [PASS] All experts are active and receiving >5% of the overall routing volume.

### 2. Unrelated Dominance Check
- [WARNING] **Identity** dominates **Gaussian Blur (low)** images (Weight: 0.6579)
- [WARNING] **Identity** dominates **Gaussian Blur (medium)** images (Weight: 0.5669)
- [WARNING] **Identity** dominates **Gaussian Blur (high)** images (Weight: 0.5600)
- [WARNING] **Identity** dominates **Rect. Occlusion (low)** images (Weight: 0.7600)
- [WARNING] **Identity** dominates **Rect. Occlusion (medium)** images (Weight: 0.6074)
