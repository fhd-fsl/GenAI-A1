# Task 2: Corruption Classifier Evaluation Findings

## M3.1 Metrics Summary (Best Trial: #20)

### 📂 Dataset Statistics (Official Test Set)
- **Base Test Images**: 3,669 (Untouched official Oxford-IIIT Pet test split)
- **Deterministic Corruptions per Image**: 10 (1 Clean + 3 S&P Severities + 3 Blur Severities + 3 Occlusion Severities)
- **Total Evaluated Samples**: 36,690
### 📊 Overall Performance (Test Set)
- **Accuracy**: 98.26%
- **Macro Precision**: 96.65%
- **Macro Recall**: 97.88%
- **Macro F1-Score**: 97.21%

### 🎯 Per-Class Metrics (F1-Score)
| Corruption Class | F1-Score |
| :--- | :--- |
| **Clean** | 91.89% |
| **Salt & Pepper** | 99.98% |
| **Gaussian Blur** | 99.41% |
| **Rectangular Occlusion** | 97.57% |

## Key Findings & Observations
1. **Outstanding Corruption Detection**: The classifier is nearly perfect at identifying artificial corruptions, achieving >99% F1-scores on both Salt & Pepper and Gaussian Blur. The network's spatial invariance (achieved via Adaptive Average Pooling) allows it to detect Rectangular Occlusions with 97.57% accuracy regardless of where the mask is located on the image.
2. **The "Clean" Baseline Tradeoff**: The only class with a slightly lower F1-Score is "Clean" (91.89%). This occurs because extremely subtle corruptions (like a very low-severity blur or a tiny occlusion mask) can occasionally be misclassified as "clean" by the model, or vice-versa. However, since a "clean" prediction bypasses the hard-routed specialists entirely, this is an acceptable tradeoff; it is better to leave a slightly blurry image untouched than to force it through a heavy reconstruction specialist unnecessarily.

## Visual Artifacts
- **`classifier_confusion_matrix.png`**: A normalized 4-class confusion matrix visually demonstrating the extremely high diagonal accuracy and the slight overlap between "clean" and low-severity corruptions.
