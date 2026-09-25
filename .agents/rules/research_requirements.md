# Research and Analysis Requirements

The assignment strictly requires independent research, analysis, and justification for technical decisions. "Simply reproducing an existing implementation or accepting AI-generated code without investigation will not satisfy the research component."

When implementing the following components, agents and developers MUST perform analysis and document findings for the final report:

## Task 1: Universal Autoencoder
- **Architecture (Skip Connections)**: If limited skip connections are used (e.g., U-Net style instead of strict bottleneck), their purpose and effect MUST be investigated and justified. Prove why they do not break the compression bottleneck.
- **Loss Function Weighting**: The L1 vs. SSIM weighting ($\alpha$) must be selected via Optuna, not accepted blindly. Document how different $\alpha$ values affect the visual output (e.g., color accuracy vs. structural sharpness).

## Task 2: Classifier and Hard-Routed Specialists
- **Classifier Architecture & Regularization**: Investigate and justify the chosen base architecture. Analyze how weight decay and dropout prevent overfitting to the synthetic corruptions.
- **Error Analysis**: Identify and discuss specific cases where classifier errors cause restoration failures (e.g., why did the classifier fail to detect a light blur?).
- **Specialist Architecture**: Justify the shared architecture used for the specialists. Is it the same as Task 1? Why or why not?

## Task 3: Soft Mixture-of-Experts
- **Balance/Entropy Regularizer**: You may propose a different differentiable balance or entropy regularizer (other than the provided variance-based one), but it MUST be supported by research and clearly justified (e.g., referencing Shazeer's sparsely-gated MoE).
- **Routing Behavior Analysis**: Analyze the gating temperature ($\tau$). Document how the network learns to mix experts. Determine and discuss if any expert becomes inactive or if one dominates unrelated inputs.

## General (All Tasks)
- **Hyperparameter Bounds**: Justify the search space bounds provided to Optuna based on literature or preliminary empirical testing.
- **Optuna Analysis**: Analyze parameter importance plots to explain which hyperparameters mattered most for each task. Every important table/diagram must be interpreted in the report.
