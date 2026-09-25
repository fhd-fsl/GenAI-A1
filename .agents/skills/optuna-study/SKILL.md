---
name: optuna-study
description: >
  How to set up and run Optuna hyperparameter optimization studies for each task.
  Includes search spaces, objectives, pruning strategies, and reporting requirements.
---

# Optuna Study Skill

## General Setup

```python
import optuna

study = optuna.create_study(
    study_name="task{N}_hpo",
    direction="minimize",  # or "maximize" depending on metric
    storage="sqlite:///optuna_studies/task{N}.db",
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
)
study.optimize(objective, n_trials=N)
```

## Task-Specific Search Spaces

### Task 1 — Universal AE
```python
lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
bottleneck_dim = trial.suggest_int("bottleneck_dim", 64, 512, step=64)
n_channels = trial.suggest_categorical("base_channels", [32, 48, 64])
dropout = trial.suggest_float("dropout", 0.0, 0.5)
alpha = trial.suggest_float("alpha", 0.5, 1.0)  # L1 vs SSIM weight
```
**Objective**: val_loss = α·L1 + (1-α)·(1-SSIM)

### Task 2 — Classifier
```python
lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
channels = trial.suggest_categorical("base_channels", [16, 32, 64])
dropout = trial.suggest_float("dropout", 0.1, 0.5)
weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
```
**Objective**: maximize val_accuracy (or minimize 1 - val_macro_f1)

### Task 2 — Specialist AEs (shared search)
```python
lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
bottleneck_dim = trial.suggest_int("bottleneck_dim", 64, 512, step=64)
channels = trial.suggest_categorical("base_channels", [32, 48, 64])
batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
alpha = trial.suggest_float("alpha", 0.5, 1.0)
```
**Objective**: average val_loss across all 3 specialists

### Task 3 — Soft MoE Joint Fine-tuning
```python
lr = trial.suggest_float("joint_lr", 1e-6, 1e-3, log=True)
tau = trial.suggest_float("temperature", 0.1, 5.0)
lambda_ce = trial.suggest_float("lambda_ce", 0.01, 0.5)
lambda_balance = trial.suggest_float("lambda_balance", 0.001, 0.1, log=True)
lambda_recon = trial.suggest_float("lambda_recon", 0.5, 1.0)
```
**Objective**: val_reconstruction_loss  
**Pruning**: Prune if routing collapse detected (one expert > 90% weight)

### Task 4 — Conditional GAN
```python
g_lr = trial.suggest_float("g_lr", 1e-5, 1e-3, log=True)
d_lr = trial.suggest_float("d_lr", 1e-5, 1e-3, log=True)
batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
base_channels = trial.suggest_categorical("base_channels", [32, 64, 128])
dropout = trial.suggest_float("dropout", 0.0, 0.5)
embed_dim = trial.suggest_categorical("embed_dim", [8, 16, 32, 64])
lambda_l1 = trial.suggest_float("lambda_l1", 10.0, 200.0)
```
**Objective**: val_generator_loss (or FID if feasible)  
**Note**: Use fewer epochs per trial (GAN is expensive), retrain best with full schedule

## Reporting Requirements
For each Optuna study, the report must include:
1. Complete search space definition
2. Number of completed trials
3. Best trial parameters
4. Optimization history plot
5. Parameter importance plot
6. Final selected configuration with justification

## Storage
- SQLite DBs in `optuna_studies/`
- Best config saved to `configs/task{N}_best.yaml`
