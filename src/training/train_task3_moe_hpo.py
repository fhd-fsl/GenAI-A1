import os
import sys
import torch
import optuna
import wandb
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.moe_loss import MoEJointLoss
from src.data.pet_dataset import OxfordPetDataset, balanced_corruption_collate_fn
from src.models.task3.soft_moe import SoftMoE

CORRUPTION_MAP = {
    "clean": 0,
    "salt_and_pepper": 1,
    "gaussian_blur": 2,
    "rectangular_occlusion": 3
}


def objective(trial: optuna.Trial) -> float:
    """Optuna objective for Soft MoE hyperparameter search."""
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # 1. Sample hyperparameters
    warmup_lr = trial.suggest_float("warmup_lr", 1e-5, 1e-3, log=True)
    finetune_lr = trial.suggest_float("finetune_lr", 1e-6, 1e-4, log=True)
    tau = trial.suggest_float("temperature", 0.1, 5.0)
    
    # Loss weights (independent)
    lambda_ce = trial.suggest_float("lambda_ce", 0.01, 0.5)
    lambda_balance = trial.suggest_float("lambda_balance", 0.001, 0.1, log=True)
    lambda_recon = trial.suggest_float("lambda_recon", 0.1, 1.0)
    lambda_ssim = trial.suggest_float("lambda_ssim", 0.1, 1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load pre-trained model
    model, _, _ = SoftMoE.from_pretrained(device=device)
    model.tau.fill_(tau)

    # 3. DataLoaders
    config = load_config()
    data_dir = os.path.join(config.get("data", {}).get("dir", "./data"), "oxford-iiit-pet")
    manifests_dir = config.get("data", {}).get("manifests_dir", "./manifests")

    train_dataset = OxfordPetDataset(data_dir=data_dir, split="train")
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        collate_fn=balanced_corruption_collate_fn, num_workers=0
    )

    val_dataset = OxfordPetDataset(data_dir=data_dir, split="val", manifests_dir=manifests_dir)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 4. Loss & Optimizer
    criterion = MoEJointLoss(
        lambda_recon=lambda_recon, lambda_ssim=lambda_ssim,
        lambda_ce=lambda_ce, lambda_balance=lambda_balance
    ).to(device)

    # --- Stage 1: Warm-up (freeze experts, train gate only) ---
    for param in model.expert_sp.parameters():
        param.requires_grad = False
    for param in model.expert_blur.parameters():
        param.requires_grad = False
    for param in model.expert_occ.parameters():
        param.requires_grad = False

    optimizer_warmup = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=warmup_lr
    )

    wandb.init(
        project="GenAI_A1",
        name=f"Task3_MoE_HPO_trial_{trial.number}",
        config=trial.params, reinit=True
    )

    warmup_epochs = 10
    finetune_epochs = 15

    print(f"\n--- Trial {trial.number} | tau={tau:.3f} ---")
    print(f"  Stage 1: Warm-up ({warmup_epochs} epochs, gate only, lr={warmup_lr:.6f})")

    for epoch in range(warmup_epochs):
        model.train()
        for corrupted, clean, labels in train_loader:
            corrupted, clean, labels = corrupted.to(device), clean.to(device), labels.to(device)
            optimizer_warmup.zero_grad()
            x_hat, weights, logits = model(corrupted)
            loss, _ = criterion(x_hat, clean, logits, weights, labels)
            loss.backward()
            optimizer_warmup.step()

    # --- Stage 2: Joint fine-tuning (unfreeze all) ---
    for param in model.parameters():
        param.requires_grad = True

    optimizer_finetune = torch.optim.Adam(model.parameters(), lr=finetune_lr)

    print(f"  Stage 2: Joint fine-tune ({finetune_epochs} epochs, lr={finetune_lr:.7f})")

    best_val_recon = float('inf')
    patience = 4
    stagnant = 0

    for epoch in range(finetune_epochs):
        # -- Train --
        model.train()
        epoch_weights_sum = torch.zeros(4, device=device)
        epoch_count = 0

        for corrupted, clean, labels in train_loader:
            corrupted, clean, labels = corrupted.to(device), clean.to(device), labels.to(device)
            optimizer_finetune.zero_grad()
            x_hat, weights, logits = model(corrupted)
            loss, _ = criterion(x_hat, clean, logits, weights, labels)
            loss.backward()
            optimizer_finetune.step()

            epoch_weights_sum += weights.detach().sum(dim=0)
            epoch_count += weights.size(0)

        avg_weights = epoch_weights_sum / epoch_count

        # -- Routing collapse check (over-dominant or inactive) --
        if avg_weights.max().item() > 0.90 or avg_weights.min().item() < 0.05:
            print(f"  [PRUNE] Routing collapse at epoch {epoch}: {avg_weights.cpu().numpy()}")
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

        # -- Validate --
        model.eval()
        val_recon_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for corrupted, clean, metadata in val_loader:
                corrupted, clean = corrupted.to(device), clean.to(device)

                x_hat, weights, logits = model(corrupted)
                # Only track reconstruction quality for the objective
                val_recon_loss += torch.nn.functional.l1_loss(x_hat, clean).item() * clean.size(0)
                val_count += clean.size(0)

        val_recon_loss /= val_count

        wandb.log({
            "val_recon_loss": val_recon_loss,
            "avg_w_clean": avg_weights[0].item(),
            "avg_w_sp": avg_weights[1].item(),
            "avg_w_blur": avg_weights[2].item(),
            "avg_w_occ": avg_weights[3].item(),
            "epoch": warmup_epochs + epoch
        })

        trial.report(val_recon_loss, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

        if val_recon_loss < best_val_recon - 0.0005:
            best_val_recon = val_recon_loss
            stagnant = 0
        else:
            stagnant += 1

        if stagnant >= patience:
            print(f"  [INFO] Early stopping at epoch {warmup_epochs + epoch}")
            break

    wandb.finish()
    return best_val_recon


if __name__ == "__main__":
    os.makedirs("optuna_studies", exist_ok=True)

    study = optuna.create_study(
        study_name="task3_moe_hpo",
        direction="minimize",
        storage="sqlite:///optuna_studies/task3_moe.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    )

    print("==================================================")
    print("     Starting Task 3 Soft MoE Optuna Study")
    print("==================================================")

    study.optimize(objective, n_trials=20)

    print("\n==================================================")
    print("               Study Complete!")
    print("==================================================")
    print(f"Best trial ID: {study.best_trial.number}")
    print(f"Best val reconstruction loss: {study.best_trial.value}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
