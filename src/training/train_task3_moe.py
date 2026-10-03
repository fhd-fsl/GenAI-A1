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
from src.utils.trainer_utils import EarlyStopping

CORRUPTION_MAP = {
    "clean": 0,
    "salt_and_pepper": 1,
    "gaussian_blur": 2,
    "rectangular_occlusion": 3
}

def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load best HPO params
    study = optuna.load_study(
        study_name="task3_moe_hpo",
        storage="sqlite:///optuna_studies/task3_moe.db"
    )
    params = study.best_trial.params
    print(f"Loaded best hyperparameters from Optuna DB (Trial {study.best_trial.number}):")
    print(params)

    # 2. Setup Model
    model, _, _ = SoftMoE.from_pretrained(device=device)
    model.tau.fill_(params["temperature"])

    # 3. Setup Data
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

    # 4. Setup Loss
    criterion = MoEJointLoss(
        lambda_recon=params["lambda_recon"],
        lambda_ssim=params["lambda_ssim"],
        lambda_ce=params["lambda_ce"],
        lambda_balance=params["lambda_balance"]
    ).to(device)

    wandb.init(project="GenAI_A1", name="Task3_MoE_Final", config=params)

    # =========================================================================
    # STAGE 1: WARM-UP
    # =========================================================================
    for param in model.expert_sp.parameters():
        param.requires_grad = False
    for param in model.expert_blur.parameters():
        param.requires_grad = False
    for param in model.expert_occ.parameters():
        param.requires_grad = False

    optimizer_warmup = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=params["warmup_lr"]
    )

    warmup_epochs = 10
    print(f"\n--- Stage 1: Warm-up ({warmup_epochs} epochs) ---")
    for epoch in range(warmup_epochs):
        model.train()
        train_loss = 0.0
        
        for corrupted, clean, labels in train_loader:
            corrupted, clean, labels = corrupted.to(device), clean.to(device), labels.to(device)
            optimizer_warmup.zero_grad()
            x_hat, weights, logits = model(corrupted)
            loss, _ = criterion(x_hat, clean, logits, weights, labels)
            loss.backward()
            optimizer_warmup.step()
            train_loss += loss.item()
            
        print(f"Warm-up Epoch {epoch+1}/{warmup_epochs} | Loss: {train_loss/len(train_loader):.4f}")

    # =========================================================================
    # STAGE 2: JOINT FINE-TUNING
    # =========================================================================
    for param in model.parameters():
        param.requires_grad = True

    optimizer_finetune = torch.optim.Adam(model.parameters(), lr=params["finetune_lr"])

    checkpoint_dir = "checkpoints/task3_moe"
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "best_model.pt")
    early_stopping = EarlyStopping(patience=5, min_delta=0.0005, save_path=save_path)

    finetune_epochs = 100
    print(f"\n--- Stage 2: Joint Fine-tuning (max {finetune_epochs} epochs) ---")
    for epoch in range(finetune_epochs):
        model.train()
        train_loss = 0.0
        
        for corrupted, clean, labels in train_loader:
            corrupted, clean, labels = corrupted.to(device), clean.to(device), labels.to(device)
            optimizer_finetune.zero_grad()
            x_hat, weights, logits = model(corrupted)
            loss, _ = criterion(x_hat, clean, logits, weights, labels)
            loss.backward()
            optimizer_finetune.step()
            train_loss += loss.item()

        model.eval()
        val_recon_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for corrupted, clean, metadata in val_loader:
                corrupted, clean = corrupted.to(device), clean.to(device)
                x_hat, weights, logits = model(corrupted)
                val_recon_loss += torch.nn.functional.l1_loss(x_hat, clean).item() * clean.size(0)
                val_count += clean.size(0)
                
        val_recon_loss /= val_count
        
        wandb.log({
            "final_val_recon": val_recon_loss,
            "epoch": warmup_epochs + epoch
        })

        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Val L1: {val_recon_loss:.5f}")

        early_stopping(val_recon_loss, model)
        if early_stopping.early_stop:
            print(f"  [INFO] Early stopping triggered at epoch {epoch+1}")
            break

    wandb.finish()
    print(f"\n[SUCCESS] Best Soft MoE model saved to {save_path}")

if __name__ == "__main__":
    main()
