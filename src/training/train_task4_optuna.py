import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
import wandb
import yaml
import numpy as np

from src.data.fs2k_dataset import FS2KDataset
from src.models.task4.generator import UNetGenerator
from src.models.task4.discriminator import PatchGANDiscriminator
from src.utils.weight_init import weights_init

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def objective(trial):
    # Hyperparameters to tune
    g_lr = trial.suggest_float("g_lr", 1e-5, 1e-3, log=True)
    d_lr = trial.suggest_float("d_lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    base_channels = trial.suggest_categorical("base_channels", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    embed_dim = trial.suggest_categorical("embed_dim", [8, 16, 32, 64])
    lambda_l1 = trial.suggest_float("lambda_l1", 10.0, 200.0)
    
    # Initialize WandB for this trial
    run = wandb.init(
        project="GenAI_A1",
        name=f"task4_optuna_trial_{trial.number}",
        config={
            "g_lr": g_lr, "d_lr": d_lr, "batch_size": batch_size,
            "base_channels": base_channels, "dropout": dropout,
            "embed_dim": embed_dim, "lambda_l1": lambda_l1,
            "trial_num": trial.number
        },
        reinit=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Loaders
    train_dataset = FS2KDataset(split="train")
    val_dataset = FS2KDataset(split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Models
    generator = UNetGenerator(base_channels=base_channels, embed_dim=embed_dim, dropout=dropout).to(device)
    discriminator = PatchGANDiscriminator(base_channels=base_channels, embed_dim=embed_dim).to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Optimizers
    opt_G = optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))

    # Losses
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    n_epochs = 15  # 15 epochs for HPO
    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        for photos, sketches, styles in train_loader:
            photos, sketches, styles = photos.to(device), sketches.to(device), styles.to(device)
            
            # --- Train Discriminator ---
            opt_D.zero_grad()
            
            # Real
            real_preds = discriminator(photos, sketches, styles)
            real_targets = torch.ones_like(real_preds)
            d_real_loss = bce_loss(real_preds, real_targets)
            
            # Fake
            fake_sketches = generator(photos, styles)
            fake_preds = discriminator(photos, fake_sketches.detach(), styles)
            fake_targets = torch.zeros_like(fake_preds)
            d_fake_loss = bce_loss(fake_preds, fake_targets)
            
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            opt_D.step()
            
            # --- Train Generator ---
            opt_G.zero_grad()
            
            fake_preds_for_G = discriminator(photos, fake_sketches, styles)
            g_adv_loss = bce_loss(fake_preds_for_G, torch.ones_like(fake_preds_for_G))
            g_recon_loss = l1_loss(fake_sketches, sketches)
            
            g_loss = g_adv_loss + lambda_l1 * g_recon_loss
            g_loss.backward()
            opt_G.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
        # --- Validation ---
        generator.eval()
        val_g_adv_loss_total = 0.0
        val_g_recon_loss_total = 0.0
        with torch.no_grad():
            for photos, sketches, styles in val_loader:
                photos, sketches, styles = photos.to(device), sketches.to(device), styles.to(device)
                fake_sketches = generator(photos, styles)
                
                fake_preds_for_G = discriminator(photos, fake_sketches, styles)
                g_adv_loss = bce_loss(fake_preds_for_G, torch.ones_like(fake_preds_for_G))
                g_recon_loss = l1_loss(fake_sketches, sketches)
                
                val_g_adv_loss_total += g_adv_loss.item()
                val_g_recon_loss_total += g_recon_loss.item()
                
        val_g_adv_loss = val_g_adv_loss_total / len(val_loader)
        val_g_recon_loss = val_g_recon_loss_total / len(val_loader)
        val_g_loss = val_g_adv_loss + lambda_l1 * val_g_recon_loss
        
        wandb.log({
            "epoch": epoch,
            "train_g_loss": epoch_g_loss / len(train_loader),
            "train_d_loss": epoch_d_loss / len(train_loader),
            "val_g_loss": val_g_loss,
            "val_g_adv_loss": val_g_adv_loss,
            "val_g_recon_loss": val_g_recon_loss
        })
        
        # Pruning
        trial.report(val_g_loss, epoch)
        if trial.should_prune():
            run.finish()
            raise optuna.exceptions.TrialPruned()

    run.finish()
    return val_g_loss

if __name__ == "__main__":
    set_seed(42)
    os.makedirs("optuna_studies", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    
    study = optuna.create_study(
        study_name="task4_hpo",
        direction="minimize",
        storage="sqlite:///optuna_studies/task4.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Run a proper HPO schedule for GANs
    study.optimize(objective, n_trials=20)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
