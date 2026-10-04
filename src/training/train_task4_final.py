import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import numpy as np
import optuna

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

def train():
    set_seed(42)
    os.makedirs("checkpoints/task4", exist_ok=True)
    
    # Load config directly from Optuna study
    db_path = "sqlite:///optuna_studies/task4.db"
    study = optuna.load_study(study_name="task4_hpo", storage=db_path)
    best_trial = study.best_trial
    config = best_trial.params
    print(f"Loaded Best Trial: #{best_trial.number} (Val Loss: {best_trial.value:.4f})")
        
    # Init wandb
    wandb.init(
        project="GenAI_A1",
        name="task4_final_training",
        config=config,
        reinit=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    train_dataset = FS2KDataset(split="train")
    val_dataset = FS2KDataset(split="val")
    
    batch_size = config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Static batch for visualization
    val_iter = iter(val_loader)
    static_val_photos, static_val_sketches, static_val_styles = next(val_iter)
    static_val_photos = static_val_photos.to(device)
    static_val_sketches = static_val_sketches.to(device)
    static_val_styles = static_val_styles.to(device)
    
    # Models
    generator = UNetGenerator(
        base_channels=config["base_channels"], 
        embed_dim=config["embed_dim"],
        dropout=config.get("dropout", 0.5)
    ).to(device)
    discriminator = PatchGANDiscriminator(
        base_channels=config["base_channels"], 
        embed_dim=config["embed_dim"]
    ).to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Optimizers
    opt_G = optim.Adam(generator.parameters(), lr=config["g_lr"], betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=config["d_lr"], betas=(0.5, 0.999))
    
    # LR Schedulers (Linear decay for second half of training)
    n_epochs = 100
    def lambda_rule(epoch):
        return 1.0 - max(0, epoch - n_epochs//2) / float(n_epochs//2 + 1)
    
    scheduler_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lambda_rule)
    scheduler_D = optim.lr_scheduler.LambdaLR(opt_D, lr_lambda=lambda_rule)
    
    # Losses
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    lambda_l1 = config["lambda_l1"]
    
    best_val_loss = float("inf")
    
    for epoch in range(n_epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_adv_loss = 0.0
        epoch_g_recon_loss = 0.0
        epoch_d_real_loss = 0.0
        epoch_d_fake_loss = 0.0
        
        for photos, sketches, styles in train_loader:
            photos, sketches, styles = photos.to(device), sketches.to(device), styles.to(device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            opt_D.zero_grad()
            
            # Real
            real_preds = discriminator(photos, sketches, styles)
            d_real_loss = bce_loss(real_preds, torch.ones_like(real_preds))
            
            # Fake
            fake_sketches = generator(photos, styles)
            fake_preds = discriminator(photos, fake_sketches.detach(), styles)
            d_fake_loss = bce_loss(fake_preds, torch.zeros_like(fake_preds))
            
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            opt_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            opt_G.zero_grad()
            
            fake_preds_for_G = discriminator(photos, fake_sketches, styles)
            g_adv_loss = bce_loss(fake_preds_for_G, torch.ones_like(fake_preds_for_G))
            g_recon_loss = l1_loss(fake_sketches, sketches)
            
            g_loss = g_adv_loss + lambda_l1 * g_recon_loss
            g_loss.backward()
            opt_G.step()
            
            epoch_g_adv_loss += g_adv_loss.item()
            epoch_g_recon_loss += g_recon_loss.item()
            epoch_d_real_loss += d_real_loss.item()
            epoch_d_fake_loss += d_fake_loss.item()
            
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
        
        # Schedulers step
        scheduler_G.step()
        scheduler_D.step()
        
        # Logging
        log_dict = {
            "epoch": epoch,
            "train_d_real_loss": epoch_d_real_loss / len(train_loader),
            "train_d_fake_loss": epoch_d_fake_loss / len(train_loader),
            "train_g_adv_loss": epoch_g_adv_loss / len(train_loader),
            "train_g_recon_loss": epoch_g_recon_loss / len(train_loader),
            "val_g_loss": val_g_loss,
            "val_g_adv_loss": val_g_adv_loss,
            "val_g_recon_loss": val_g_recon_loss,
            "lr_g": opt_G.param_groups[0]['lr'],
            "lr_d": opt_D.param_groups[0]['lr']
        }
        
        # Periodic sample logging
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                generated_samples = generator(static_val_photos, static_val_styles)
            
            # Denormalize from [-1, 1] to [0, 1] for proper visualization
            denorm = lambda x: (x + 1.0) / 2.0
            
            images = []
            for i in range(min(4, static_val_photos.size(0))):
                images.append(wandb.Image(denorm(static_val_photos[i]), caption=f"Photo (Style {static_val_styles[i].item()})"))
                images.append(wandb.Image(denorm(generated_samples[i]), caption="Generated"))
                images.append(wandb.Image(denorm(static_val_sketches[i]), caption="Target"))
            
            log_dict["Validation Samples"] = images
            
        wandb.log(log_dict)
        
        # Checkpointing
        if val_g_loss < best_val_loss:
            best_val_loss = val_g_loss
            torch.save(generator.state_dict(), "checkpoints/task4/generator_best.pth")
            torch.save(discriminator.state_dict(), "checkpoints/task4/discriminator_best.pth")
            
    # Save final models
    torch.save(generator.state_dict(), "checkpoints/task4/generator_final.pth")
    torch.save(discriminator.state_dict(), "checkpoints/task4/discriminator_final.pth")
    
    wandb.finish()
    print("Training complete. Best and final models saved to checkpoints/task4/.")

if __name__ == "__main__":
    train()
