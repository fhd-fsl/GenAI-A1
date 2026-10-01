import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import optuna
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from src.utils.config import load_config
from src.utils.loss import CombinedL1SSIMLoss
from src.data.pet_dataset import OxfordPetDataset
from src.data.corruptions import apply_salt_and_pepper, apply_gaussian_blur, apply_rectangular_occlusion
from src.models.task1.model import UniversalAutoencoder

def shared_specialists_collate_fn(batch):
    """
    Takes a batch of purely clean images (from the train split) and synthesizes
    three separate corrupted batches so we can train 3 specialists simultaneously.
    """
    clean_images = torch.stack(batch, dim=0)
    sp_list, blur_list, occ_list = [], [], []
    
    for clean_tensor in clean_images:
        # 1. Salt & Pepper
        p = random.uniform(0.02, 0.15)
        sp_list.append(apply_salt_and_pepper(clean_tensor, prob=p))
        
        # 2. Gaussian Blur
        kernel = random.choice([3, 5, 7])
        sigma = random.uniform(0.5, 2.5)
        blur_list.append(apply_gaussian_blur(clean_tensor, kernel_size=kernel, sigma=sigma))
        
        # 3. Rectangular Occlusion
        num_rects = random.randint(1, 3)
        area_pct = random.uniform(0.10, 0.35)
        occ, _ = apply_rectangular_occlusion(clean_tensor, num_rects=num_rects, target_area_pct=area_pct)
        occ_list.append(occ)
        
    return torch.stack(sp_list), torch.stack(blur_list), torch.stack(occ_list), clean_images

def objective(trial):
    config = load_config()
    
    # 1. Shared Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    bottleneck_dim = trial.suggest_int("bottleneck_dim", 64, 512, step=64)
    base_channels = trial.suggest_categorical("base_channels", [32, 48, 64])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    alpha = trial.suggest_float("alpha", 0.5, 1.0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. DataLoaders
    base_data_dir = config.get("data", {}).get("dir", "./data")
    data_dir = os.path.join(base_data_dir, "oxford-iiit-pet")
    manifests_dir = config.get("data", {}).get("manifests_dir", "./manifests")
    
    train_dataset = OxfordPetDataset(data_dir=data_dir, split="train")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=shared_specialists_collate_fn,
        num_workers=0
    )
    
    val_dataset = OxfordPetDataset(data_dir=data_dir, split="val", manifests_dir=manifests_dir)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # 3. Instantiate 3 Independent Specialists using the Shared Architecture
    model_sp = UniversalAutoencoder(base_channels=base_channels, bottleneck_dim=bottleneck_dim).to(device)
    model_blur = UniversalAutoencoder(base_channels=base_channels, bottleneck_dim=bottleneck_dim).to(device)
    model_occ = UniversalAutoencoder(base_channels=base_channels, bottleneck_dim=bottleneck_dim).to(device)
    
    opt_sp = torch.optim.Adam(model_sp.parameters(), lr=lr)
    opt_blur = torch.optim.Adam(model_blur.parameters(), lr=lr)
    opt_occ = torch.optim.Adam(model_occ.parameters(), lr=lr)
    
    criterion = CombinedL1SSIMLoss(alpha=alpha).to(device)
    
    wandb.init(
        project="GenAI_A1", 
        name=f"Task2_SpecialistHPO_trial_{trial.number}",
        config=trial.params,
        reinit=True
    )
    
    epochs = 15 # Cap epochs for HPO compute feasibility
    best_avg_val_loss = float('inf')
    patience = 4
    stagnant_epochs = 0
    
    for epoch in range(epochs):
        # -- Train Phase --
        model_sp.train()
        model_blur.train()
        model_occ.train()
        
        train_loss_sp = 0.0
        train_loss_blur = 0.0
        train_loss_occ = 0.0
        
        for sp_img, blur_img, occ_img, clean_img in train_loader:
            sp_img, blur_img, occ_img, clean_img = sp_img.to(device), blur_img.to(device), occ_img.to(device), clean_img.to(device)
            
            # S&P Step
            opt_sp.zero_grad()
            loss_sp = criterion(model_sp(sp_img), clean_img)
            loss_sp.backward()
            opt_sp.step()
            train_loss_sp += loss_sp.item() * clean_img.size(0)
            
            # Blur Step
            opt_blur.zero_grad()
            loss_blur = criterion(model_blur(blur_img), clean_img)
            loss_blur.backward()
            opt_blur.step()
            train_loss_blur += loss_blur.item() * clean_img.size(0)
            
            # Occlusion Step
            opt_occ.zero_grad()
            loss_occ = criterion(model_occ(occ_img), clean_img)
            loss_occ.backward()
            opt_occ.step()
            train_loss_occ += loss_occ.item() * clean_img.size(0)
            
        N_train = len(train_loader.dataset)
        train_loss_sp /= N_train
        train_loss_blur /= N_train
        train_loss_occ /= N_train
        avg_train_loss = (train_loss_sp + train_loss_blur + train_loss_occ) / 3.0
        
        # -- Validation Phase --
        model_sp.eval()
        model_blur.eval()
        model_occ.eval()
        
        val_loss_sp = 0.0
        val_loss_blur = 0.0
        val_loss_occ = 0.0
        c_sp = c_blur = c_occ = 0
        
        with torch.no_grad():
            for corrupted, clean, metadata in val_loader:
                corrupted = corrupted.to(device)
                clean = clean.to(device)
                corruptions = metadata["corruption"]
                
                for i in range(corrupted.size(0)):
                    c_type = corruptions[i]
                    if c_type == "salt_and_pepper":
                        val_loss_sp += criterion(model_sp(corrupted[i:i+1]), clean[i:i+1]).item()
                        c_sp += 1
                    elif c_type == "gaussian_blur":
                        val_loss_blur += criterion(model_blur(corrupted[i:i+1]), clean[i:i+1]).item()
                        c_blur += 1
                    elif c_type == "rectangular_occlusion":
                        val_loss_occ += criterion(model_occ(corrupted[i:i+1]), clean[i:i+1]).item()
                        c_occ += 1
                        
        val_loss_sp /= max(1, c_sp)
        val_loss_blur /= max(1, c_blur)
        val_loss_occ /= max(1, c_occ)
        
        avg_val_loss = (val_loss_sp + val_loss_blur + val_loss_occ) / 3.0
        
        wandb.log({
            "train_loss_avg": avg_train_loss,
            "val_loss_avg": avg_val_loss,
            "val_loss_sp": val_loss_sp,
            "val_loss_blur": val_loss_blur,
            "val_loss_occ": val_loss_occ,
            "epoch": epoch
        })
        
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()
            
        if avg_val_loss < best_avg_val_loss - 0.001:
            best_avg_val_loss = avg_val_loss
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1
            
        if stagnant_epochs >= patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break
            
    wandb.finish()
    return best_avg_val_loss

if __name__ == "__main__":
    os.makedirs("optuna_studies", exist_ok=True)
    
    study = optuna.create_study(
        study_name="task2_specialists_hpo",
        direction="minimize",
        storage="sqlite:///optuna_studies/task2_specialists.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    )
    
    print("==================================================")
    print("       Starting Shared Specialist Optuna Study")
    print("==================================================")
    
    # 20 trials x 3 models is heavy, but we cap epochs at 15 and use strict pruning
    study.optimize(objective, n_trials=20)
    
    print("\n==================================================")
    print("               Study Complete!                    ")
    print("==================================================")
    print(f"Best trial ID: {study.best_trial.number}")
    print(f"Best average validation loss: {study.best_trial.value}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
        
    import yaml
    os.makedirs("configs", exist_ok=True)
    with open("configs/task2_specialists_best.yaml", "w") as f:
        yaml.dump(study.best_trial.params, f)
        
    print("[SUCCESS] Optimal shared parameters permanently saved to configs/task2_specialists_best.yaml")
