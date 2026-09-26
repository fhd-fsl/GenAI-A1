import os
import sys
# Add the project root to the Python path so 'src' imports work when the script is executed directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import optuna
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

# IMPORTANT: Windows multiprocessing fix for torch.utils.data.DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from src.utils.config import load_config
from src.utils.loss import CombinedL1SSIMLoss
from src.utils.trainer_utils import EarlyStopping
from src.data.pet_dataset import OxfordPetDataset, balanced_corruption_collate_fn
from src.models.task1.model import UniversalAutoencoder

def objective(trial):
    config = load_config()
    
    # 1. Hyperparameters (Dynamically suggested by Optuna)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    bottleneck_dim = trial.suggest_int("bottleneck_dim", 64, 512, step=64)
    base_channels = trial.suggest_categorical("base_channels", [32, 48, 64])
    dropout_rate = trial.suggest_float("dropout", 0.0, 0.5)
    alpha = trial.suggest_float("alpha", 0.5, 1.0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. DataLoaders
    # Using the default data paths. The config dict can override this if needed.
    base_data_dir = config.get("data", {}).get("dir", "./data")
    data_dir = os.path.join(base_data_dir, "oxford-iiit-pet")
    manifests_dir = config.get("data", {}).get("manifests_dir", "./manifests")
    
    # Train uses the balanced collate function for extreme stability
    train_dataset = OxfordPetDataset(data_dir=data_dir, split="train")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=balanced_corruption_collate_fn,
        num_workers=0, # 0 prevents Windows BrokenPipeError
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Val loads deterministic evaluations from the JSON manifest
    val_dataset = OxfordPetDataset(data_dir=data_dir, split="val", manifests_dir=manifests_dir)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 3. Model, Loss, Optimizer
    model = UniversalAutoencoder(
        base_channels=base_channels,
        bottleneck_dim=bottleneck_dim,
        dropout_rate=dropout_rate
    ).to(device)
    
    criterion = CombinedL1SSIMLoss(alpha=alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 4. Weights & Biases Tracking
    # We pass the trial.params directly into config so W&B dashboard sees the exact setup
    wandb.init(
        project="GenAI_A1", 
        name=f"Task1_trial_{trial.number}",
        config=trial.params,
        reinit=True
    )
    
    # 5. Training Loop
    epochs = 30 # Fixed max epochs. We rely entirely on Early Stopping to save time.
    checkpoint_dir = f"checkpoints/task1/trial_{trial.number}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "best_model.pt")
    
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, save_path=save_path)
    
    for epoch in range(epochs):
        # -- Train Phase --
        model.train()
        train_loss = 0.0
        
        for corrupted, clean, _ in train_loader:
            corrupted, clean = corrupted.to(device), clean.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(corrupted)
            
            loss = criterion(reconstructed, clean)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * corrupted.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # -- Validation Phase --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for corrupted, clean, _ in val_loader:
                corrupted, clean = corrupted.to(device), clean.to(device)
                reconstructed = model(corrupted)
                loss = criterion(reconstructed, clean)
                val_loss += loss.item() * corrupted.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # Log to W&B
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
        
        # Optuna Pruning: Stops the trial early if it's performing significantly worse than older trials
        trial.report(val_loss, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()
            
        # Early Stopping: Stops if this specific trial stops improving for 5 epochs
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
    wandb.finish()
    
    return early_stopping.best_loss

if __name__ == "__main__":
    os.makedirs("optuna_studies", exist_ok=True)
    
    # Create SQLite database to persist the Optuna study
    study = optuna.create_study(
        study_name="task1_hpo",
        direction="minimize",
        storage="sqlite:///optuna_studies/task1.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print("==================================================")
    print(f"       Starting Task 1 Optuna Study on {device_name.upper()}...")
    print("==================================================")
    
    # Run 20 trials. This will take a while, but it's fully automated!
    study.optimize(objective, n_trials=20)
    
    print("\n==================================================")
    print("               Study Complete!                    ")
    print("==================================================")
    print(f"Best trial ID: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_trial.value}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
