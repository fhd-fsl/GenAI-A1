import os
import sys
# Add the project root to the Python path so 'src' imports work when the script is executed directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import optuna
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

# IMPORTANT: Windows multiprocessing fix for torch.utils.data.DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from src.utils.config import load_config
from src.utils.trainer_utils import EarlyStopping
from src.data.pet_dataset import OxfordPetDataset, balanced_corruption_collate_fn
from src.models.task2.classifier import CorruptionClassifier

# Mapping string labels to integers for CrossEntropyLoss
CORRUPTION_MAP = {
    "clean": 0,
    "salt_and_pepper": 1,
    "gaussian_blur": 2,
    "rectangular_occlusion": 3
}

def objective(trial):
    config = load_config()
    
    # 1. Hyperparameters (Dynamically suggested by Optuna)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    base_channels = trial.suggest_categorical("base_channels", [16, 32, 64])
    dropout_rate = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. DataLoaders
    base_data_dir = config.get("data", {}).get("dir", "./data")
    data_dir = os.path.join(base_data_dir, "oxford-iiit-pet")
    manifests_dir = config.get("data", {}).get("manifests_dir", "./manifests")
    
    # Train loader
    train_dataset = OxfordPetDataset(data_dir=data_dir, split="train")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=balanced_corruption_collate_fn,
        num_workers=0, # 0 prevents Windows BrokenPipeError
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Val loader
    val_dataset = OxfordPetDataset(data_dir=data_dir, split="val", manifests_dir=manifests_dir)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 3. Model, Loss, Optimizer
    model = CorruptionClassifier(
        base_channels=base_channels,
        dropout_rate=dropout_rate
    ).to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    # Using weight_decay for regularization as requested by assignment
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 4. Weights & Biases Tracking
    wandb.init(
        project="GenAI_A1", 
        name=f"Task2_Classifier_trial_{trial.number}",
        config=trial.params,
        reinit=True
    )
    
    # 5. Training Loop
    epochs = 30 # Fixed max epochs, relying on early stopping
    checkpoint_dir = f"checkpoints/task2_classifier/trial_{trial.number}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "best_model.pt")
    
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, save_path=save_path)
    
    for epoch in range(epochs):
        # -- Train Phase --
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        for corrupted, _, labels in train_loader:
            corrupted = corrupted.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(corrupted)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * corrupted.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_loss /= total_train
        train_acc = train_correct / total_train
        
        # -- Validation Phase --
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for corrupted, _, label_dicts in val_loader:
                corrupted = corrupted.to(device)
                labels = label_dicts["label"].to(device)
                
                outputs = model(corrupted)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * corrupted.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss /= total_val
        val_acc = val_correct / total_val
        
        # Log to W&B
        wandb.log({
            "train_loss": train_loss, 
            "train_acc": train_acc,
            "val_loss": val_loss, 
            "val_acc": val_acc,
            "epoch": epoch
        })
        
        # Optuna Pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()
            
        # Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"[INFO] Early stopping triggered at epoch {epoch}")
            break
            
    wandb.finish()
    
    return early_stopping.best_loss

if __name__ == "__main__":
    os.makedirs("optuna_studies", exist_ok=True)
    
    study = optuna.create_study(
        study_name="task2_classifier_hpo",
        direction="minimize",
        storage="sqlite:///optuna_studies/task2_classifier.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print("==================================================")
    print(f"       Starting Task 2 Classifier Optuna Study")
    print(f"       Hardware: {device_name.upper()}")
    print("==================================================")
    
    study.optimize(objective, n_trials=20)
    
    print("\n==================================================")
    print("               Study Complete!                    ")
    print("==================================================")
    print(f"Best trial ID: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_trial.value}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
