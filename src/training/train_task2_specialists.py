import os
import sys
import yaml
import torch
import wandb
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Windows multiprocessing fix
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from src.utils.config import load_config
from src.utils.loss import CombinedL1SSIMLoss
from src.utils.trainer_utils import EarlyStopping
from src.data.pet_dataset import OxfordPetDataset
from src.data.corruptions import apply_salt_and_pepper, apply_gaussian_blur, apply_rectangular_occlusion
from src.models.task1.model import UniversalAutoencoder

def get_corruption_func(corruption_name):
    if corruption_name == "salt_and_pepper":
        return lambda t: apply_salt_and_pepper(t, prob=random.uniform(0.02, 0.15))
    elif corruption_name == "gaussian_blur":
        return lambda t: apply_gaussian_blur(t, kernel_size=random.choice([3, 5, 7]), sigma=random.uniform(0.5, 2.5))
    elif corruption_name == "rectangular_occlusion":
        return lambda t: apply_rectangular_occlusion(t, num_rects=random.randint(1, 3), target_area_pct=random.uniform(0.10, 0.35))[0]
    else:
        raise ValueError("Unknown corruption")

def train_specialist(specialist_name, params, device):
    print(f"\n==================================================")
    print(f"       Training {specialist_name.upper()} Specialist")
    print(f"==================================================")
    
    # 1. Instantiate Model & Optimizers using Shared Parameters
    model = UniversalAutoencoder(
        base_channels=params["base_channels"], 
        bottleneck_dim=params["bottleneck_dim"]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = CombinedL1SSIMLoss(alpha=params["alpha"]).to(device)
    
    # 2. Setup DataLoaders
    # We will use the base dataloader and apply ONLY the specific corruption in the loop
    config = load_config()
    data_dir = os.path.join(config.get("data", {}).get("dir", "./data"), "oxford-iiit-pet")
    manifests_dir = config.get("data", {}).get("manifests_dir", "./manifests")
    
    train_dataset = OxfordPetDataset(data_dir=data_dir, split="train")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=params["batch_size"], 
        shuffle=True, 
        num_workers=0
    )
    
    val_dataset = OxfordPetDataset(data_dir=data_dir, split="val", manifests_dir=manifests_dir)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=params["batch_size"], 
        shuffle=False,
        num_workers=0
    )
    
    corruption_fn = get_corruption_func(specialist_name)
    
    # 3. Setup W&B & Checkpointing
    wandb.init(
        project="GenAI_A1", 
        name=f"Task2_Specialist_Final_{specialist_name}",
        config=params,
        reinit=True
    )
    
    checkpoint_dir = f"checkpoints/task2_specialists/{specialist_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "best_model.pt")
    
    epochs = 100
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, save_path=save_path)
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # We manually corrupt the clean tensor on the fly for THIS specific corruption
        for clean_img in train_loader:
            clean_img = clean_img.to(device)
            corrupted_list = []
            for img in clean_img:
                corrupted_list.append(corruption_fn(img))
            
            corrupted_img = torch.stack(corrupted_list).to(device)
            
            optimizer.zero_grad()
            reconstructed = model(corrupted_img)
            loss = criterion(reconstructed, clean_img)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * clean_img.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        c_val = 0
        with torch.no_grad():
            for corrupted, clean, metadata in val_loader:
                corrupted = corrupted.to(device)
                clean = clean.to(device)
                
                # Only evaluate on the specific corruption this specialist handles
                for i in range(corrupted.size(0)):
                    if metadata["corruption"][i] == specialist_name:
                        pred = model(corrupted[i:i+1])
                        val_loss += criterion(pred, clean[i:i+1]).item()
                        c_val += 1
                        
        val_loss /= max(1, c_val)
        
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch
        })
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"[INFO] Early stopping triggered at epoch {epoch}")
            break
            
    wandb.finish()
    print(f"[SUCCESS] {specialist_name.upper()} model saved to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    import optuna
    study_db_path = "sqlite:///optuna_studies/task2_specialists.db"
    if not os.path.exists("optuna_studies/task2_specialists.db"):
        raise FileNotFoundError("Specialist Optuna database not found. Did you run HPO first?")
        
    study = optuna.load_study(study_name="task2_specialists_hpo", storage=study_db_path)
    params = study.best_trial.params
    
    print(f"Loaded shared architecture parameters directly from Optuna DB (Trial {study.best_trial.number}):")
    print(params)
    
    # Train each specialist sequentially to preserve VRAM
    train_specialist("salt_and_pepper", params, device)
    train_specialist("gaussian_blur", params, device)
    train_specialist("rectangular_occlusion", params, device)
    
    print("\n==================================================")
    print("    All 3 Specialists Successfully Trained!")
    print("==================================================")

if __name__ == "__main__":
    main()
