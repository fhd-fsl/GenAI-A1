import os
import torch
import optuna
import random
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Add the project root to the python path so imports work when running directly
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import load_config
from src.data.pet_dataset import OxfordPetDataset
from src.models.task1.model import UniversalAutoencoder

def generate_error_map(clean, reconstructed):
    """
    Generates an absolute difference error map.
    Scales it to [0, 1] so it is highly visible to the human eye,
    and returns a 3-channel grayscale image to align with the visual grid.
    """
    diff = torch.abs(clean - reconstructed)
    diff = diff.mean(dim=1, keepdim=True) # Average the RGB channels
    
    # Normalize for visibility (prevent entirely black images if error is small)
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    
    # Convert back to 3 channels so vutils.save_image doesn't complain about shape mismatch
    return diff_norm.repeat(1, 3, 1, 1)

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print(f"       Task 1 Evaluation (Device: {device})")
    print("==================================================")
    
    # 1. Connect to Optuna to fetch the absolute best model
    db_path = "sqlite:///optuna_studies/task1.db"
    study = optuna.load_study(study_name="task1_hpo", storage=db_path)
    best_trial = study.best_trial
    print(f"Loaded Best Trial: #{best_trial.number} (Val Loss: {best_trial.value:.4f})")
    
    # 2. Reconstruct the precise architecture from the DB params
    model = UniversalAutoencoder(
        base_channels=best_trial.params["base_channels"],
        bottleneck_dim=best_trial.params["bottleneck_dim"],
        dropout_rate=best_trial.params.get("dropout", 0.0)
    ).to(device)
    
    checkpoint_path = f"checkpoints/task1/trial_{best_trial.number}/best_model.pt"
    # User might get a weights_only security warning from torch, it is safely ignored.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    alpha = best_trial.params["alpha"]
    
    # 3. Prepare the Test Data
    base_data_dir = config.get("data", {}).get("dir", "./data")
    data_dir = os.path.join(base_data_dir, "oxford-iiit-pet")
    manifests_dir = config.get("data", {}).get("manifests_dir", "./manifests")
    
    test_dataset = OxfordPetDataset(data_dir=data_dir, split="test", manifests_dir=manifests_dir)
    # Batch size 1 is required to accurately extract the string labels for severity/corruption
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) 
    
    # 4. Metrics Dictionaries
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    metrics = {
        "corruption": defaultdict(lambda: {"l1": 0.0, "ssim": 0.0, "count": 0}),
        "severity": defaultdict(lambda: {"l1": 0.0, "ssim": 0.0, "count": 0})
    }
    
    representatives = []
    failures = [] # Will keep the 4 worst
    num_good_candidates_seen = 0

    # 5. Inference Loop
    print("\nRunning Inference on Test Set...")
    with torch.no_grad():
        for corrupted, clean, label_dict in tqdm(test_loader):
            corrupted = corrupted.to(device)
            clean = clean.to(device)
            
            reconstructed = model(corrupted)
            
            l1_loss = F.l1_loss(reconstructed, clean).item()
            ssim_score = ssim_metric(reconstructed, clean).item()
            
            # Calculate the exact loss the model was optimizing for
            combined_loss = (alpha * l1_loss) + ((1.0 - alpha) * (1.0 - ssim_score))
            
            # Extract string labels
            corr_type = label_dict['corruption'][0]
            severity = label_dict['severity'][0]
            
            # Tally metrics
            metrics["corruption"][corr_type]["l1"] += l1_loss
            metrics["corruption"][corr_type]["ssim"] += ssim_score
            metrics["corruption"][corr_type]["count"] += 1
            
            metrics["severity"][severity]["l1"] += l1_loss
            metrics["severity"][severity]["ssim"] += ssim_score
            metrics["severity"][severity]["count"] += 1
            
            record = {
                "clean": clean.cpu().squeeze(0),
                "corrupted": corrupted.cpu().squeeze(0),
                "reconstructed": reconstructed.cpu().squeeze(0),
                "combined_loss": combined_loss
            }
            
            # Use Reservoir Sampling to get exactly 12 uniformly random representatives across the whole dataset
            if combined_loss < 0.15:
                if len(representatives) < 12:
                    representatives.append(record)
                else:
                    j = random.randint(0, num_good_candidates_seen)
                    if j < 12:
                        representatives[j] = record
                num_good_candidates_seen += 1
                
            # Keep track of the worst 4 failures
            failures.append(record)
            failures.sort(key=lambda x: x["combined_loss"], reverse=True)
            failures = failures[:4]

    # 6. Console Report
    print("\n==================================================")
    print("               FINAL EVALUATION RESULTS           ")
    print("==================================================")
    print("\n--- Per Corruption Type ---")
    for k, v in metrics["corruption"].items():
        avg_l1 = v["l1"] / v["count"]
        avg_ssim = v["ssim"] / v["count"]
        print(f"[{k.upper():<9}] L1: {avg_l1:.4f} | SSIM: {avg_ssim:.4f}")
        
    print("\n--- Per Severity Level ---")
    for k, v in metrics["severity"].items():
        if k == "none": continue # "none" severity is identical to "clean" corruption
        avg_l1 = v["l1"] / v["count"]
        avg_ssim = v["ssim"] / v["count"]
        print(f"[{k.upper():<9}] L1: {avg_l1:.4f} | SSIM: {avg_ssim:.4f}")
        
    # 7. Generate Visual Grids
    print("\n==================================================")
    print("Generating Visual Grids (Target | Input | Recon | Error Map)...")
    os.makedirs("results/task1", exist_ok=True)
    
    def save_grid(records, filename):
        grid_rows = []
        for r in records:
            c = r["clean"].unsqueeze(0)
            i = r["corrupted"].unsqueeze(0)
            recon = r["reconstructed"].unsqueeze(0)
            err = generate_error_map(c, recon)
            
            row = torch.cat([c, i, recon, err], dim=0)
            grid_rows.append(row)
            
        full_grid = torch.cat(grid_rows, dim=0)
        vutils.save_image(full_grid, filename, nrow=4, padding=2, normalize=False)

    save_grid(representatives, "results/task1/representative_examples.png")
    save_grid(failures, "results/task1/failure_cases.png")
    
    print("Saved -> results/task1/representative_examples.png")
    print("Saved -> results/task1/failure_cases.png")
    print("==================================================")

if __name__ == "__main__":
    main()
