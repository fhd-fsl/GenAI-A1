import os
import torch
import random
import yaml
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
import optuna
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.fs2k_dataset import FS2KDataset
from src.models.task4.generator import UNetGenerator

def generate_error_map(clean, reconstructed):
    """
    Generates an absolute difference error map.
    Scales it to [0, 1] so it is highly visible to the human eye,
    and returns a 3-channel grayscale image to align with the visual grid.
    """
    diff = torch.abs(clean - reconstructed)
    diff = diff.mean(dim=1, keepdim=True) # Average the channels
    
    # Normalize for visibility (prevent entirely black images if error is small)
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    
    # Convert back to 3 channels so vutils.save_image doesn't complain about shape mismatch
    return diff_norm.repeat(1, 3, 1, 1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print(f"       Task 4 Evaluation (Device: {device})")
    print("==================================================")
    
    # Load config from the Optuna run directly
    db_path = "sqlite:///optuna_studies/task4.db"
    study = optuna.load_study(study_name="task4_hpo", storage=db_path)
    config = study.best_trial.params
    print(f"Loaded Best Trial: #{study.best_trial.number}")
        
    # Reconstruct the architecture
    generator = UNetGenerator(
        base_channels=config["base_channels"],
        embed_dim=config["embed_dim"],
        dropout=config.get("dropout", 0.5)
    ).to(device)
    
    checkpoint_path = "checkpoints/task4/generator_final.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return
        
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    generator.eval()
    
    # Prepare the Test Data
    test_dataset = FS2KDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) 
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    
    metrics = {
        "style": defaultdict(lambda: {"l1": 0.0, "ssim": 0.0, "psnr": 0.0, "count": 0})
    }
    
    representatives = []
    failures = []
    num_good_candidates_seen = 0

    print("\nRunning Inference on Test Set...")
    with torch.no_grad():
        for photos, sketches, styles in tqdm(test_loader):
            photos = photos.to(device)
            sketches = sketches.to(device)
            styles = styles.to(device)
            
            fake_sketches = generator(photos, styles)
            
            # Denormalize from [-1, 1] to [0, 1]
            denorm = lambda x: (x + 1.0) / 2.0
            photos_norm = denorm(photos)
            sketches_norm = denorm(sketches)
            fake_sketches_norm = denorm(fake_sketches)
            
            l1_loss = F.l1_loss(fake_sketches_norm, sketches_norm).item()
            ssim_score = ssim_metric(fake_sketches_norm, sketches_norm).item()
            psnr_score = psnr_metric(fake_sketches_norm, sketches_norm).item()
            style_val = styles.item()
            
            # Tally metrics
            metrics["style"][style_val]["l1"] += l1_loss
            metrics["style"][style_val]["ssim"] += ssim_score
            metrics["style"][style_val]["psnr"] += psnr_score
            metrics["style"][style_val]["count"] += 1
            
            record = {
                "photo": photos_norm.cpu().squeeze(0),
                "sketch": sketches_norm.cpu().squeeze(0),
                "fake": fake_sketches_norm.cpu().squeeze(0),
                "l1_loss": l1_loss,
                "style": style_val
            }
            
            # Use Reservoir Sampling to get exactly 12 uniformly random representatives across the whole dataset
            if l1_loss < 0.2:
                if len(representatives) < 12:
                    representatives.append(record)
                else:
                    j = random.randint(0, num_good_candidates_seen)
                    if j < 12:
                        representatives[j] = record
                num_good_candidates_seen += 1
                
            # Keep track of the worst 4 failures
            failures.append(record)
            failures.sort(key=lambda x: x["l1_loss"], reverse=True)
            failures = failures[:4]

    # Console Report
    print("\n==================================================")
    print("               FINAL EVALUATION RESULTS           ")
    print("==================================================")
    print("\n--- Per Style ---")
    for k in sorted(metrics["style"].keys()):
        v = metrics["style"][k]
        if v["count"] > 0:
            avg_l1 = v["l1"] / v["count"]
            avg_ssim = v["ssim"] / v["count"]
            avg_psnr = v["psnr"] / v["count"]
            print(f"[Style {k}] L1: {avg_l1:.4f} | SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.4f} (Count: {v['count']})")
        
    # Generate Visual Grids
    print("\n==================================================")
    print("Generating Visual Grids (Photo | Target Sketch | Gen Sketch | Error Map)...")
    os.makedirs("results/task4", exist_ok=True)
    
    def save_grid(records, filename):
        if not records:
            return
        grid_rows = []
        for r in records:
            p = r["photo"].unsqueeze(0)
            s = r["sketch"].unsqueeze(0)
            f = r["fake"].unsqueeze(0)
            err = generate_error_map(s, f)
            
            row = torch.cat([p, s, f, err], dim=0)
            grid_rows.append(row)
            
        full_grid = torch.cat(grid_rows, dim=0)
        vutils.save_image(full_grid, filename, nrow=4, padding=2, normalize=False)

    save_grid(representatives, "results/task4/representative_examples.png")
    save_grid(failures, "results/task4/failure_cases.png")
    
    print("Saved -> results/task4/representative_examples.png")
    print("Saved -> results/task4/failure_cases.png")
    print("==================================================")

if __name__ == "__main__":
    main()
