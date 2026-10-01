import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import load_config
from torchmetrics.image import StructuralSimilarityIndexMeasure
from src.data.pet_dataset import OxfordPetDataset
from src.models.task2.classifier import CorruptionClassifier
from src.models.task1.model import UniversalAutoencoder

CORRUPTION_MAP = {
    "clean": 0,
    "salt_and_pepper": 1,
    "gaussian_blur": 2,
    "rectangular_occlusion": 3
}
INV_CORRUPTION_MAP = {v: k for k, v in CORRUPTION_MAP.items()}

def load_models(device):
    print("Loading models...")
    
    # 1. Load Classifier
    import optuna
    study_db_path = "sqlite:///optuna_studies/task2_classifier.db"
    classifier_study = optuna.load_study(study_name="task2_classifier_hpo", storage=study_db_path)
    cls_params = classifier_study.best_trial.params
    
    classifier = CorruptionClassifier(
        base_channels=cls_params["base_channels"],
        dropout_rate=cls_params.get("dropout", 0.0)
    ).to(device)
    
    cls_ckpt = f"checkpoints/task2_classifier/trial_{classifier_study.best_trial.number}/best_model.pt"
    classifier.load_state_dict(torch.load(cls_ckpt, map_location=device, weights_only=True))
    classifier.eval()
    
    # 2. Load Specialists
    spec_study_db_path = "sqlite:///optuna_studies/task2_specialists.db"
    spec_study = optuna.load_study(study_name="task2_specialists_hpo", storage=spec_study_db_path)
    spec_params = spec_study.best_trial.params
    
    def load_expert(name):
        model = UniversalAutoencoder(
            base_channels=spec_params["base_channels"], 
            bottleneck_dim=spec_params["bottleneck_dim"]
        ).to(device)
        ckpt = f"checkpoints/task2_specialists/{name}/best_model.pt"
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        else:
            print(f"[WARNING] Checkpoint not found for {name}. Ensure training finished.")
        model.eval()
        return model
        
    experts = {
        "salt_and_pepper": load_expert("salt_and_pepper"),
        "gaussian_blur": load_expert("gaussian_blur"),
        "rectangular_occlusion": load_expert("rectangular_occlusion"),
        "clean": None # Identity bypass
    }
    
    return classifier, experts

def main():
    print("==================================================")
    print("       Task 2 Hard-Routed Inference Evaluation")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, experts = load_models(device)
    
    config = load_config()
    data_dir = os.path.join(config.get("data", {}).get("dir", "./data"), "oxford-iiit-pet")
    manifests_dir = config.get("data", {}).get("manifests_dir", "./manifests")
    
    test_dataset = OxfordPetDataset(data_dir=data_dir, split="test", manifests_dir=manifests_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    l1_loss = torch.nn.L1Loss(reduction='none')
    
    # Metrics
    oracle_l1_total = 0.0
    oracle_ssim_total = 0.0
    predicted_l1_total = 0.0
    predicted_ssim_total = 0.0
    
    failure_cases = [] # Stores (clean, corrupted, oracle_recon, predicted_recon, true_label, pred_label)
    success_cases = [] # Stores representative examples of good generations
    success_collected = {"salt_and_pepper": False, "gaussian_blur": False, "rectangular_occlusion": False}
    
    print("\nRunning inference on test set (Oracle vs Predicted)...")
    
    with torch.no_grad():
        for corrupted, clean, metadata in tqdm(test_loader):
            corrupted = corrupted.to(device)
            clean = clean.to(device)
            true_labels_str = metadata["corruption"]
            
            # --- 1. Classifier Prediction ---
            cls_outputs = classifier(corrupted)
            _, predicted_idx = torch.max(cls_outputs, 1)
            predicted_labels_str = [INV_CORRUPTION_MAP[idx.item()] for idx in predicted_idx]
            
            # Arrays to hold the reconstructions for this batch
            oracle_recon = torch.zeros_like(corrupted)
            predicted_recon = torch.zeros_like(corrupted)
            
            # --- 2. Hard Routing ---
            for i in range(corrupted.size(0)):
                true_label = true_labels_str[i]
                pred_label = predicted_labels_str[i]
                
                # Oracle Routing
                if true_label == "clean":
                    oracle_recon[i] = corrupted[i] # Identity Bypass
                else:
                    oracle_recon[i] = experts[true_label](corrupted[i:i+1])[0]
                    
                # Predicted Routing
                if pred_label == "clean":
                    predicted_recon[i] = corrupted[i] # Identity Bypass
                else:
                    predicted_recon[i] = experts[pred_label](corrupted[i:i+1])[0]
                    
                # Track classifier error induced restoration failures
                if true_label != pred_label and len(failure_cases) < 4:
                    if true_label != "clean" or pred_label != "clean": # Interesting failures only
                        failure_cases.append({
                            "clean": clean[i].cpu(),
                            "corrupted": corrupted[i].cpu(),
                            "oracle": oracle_recon[i].cpu(),
                            "predicted": predicted_recon[i].cpu(),
                            "true_label": true_label,
                            "pred_label": pred_label
                        })
                # Track diverse representative good examples (one for each corruption)
                elif true_label == pred_label and true_label in success_collected and not success_collected[true_label]:
                    success_collected[true_label] = True
                    success_cases.append({
                            "clean": clean[i].cpu(),
                            "corrupted": corrupted[i].cpu(),
                            "oracle": oracle_recon[i].cpu(),
                            "predicted": predicted_recon[i].cpu(),
                            "true_label": true_label,
                            "pred_label": pred_label
                        })
            
            # --- 3. Compute Batch Metrics ---
            oracle_l1 = l1_loss(oracle_recon, clean).mean(dim=[1,2,3]).sum().item()
            oracle_ssim = ssim_metric(oracle_recon, clean).item() * clean.size(0)
            
            pred_l1 = l1_loss(predicted_recon, clean).mean(dim=[1,2,3]).sum().item()
            pred_ssim = ssim_metric(predicted_recon, clean).item() * clean.size(0)
            
            oracle_l1_total += oracle_l1
            oracle_ssim_total += oracle_ssim
            predicted_l1_total += pred_l1
            predicted_ssim_total += pred_ssim
            
    # --- Final Stats ---
    N = len(test_loader.dataset)
    
    final_oracle_l1 = oracle_l1_total / N
    final_oracle_ssim = oracle_ssim_total / N
    final_pred_l1 = predicted_l1_total / N
    final_pred_ssim = predicted_ssim_total / N
    
    print("\n==================================================")
    print("                 FINAL METRICS                    ")
    print("==================================================")
    print(f"[ORACLE ROUTING]    L1 Loss: {final_oracle_l1:.5f} | SSIM: {final_oracle_ssim:.5f}")
    print(f"[PREDICTED ROUTING] L1 Loss: {final_pred_l1:.5f} | SSIM: {final_pred_ssim:.5f}")
    
    # --- Visual Artifacts ---
    if len(failure_cases) > 0:
        fig, axes = plt.subplots(len(failure_cases), 4, figsize=(16, 4 * len(failure_cases)))
        
        for idx, case in enumerate(failure_cases):
            # Move channels back to HWC for matplotlib
            cln_img = case["clean"].permute(1, 2, 0).numpy().clip(0, 1)
            cor_img = case["corrupted"].permute(1, 2, 0).numpy().clip(0, 1)
            orcl_img = case["oracle"].permute(1, 2, 0).numpy().clip(0, 1)
            prd_img = case["predicted"].permute(1, 2, 0).numpy().clip(0, 1)
            
            if len(failure_cases) == 1:
                ax_row = axes
            else:
                ax_row = axes[idx]
                
            ax_row[0].imshow(cln_img)
            ax_row[0].set_title(f"Target (Clean)")
            ax_row[0].axis('off')
            
            ax_row[1].imshow(cor_img)
            ax_row[1].set_title(f"Corrupted Input\n(True: {case['true_label']})")
            ax_row[1].axis('off')
            
            ax_row[2].imshow(orcl_img)
            ax_row[2].set_title(f"Oracle Reconstruction\n(Routed to: {case['true_label']})")
            ax_row[2].axis('off')
            
            ax_row[3].imshow(prd_img)
            ax_row[3].set_title(f"Predicted Reconstruction\n(Misrouted to: {case['pred_label']})")
            ax_row[3].axis('off')
            
        plt.tight_layout()
        os.makedirs("results/task2", exist_ok=True)
        plt.savefig("results/task2/routing_failures.png", dpi=300)
        plt.close()
        print("\n[SUCCESS] Failure cases grid saved to results/task2/routing_failures.png")
        
    if len(success_cases) > 0:
        fig, axes = plt.subplots(len(success_cases), 4, figsize=(16, 4 * len(success_cases)))
        
        for idx, case in enumerate(success_cases):
            # Move channels back to HWC for matplotlib
            cln_img = case["clean"].permute(1, 2, 0).numpy().clip(0, 1)
            cor_img = case["corrupted"].permute(1, 2, 0).numpy().clip(0, 1)
            orcl_img = case["oracle"].permute(1, 2, 0).numpy().clip(0, 1)
            prd_img = case["predicted"].permute(1, 2, 0).numpy().clip(0, 1)
            
            if len(success_cases) == 1:
                ax_row = axes
            else:
                ax_row = axes[idx]
                
            ax_row[0].imshow(cln_img)
            ax_row[0].set_title(f"Target (Clean)")
            ax_row[0].axis('off')
            
            ax_row[1].imshow(cor_img)
            ax_row[1].set_title(f"Corrupted Input\n(True: {case['true_label']})")
            ax_row[1].axis('off')
            
            ax_row[2].imshow(orcl_img)
            ax_row[2].set_title(f"Oracle Reconstruction\n(Routed to: {case['true_label']})")
            ax_row[2].axis('off')
            
            ax_row[3].imshow(prd_img)
            ax_row[3].set_title(f"Predicted Reconstruction\n(Properly Routed to: {case['pred_label']})")
            ax_row[3].axis('off')
            
        plt.tight_layout()
        plt.savefig("results/task2/representative_examples.png", dpi=300)
        plt.close()
        print("[SUCCESS] Representative successes grid saved to results/task2/representative_examples.png")

if __name__ == "__main__":
    main()
