import os
import sys
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import load_config
from src.data.pet_dataset import OxfordPetDataset
from src.models.task3.soft_moe import SoftMoE

CORRUPTION_MAP = {
    "clean": 0,
    "salt_and_pepper": 1,
    "gaussian_blur": 2,
    "rectangular_occlusion": 3
}
CORRUPTION_NAMES = ["Clean", "Salt & Pepper", "Gaussian Blur", "Rect. Occlusion"]
SEVERITY_LEVELS = ["low", "medium", "high"]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load trained model
    import optuna
    study = optuna.load_study(
        study_name="task3_moe_hpo",
        storage="sqlite:///optuna_studies/task3_moe.db"
    )
    params = study.best_trial.params

    model, _, _ = SoftMoE.from_pretrained(device=device)
    model.tau.fill_(params["temperature"])

    ckpt = "checkpoints/task3_moe/best_model.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    else:
        print("[WARNING] No fine-tuned checkpoint found. Using pre-trained weights only.")

    model.eval()

    # 2. DataLoader
    config = load_config()
    data_dir = os.path.join(config.get("data", {}).get("dir", "./data"), "oxford-iiit-pet")
    manifests_dir = config.get("data", {}).get("manifests_dir", "./manifests")

    test_dataset = OxfordPetDataset(data_dir=data_dir, split="test", manifests_dir=manifests_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    l1_loss_fn = torch.nn.L1Loss(reduction='none')

    # 3. Evaluation Setup
    # Global metrics
    total_l1 = 0.0
    total_ssim = 0.0
    total_count = 0
    total_inference_time = 0.0

    # Per-corruption metrics
    # Dict mapping c_idx -> {"l1": sum, "ssim": sum, "count": int}
    metrics_by_corruption = {i: {"l1": 0.0, "ssim": 0.0, "count": 0} for i in range(4)}

    # Weights by (c_idx, severity)
    # dict mapping (c_idx, severity) -> list of [4] weight vectors
    weights_by_condition = {(i, s): [] for i in range(4) for s in SEVERITY_LEVELS}

    dominant_examples = {}
    distributed_examples = {}

    print("==================================================")
    print("   Task 3 Soft MoE Evaluation")
    print("==================================================")

    with torch.no_grad():
        for corrupted, clean, metadata in tqdm(test_loader, desc="Evaluating"):
            corrupted, clean = corrupted.to(device), clean.to(device)
            corruption_labels = metadata["corruption"]
            severity_labels = metadata.get("severity", ["low"] * len(corruption_labels))

            # Measure Inference Time
            start_time = time.perf_counter()
            x_hat, weights, logits = model(corrupted)
            end_time = time.perf_counter()
            total_inference_time += (end_time - start_time)

            # L1
            batch_l1 = l1_loss_fn(x_hat, clean).mean(dim=[1, 2, 3])
            total_l1 += batch_l1.sum().item()

            # SSIM
            # We must compute SSIM per sample for accurate per-corruption tracking
            for i in range(corrupted.size(0)):
                c_type = corruption_labels[i]
                c_idx = CORRUPTION_MAP[c_type]
                sev = severity_labels[i]

                # If clean doesn't have proper severity, default to 'low'
                if sev not in SEVERITY_LEVELS:
                    sev = "low"

                l1_val = batch_l1[i].item()
                ssim_val = ssim_metric(x_hat[i:i+1], clean[i:i+1]).item()

                total_ssim += ssim_val
                total_count += 1

                metrics_by_corruption[c_idx]["l1"] += l1_val
                metrics_by_corruption[c_idx]["ssim"] += ssim_val
                metrics_by_corruption[c_idx]["count"] += 1

                w = weights[i].cpu().numpy()
                weights_by_condition[(c_idx, sev)].append(w)

                max_w = w.max()
                if max_w > 0.80 and c_type not in dominant_examples:
                    dominant_examples[c_type] = {
                        "clean": clean[i].cpu(), "corrupted": corrupted[i].cpu(),
                        "reconstructed": x_hat[i].cpu(), "weights": w,
                        "corruption": c_type, "severity": sev
                    }
                elif max_w < 0.50 and c_type not in distributed_examples:
                    distributed_examples[c_type] = {
                        "clean": clean[i].cpu(), "corrupted": corrupted[i].cpu(),
                        "reconstructed": x_hat[i].cpu(), "weights": w,
                        "corruption": c_type, "severity": sev
                    }

    # 4. Final metrics
    avg_l1 = total_l1 / total_count
    avg_ssim = total_ssim / total_count
    avg_inf_time_ms = (total_inference_time / total_count) * 1000

    print(f"\n[RESULTS] Global L1 Loss: {avg_l1:.5f} | Global SSIM: {avg_ssim:.5f}")
    print(f"          Avg Inference Time: {avg_inf_time_ms:.2f} ms / image")
    print(f"          Test samples: {total_count}")

    print("\n--- Per-Corruption Metrics ---")
    for c_idx in range(4):
        c_name = CORRUPTION_NAMES[c_idx]
        stats = metrics_by_corruption[c_idx]
        if stats["count"] > 0:
            c_l1 = stats["l1"] / stats["count"]
            c_ssim = stats["ssim"] / stats["count"]
            print(f"{c_name:18s} | L1: {c_l1:.5f} | SSIM: {c_ssim:.5f}")

    # 5. Routing heatmap (Type & Severity vs Expert weight)
    os.makedirs("results/task3", exist_ok=True)

    heatmap_rows = []
    row_labels = []

    for c_idx in range(4):
        c_name = CORRUPTION_NAMES[c_idx]
        if c_idx == 0:  # Clean only has one row
            w_list = weights_by_condition[(c_idx, "low")]
            if w_list:
                heatmap_rows.append(np.stack(w_list).mean(axis=0))
                row_labels.append(c_name)
        else:
            for sev in SEVERITY_LEVELS:
                w_list = weights_by_condition[(c_idx, sev)]
                if w_list:
                    heatmap_rows.append(np.stack(w_list).mean(axis=0))
                    row_labels.append(f"{c_name} ({sev})")

    heatmap_data = np.stack(heatmap_rows) if heatmap_rows else np.zeros((1, 4))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=["Identity", "S&P Expert", "Blur Expert", "Occ Expert"],
        yticklabels=row_labels, ax=ax
    )
    ax.set_xlabel("Expert Branch")
    ax.set_ylabel("True Corruption (Severity)")
    ax.set_title("Average Routing Weights by Corruption and Severity")
    plt.tight_layout()
    plt.savefig("results/task3/routing_heatmap.png", dpi=300)
    plt.close()
    print("[SUCCESS] Routing heatmap saved to results/task3/routing_heatmap.png")

    # 6. Representative examples grid
    def plot_examples(examples, title, filename):
        if not examples:
            return
        n = len(examples)
        fig, axes = plt.subplots(n, 4, figsize=(18, 4.5 * n))
        if n == 1:
            axes = [axes]

        for idx, ex in enumerate(examples):
            cln = ex["clean"].permute(1, 2, 0).numpy().clip(0, 1)
            cor = ex["corrupted"].permute(1, 2, 0).numpy().clip(0, 1)
            rec = ex["reconstructed"].permute(1, 2, 0).numpy().clip(0, 1)
            w = ex["weights"]

            axes[idx][0].imshow(cln)
            axes[idx][0].set_title("Target (Clean)")
            axes[idx][0].axis('off')

            axes[idx][1].imshow(cor)
            axes[idx][1].set_title(f"Corrupted\n({ex['corruption']}, {ex['severity']})")
            axes[idx][1].axis('off')

            axes[idx][2].imshow(rec)
            axes[idx][2].set_title("MoE Reconstruction")
            axes[idx][2].axis('off')

            colors = ['#4CAF50', '#FF9800', '#2196F3', '#9C27B0']
            bars = axes[idx][3].bar(
                ["Identity", "S&P", "Blur", "Occ"], w, color=colors
            )
            axes[idx][3].set_ylim(0, 1)
            axes[idx][3].set_title("Routing Weights")
            for bar, val in zip(bars, w):
                axes[idx][3].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                                  f'{val:.2f}', ha='center', fontsize=9)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"results/task3/{filename}", dpi=300)
        plt.close()

    plot_examples(list(dominant_examples.values())[:3], "Single Expert Dominance Examples", "dominant_examples.png")
    plot_examples(list(distributed_examples.values())[:3], "Distributed Weight Examples", "distributed_examples.png")

    # 7. Save evaluation report
    with open("results/task3/evaluation_report.md", "w") as f:
        f.write("# Task 3: Soft Mixture-of-Experts Evaluation\n\n")
        f.write(f"## Test Set Global Metrics (N={total_count})\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| L1 Loss | {avg_l1:.5f} |\n")
        f.write(f"| SSIM | {avg_ssim:.5f} |\n")
        f.write(f"| Avg Inference Time | {avg_inf_time_ms:.2f} ms/image |\n\n")

        f.write("## Per-Corruption Metrics\n")
        f.write("| Corruption | Count | L1 Loss | SSIM |\n")
        f.write("|------------|-------|---------|------|\n")
        for c_idx in range(4):
            c_name = CORRUPTION_NAMES[c_idx]
            stats = metrics_by_corruption[c_idx]
            if stats["count"] > 0:
                c_l1 = stats["l1"] / stats["count"]
                c_ssim = stats["ssim"] / stats["count"]
                f.write(f"| {c_name} | {stats['count']} | {c_l1:.5f} | {c_ssim:.5f} |\n")
        
        f.write("\n## Routing Weight Analysis (Type & Severity)\n")
        f.write("| Condition | Identity | S&P Expert | Blur Expert | Occ Expert |\n")
        f.write("|-----------|----------|------------|-------------|------------|\n")
        for i, row in enumerate(heatmap_rows):
            f.write(f"| {row_labels[i]} | {row[0]:.4f} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f} |\n")

        # Explicit Expert Activity & Dominance Check
        f.write("\n## Expert Activity & Dominance Check\n")
        overall_avg = heatmap_data.mean(axis=0)
        
        # 1. Inactivity Check
        f.write("### 1. Inactivity Check\n")
        all_active = True
        expert_names = ["Identity", "S&P Expert", "Blur Expert", "Occ Expert"]
        for idx, name in enumerate(expert_names):
            if overall_avg[idx] < 0.05:
                f.write(f"- [FAIL] **{name}** is INACTIVE (Overall Avg Weight: {overall_avg[idx]:.4f})\n")
                all_active = False
        if all_active:
            f.write("- [PASS] All experts are active and receiving >5% of the overall routing volume.\n")
            
        # 2. Unrelated Dominance Check
        f.write("\n### 2. Unrelated Dominance Check\n")
        dominance_found = False
        for i, row in enumerate(heatmap_rows):
            cond_name = row_labels[i]
            target_idx = 0 if "Clean" in cond_name else (1 if "Salt & Pepper" in cond_name else (2 if "Blur" in cond_name else 3))
            
            for ext_idx in range(4):
                if ext_idx != target_idx and row[ext_idx] > 0.50:
                    f.write(f"- [WARNING] **{expert_names[ext_idx]}** dominates **{cond_name}** images (Weight: {row[ext_idx]:.4f})\n")
                    dominance_found = True
                    
        if not dominance_found:
            f.write("- [PASS] No expert dominates unrelated inputs (max unrelated weight threshold: 0.50).\n")

    print("[SUCCESS] Evaluation report saved to results/task3/evaluation_report.md")


if __name__ == "__main__":
    main()
