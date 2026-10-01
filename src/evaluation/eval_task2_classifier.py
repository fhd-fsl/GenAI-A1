import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import optuna
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

from src.utils.config import load_config
from src.data.pet_dataset import OxfordPetDataset
from src.models.task2.classifier import CorruptionClassifier

# The correct class mappings matching our training
CORRUPTION_CLASSES = ["clean", "salt_and_pepper", "gaussian_blur", "rectangular_occlusion"]
CORRUPTION_MAP = {k: i for i, k in enumerate(CORRUPTION_CLASSES)}

def main():
    print("==================================================")
    print("       Task 2 Classifier Final Evaluation")
    print("==================================================")
    
    # 1. Load the Best Optuna Trial Configuration
    study_db_path = "sqlite:///optuna_studies/task2_classifier.db"
    if not os.path.exists("optuna_studies/task2_classifier.db"):
        raise FileNotFoundError("Classifier Optuna database not found.")
        
    study = optuna.load_study(study_name="task2_classifier_hpo", storage=study_db_path)
    best_trial = study.best_trial
    print(f"Loading Best Trial: #{best_trial.number}")
    
    # 2. Instantiate Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CorruptionClassifier(
        base_channels=best_trial.params["base_channels"],
        dropout_rate=best_trial.params.get("dropout", 0.0)
    ).to(device)
    
    # 3. Load Trained Weights
    checkpoint_path = f"checkpoints/task2_classifier/trial_{best_trial.number}/best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    # 4. Load Test Dataset
    config = load_config()
    data_dir = os.path.join(config.get("data", {}).get("dir", "./data"), "oxford-iiit-pet")
    manifests_dir = config.get("data", {}).get("manifests_dir", "./manifests")
    
    # Using the deterministic test split for final evaluation
    test_dataset = OxfordPetDataset(data_dir=data_dir, split="test", manifests_dir=manifests_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 5. Inference Loop
    all_preds = []
    all_labels = []
    
    print("\nRunning inference on the test set...")
    with torch.no_grad():
        for corrupted, _, label_dicts in test_loader:
            corrupted = corrupted.to(device)
            
            # Map string labels to integer targets
            labels_str = label_dicts["corruption"]
            labels = [CORRUPTION_MAP[l] for l in labels_str]
            all_labels.extend(labels)
            
            # Forward pass
            outputs = model(corrupted)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            
    # 6. Calculate Metrics (sklearn)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    print("\n--- Overall Metrics ---")
    print(f"Accuracy:       {accuracy * 100:.2f}%")
    print(f"Macro Precision:{precision * 100:.2f}%")
    print(f"Macro Recall:   {recall * 100:.2f}%")
    print(f"Macro F1-Score: {f1 * 100:.2f}%")
    
    # Per-class metrics
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    print("\n--- Per-Class Metrics (F1-Score) ---")
    for i, cls_name in enumerate(CORRUPTION_CLASSES):
        print(f"{cls_name.ljust(25)}: {per_class_f1[i] * 100:.2f}%")
        
    # 7. Generate Normalized Confusion Matrix
    print("\nGenerating Normalized Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CORRUPTION_CLASSES)
    # Using Blues colormap to match standard research paper aesthetics
    disp.plot(cmap='Blues', ax=ax, values_format='.3f') 
    
    plt.title('Normalized Confusion Matrix: Corruption Classifier')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 8. Save Artifacts
    os.makedirs("results/task2", exist_ok=True)
    cm_path = "results/task2/classifier_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Confusion matrix saved to: {cm_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
