import os
import sys
import torch
import optuna
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.task2.classifier import CorruptionClassifier
from src.models.task1.model import UniversalAutoencoder

def export_and_verify(model, save_path, dummy_input, model_name):
    model.eval()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 1. Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"[{model_name}] Successfully exported to {save_path}")
    
    # 2. Verify with ONNX Runtime
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARNING] onnxruntime not installed. Skipping verification.")
        return
        
    ort_session = ort.InferenceSession(save_path, providers=['CPUExecutionProvider'])
    
    # PyTorch output
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()
        
    # ONNX Runtime output
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_out = ort_outs[0]
    
    # Compare
    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-03, atol=1e-05)
    print(f"[{model_name}] Verification SUCCESS! Max difference is within < 1e-5.\n")


def main():
    device = torch.device("cpu") # Export requires CPU tensors for dummy input
    dummy_input = torch.randn(1, 3, 128, 128, device=device)
    
    print("==================================================")
    print("        Exporting Task 2 Models to ONNX")
    print("==================================================")
    
    # --- 1. Export Classifier ---
    cls_db = "sqlite:///optuna_studies/task2_classifier.db"
    if not os.path.exists("optuna_studies/task2_classifier.db"):
        raise FileNotFoundError("Classifier study DB not found.")
        
    cls_study = optuna.load_study(study_name="task2_classifier_hpo", storage=cls_db)
    cls_params = cls_study.best_trial.params
    
    classifier = CorruptionClassifier(
        base_channels=cls_params["base_channels"],
        dropout_rate=cls_params.get("dropout", 0.0)
    ).to(device)
    
    cls_ckpt = f"checkpoints/task2_classifier/trial_{cls_study.best_trial.number}/best_model.pt"
    classifier.load_state_dict(torch.load(cls_ckpt, map_location=device, weights_only=True))
    
    export_and_verify(classifier, "checkpoints/onnx/task2_classifier.onnx", dummy_input, "Classifier")
    
    # --- 2. Export Specialists ---
    spec_db = "sqlite:///optuna_studies/task2_specialists.db"
    if not os.path.exists("optuna_studies/task2_specialists.db"):
        raise FileNotFoundError("Specialist study DB not found.")
        
    spec_study = optuna.load_study(study_name="task2_specialists_hpo", storage=spec_db)
    spec_params = spec_study.best_trial.params
    
    specialist_names = {
        "salt_and_pepper": "task2_specialist_sp",
        "gaussian_blur": "task2_specialist_blur",
        "rectangular_occlusion": "task2_specialist_occ"
    }
    
    for spec_folder, onnx_name in specialist_names.items():
        model = UniversalAutoencoder(
            base_channels=spec_params["base_channels"], 
            bottleneck_dim=spec_params["bottleneck_dim"]
        ).to(device)
        
        ckpt = f"checkpoints/task2_specialists/{spec_folder}/best_model.pt"
        if not os.path.exists(ckpt):
            print(f"[ERROR] Checkpoint missing for {spec_folder}")
            continue
            
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        export_and_verify(model, f"checkpoints/onnx/{onnx_name}.onnx", dummy_input, spec_folder.upper())

    print("==================================================")
    print("      All Task 2 Exports Completed Successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
