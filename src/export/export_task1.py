import os
import torch
import onnx
import optuna
import numpy as np
import onnxruntime as ort

import sys
sys.path.append(os.path.abspath("."))
from src.models.task1.model import UniversalAutoencoder

def export_to_onnx(model, dummy_input, save_path, input_names, output_names, dynamic_axes=None):
    """Export a PyTorch model to ONNX format."""
    model.eval()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes or {input_names[0]: {0: "batch_size"}, output_names[0]: {0: "batch_size"}}
    )
    
    # Verify ONNX model structure
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model structure saved and verified: {save_path}")

def verify_onnx_consistency(pytorch_model, onnx_path, dummy_input, atol=1e-5):
    """Verify ONNX output mathematically matches PyTorch output."""
    pytorch_model.eval()
    with torch.no_grad():
        pt_output = pytorch_model(dummy_input).numpy()
    
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_input = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = session.run(None, ort_input)[0]
    
    max_diff = np.max(np.abs(pt_output - ort_output))
    
    if max_diff < atol:
        print(f"[SUCCESS] ONNX mathematical consistency verified! Max abs diff: {max_diff:.2e} (Limit: {atol})")
    else:
        raise AssertionError(f"[ERROR] ONNX mismatch! Max diff: {max_diff:.2e} exceeds tolerance of {atol}")

def main():
    print("==================================================")
    print("       Task 1 ONNX Export")
    print("==================================================")
    
    # 1. Load the Best Optuna Trial Configuration
    study_db_path = "sqlite:///optuna_studies/task1.db"
    if not os.path.exists("optuna_studies/task1.db"):
        raise FileNotFoundError("Optuna database not found. Please train the model first.")
        
    study = optuna.load_study(study_name="task1_hpo", storage=study_db_path)
    best_trial = study.best_trial
    print(f"Loaded Best Trial: #{best_trial.number}")
    
    # 2. Instantiate Model
    model = UniversalAutoencoder(
        base_channels=best_trial.params["base_channels"],
        bottleneck_dim=best_trial.params["bottleneck_dim"],
        dropout_rate=best_trial.params.get("dropout", 0.0)
    )
    
    # 3. Load Trained Weights
    checkpoint_path = f"checkpoints/task1/trial_{best_trial.number}/best_model.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        
    # weights_only=True disables executing arbitrary unpickling code
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    
    # 4. Export Setup
    dummy_input = torch.randn(1, 3, 128, 128)
    onnx_save_path = "checkpoints/onnx/universal_ae.onnx"
    
    print("\nExporting model to ONNX format...")
    export_to_onnx(
        model=model,
        dummy_input=dummy_input,
        save_path=onnx_save_path,
        input_names=["input_image"],
        output_names=["reconstructed_image"],
    )
    
    print("\nVerifying PyTorch vs ONNX computational consistency...")
    # Use a new random tensor for verification to ensure it handles any input
    verification_input = torch.rand(4, 3, 128, 128) 
    verify_onnx_consistency(
        pytorch_model=model,
        onnx_path=onnx_save_path,
        dummy_input=verification_input,
        atol=1e-5
    )
    print("==================================================")

if __name__ == "__main__":
    main()
