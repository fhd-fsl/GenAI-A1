import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
import sys
import optuna
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.task4.generator import UNetGenerator

def export_generator():
    print("==================================================")
    print("       Task 4: ONNX Export (Generator Only)")
    print("==================================================")

    # 1. Load config from Optuna directly
    db_path = "sqlite:///optuna_studies/task4.db"
    study = optuna.load_study(study_name="task4_hpo", storage=db_path)
    config = study.best_trial.params
    print(f"Loaded Best Trial: #{study.best_trial.number}")
        
    # 2. Instantiate and load model
    device = torch.device("cpu") # Export on CPU
    generator = UNetGenerator(
        base_channels=config["base_channels"],
        embed_dim=config["embed_dim"],
        dropout=config.get("dropout", 0.5)
    ).to(device)
    
    # Use the final model since validation loss is unreliable for GANs
    checkpoint_path = "checkpoints/task4/generator_final.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return
        
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    generator.eval()
    
    # 3. Create dummy inputs
    # photo (B, 3, 128, 128) float32
    dummy_photo = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    # style_idx (B,) int64
    dummy_style = torch.tensor([0], dtype=torch.long)
    dummy_input = (dummy_photo, dummy_style)
    
    os.makedirs("checkpoints/onnx", exist_ok=True)
    save_path = "checkpoints/onnx/generator.onnx"
    
    # 4. Export
    input_names = ["photo", "style_idx"]
    output_names = ["generated_sketch"]
    dynamic_axes = {
        "photo": {0: "batch_size"},
        "style_idx": {0: "batch_size"},
        "generated_sketch": {0: "batch_size"}
    }
    
    torch.onnx.export(
        generator,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    # 5. Verify ONNX validity
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved and verified: {save_path}")
    
    # 6. Verify PyTorch vs ONNX consistency
    with torch.no_grad():
        pt_output = generator(dummy_photo, dummy_style).numpy()
        
    session = ort.InferenceSession(save_path)
    ort_inputs = {
        session.get_inputs()[0].name: dummy_photo.numpy(),
        session.get_inputs()[1].name: dummy_style.numpy()
    }
    ort_output = session.run(None, ort_inputs)[0]
    
    max_diff = np.max(np.abs(pt_output - ort_output))
    assert max_diff < 1e-5, f"ONNX mismatch! Max diff: {max_diff}"
    print(f"ONNX consistency verified: max diff = {max_diff:.2e}")
    print("==================================================")

if __name__ == "__main__":
    export_generator()
