import os
import sys
import torch
import optuna
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.task3.soft_moe import SoftMoE


def main():
    device = torch.device("cpu")  # ONNX export requires CPU tensors

    # 1. Load best params
    study = optuna.load_study(
        study_name="task3_moe_hpo",
        storage="sqlite:///optuna_studies/task3_moe.db"
    )
    params = study.best_trial.params

    # 2. Load the fine-tuned model
    model, _, _ = SoftMoE.from_pretrained(device=device)
    model.tau.fill_(params["temperature"])

    ckpt = "checkpoints/task3_moe/best_model.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    # 3. Export to ONNX
    os.makedirs("checkpoints/onnx", exist_ok=True)
    save_path = "checkpoints/onnx/task3_soft_moe.onnx"
    dummy_input = torch.ones(1, 3, 128, 128, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['reconstructed', 'weights', 'logits'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'reconstructed': {0: 'batch_size'},
            'weights': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    print(f"[SoftMoE] Successfully exported to {save_path}")

    # 4. Verify with ONNX Runtime
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARNING] onnxruntime not installed. Skipping verification.")
        return

    ort_session = ort.InferenceSession(save_path, providers=['CPUExecutionProvider'])

    with torch.no_grad():
        torch_recon, torch_weights, torch_logits = model(dummy_input)
        torch_recon = torch_recon.numpy()
        torch_weights = torch_weights.numpy()
        torch_logits = torch_logits.numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    recon_diff = np.max(np.abs(torch_recon - ort_outs[0]))
    weights_diff = np.max(np.abs(torch_weights - ort_outs[1]))
    logits_diff = np.max(np.abs(torch_logits - ort_outs[2]))
    
    print(f"Max diff Recon: {recon_diff}")
    print(f"Max diff Weights: {weights_diff}")
    print(f"Max diff Logits: {logits_diff}")

    assert recon_diff < 1e-5, f"Reconstruction diff {recon_diff} exceeds 1e-5"
    assert weights_diff < 1e-5, f"Weights diff {weights_diff} exceeds 1e-5"
    assert logits_diff < 1e-5, f"Logits diff {logits_diff} exceeds 1e-5"

    print("[SoftMoE] Verification SUCCESS! All outputs match within strict absolute tolerance.\n")

    print("==================================================")
    print("   Task 3 ONNX Export Completed Successfully!")
    print("==================================================")


if __name__ == "__main__":
    main()
