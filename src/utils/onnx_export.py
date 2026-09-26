import torch
import onnx
import onnxruntime as ort
import numpy as np
import os

def export_to_onnx(model: torch.nn.Module, 
                   dummy_input: torch.Tensor, 
                   save_path: str, 
                   input_names: list = None, 
                   output_names: list = None, 
                   dynamic_axes: dict = None):
    """
    Export a PyTorch model to ONNX format.
    """
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]
        
    if dynamic_axes is None:
        dynamic_axes = {
            input_names[0]: {0: "batch_size"}, 
            output_names[0]: {0: "batch_size"}
        }

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model.eval()
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    # Verify the exported model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model successfully saved and verified: {save_path}")

def verify_onnx_consistency(pytorch_model: torch.nn.Module, 
                            onnx_path: str, 
                            dummy_input: torch.Tensor, 
                            atol: float = 1e-5):
    """
    Verify that the ONNX model output matches the PyTorch model output.
    """
    pytorch_model.eval()
    
    # Get PyTorch prediction
    with torch.no_grad():
        pt_output = pytorch_model(dummy_input)
        if isinstance(pt_output, tuple):
            pt_output = pt_output[0] # handle models that return tuples
        pt_output = pt_output.cpu().numpy()
    
    # Get ONNX Runtime prediction (Prioritize NVIDIA GPU, fallback to CPU)
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # We assume a single input for verification purposes
    ort_input = {session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_output = session.run(None, ort_input)[0]
    
    # Check max absolute difference
    max_diff = np.max(np.abs(pt_output - ort_output))
    assert max_diff < atol, f"ONNX consistency check failed! Max diff: {max_diff}"
    
    print(f"ONNX consistency verified: max absolute difference = {max_diff:.2e}")
