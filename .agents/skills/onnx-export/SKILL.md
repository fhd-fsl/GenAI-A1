---
name: onnx-export
description: >
  How to export trained PyTorch models to ONNX format and verify consistency.
  Covers all 4 tasks' export requirements.
---

# ONNX Export Skill

## Export Pattern

```python
import torch
import onnx
import onnxruntime as ort
import numpy as np

def export_to_onnx(model, dummy_input, save_path, input_names, output_names, dynamic_axes=None):
    """Export a PyTorch model to ONNX format."""
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
        dynamic_axes=dynamic_axes or {input_names[0]: {0: "batch_size"}, output_names[0]: {0: "batch_size"}}
    )
    
    # Verify
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved and verified: {save_path}")

def verify_onnx_consistency(pytorch_model, onnx_path, dummy_input, atol=1e-5):
    """Verify ONNX output matches PyTorch output."""
    pytorch_model.eval()
    with torch.no_grad():
        pt_output = pytorch_model(dummy_input).numpy()
    
    session = ort.InferenceSession(onnx_path)
    ort_input = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = session.run(None, ort_input)[0]
    
    max_diff = np.max(np.abs(pt_output - ort_output))
    assert max_diff < atol, f"ONNX mismatch! Max diff: {max_diff}"
    print(f"ONNX verified: max diff = {max_diff:.2e}")
```

## Per-Task Exports

### Task 1 — Universal AE
- **Export**: Full autoencoder (encoder + decoder)
- **Input**: `(batch, 3, 128, 128)` float32 [0,1]
- **Output**: `(batch, 3, 128, 128)` float32
- **Save**: `checkpoints/onnx/universal_ae.onnx`

### Task 2 — Classifier + 3 Specialists
- **Export 4 models**:
  - `checkpoints/onnx/classifier.onnx` — input: (B,3,128,128) → output: (B,4) logits
  - `checkpoints/onnx/specialist_salt.onnx` — AE
  - `checkpoints/onnx/specialist_blur.onnx` — AE
  - `checkpoints/onnx/specialist_occlusion.onnx` — AE

### Task 3 — Soft MoE (Gate + Fine-Tuned Experts)
- **Cannot reuse Task 2 ONNX files** — joint fine-tuning updates expert weights,
  so Task 3's experts have diverged from the Task 2 specialists
- **Export gate + all 3 fine-tuned experts separately** (not as one combined model):
  - `checkpoints/onnx/moe/gate.onnx` — input: (B,3,128,128) → output: (B,4) routing weights (post-softmax)
  - `checkpoints/onnx/moe/expert_salt.onnx` — fine-tuned AE
  - `checkpoints/onnx/moe/expert_blur.onnx` — fine-tuned AE
  - `checkpoints/onnx/moe/expert_occlusion.onnx` — fine-tuned AE
- **Inference pipeline** (orchestrated in backend, not inside ONNX):
  1. Run `gate.onnx` → get `[w_clean, w_salt, w_blur, w_occlusion]`
  2. Run each fine-tuned expert ONNX on the input
  3. Weighted sum in numpy: `x̂ = w₁·x̃ + w₂·A_salt(x̃) + w₃·A_blur(x̃) + w₄·A_occlusion(x̃)`
- **Why separate files instead of one combined model**: routing weights are directly
  inspectable for the UI (assignment requires displaying all 4 weights + expert contribution)
- **Why not reuse Task 2 exports**: the warm-up → unfreeze → joint fine-tune cycle
  means expert weights are no longer identical to the Task 2 specialists

### Task 4 — Generator Only
- **Export**: Generator (discriminator is training-only)
- **Input**: `(batch, 3, 128, 128)` photo + `(batch,)` style_index (int64)
- **Output**: `(batch, 1, 128, 128)` or `(batch, 3, 128, 128)` sketch
- **Save**: `checkpoints/onnx/generator.onnx`
- **Note**: Style embedding must be inside the exported model

## Backend Usage
The FastAPI backend loads ONNX models via:
```python
session = ort.InferenceSession("checkpoints/onnx/model.onnx")
result = session.run(None, {"input": preprocessed_image})
```
