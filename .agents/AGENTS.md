# GenAI Assignment #1 — Agent Rules

## Project Summary
This project implements 4 generative AI systems (3 image-restoration + 1 face-to-sketch GAN) integrated into a single web application (React + FastAPI), deployed via Docker Compose.

## Architecture Constraints
- **Framework**: PyTorch for all model training
- **Hyperparameters**: Optuna for every task (mandatory)
- **Experiment tracking**: MLflow or Weights & Biases (mandatory)
- **Inference**: All models exported to ONNX, served via onnxruntime
- **Backend**: FastAPI (Python 3.12)
- **Frontend**: React + Tailwind CSS
- **Deployment**: Docker Compose (local, one-command startup)

## Dataset Rules
- **Tasks 1–3**: Oxford-IIIT Pet Dataset
  - Resolution: 128×128 RGB
  - Split: 80% train / 20% val (seed 42)
  - Test set: official, untouched until final eval
  - Corruption: applied at runtime in data loader (never pre-saved)
- **Task 4**: FS2K Facial Sketch Synthesis Dataset
  - Resolution: 128×128
  - Split: official train/test + 15% stratified val from train (seed 42)
  - Paired transforms only (same spatial augmentation for photo + sketch)

## Coding Conventions
- Use `src/` for all training/model code
- Use `configs/` for YAML config files
- Use `scripts/` for runnable entry points (train, evaluate, export)
- Always set `torch.manual_seed(42)` and equivalent for reproducibility
- Type hints on all function signatures
- Docstrings on all public functions and classes

## File Naming
- Models: `src/models/task{N}/` — `encoder.py`, `decoder.py`, `model.py`
- Training: `src/training/train_task{N}.py`
- Evaluation: `src/evaluation/eval_task{N}.py`
- Config: `configs/task{N}.yaml`
- ONNX export: `src/export/export_task{N}.py`

## ONNX Export Checklist
1. Export with `torch.onnx.export()` using dynamic batch axis
2. Verify with `onnxruntime.InferenceSession`
3. Assert max absolute difference < 1e-5 vs PyTorch output
4. Save to `checkpoints/onnx/`

## Git Rules
- Never commit datasets or large model files
- Use `.gitignore` for `data/`, `checkpoints/`, `optuna_studies/`, `__pycache__/`
- Use Git LFS or download links for trained models
