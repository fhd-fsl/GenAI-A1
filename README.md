# GenAI Assignment #1 — Image Restoration & Face-to-Sketch Generation

Four generative AI systems integrated into a single web application:

1. **Universal Restoration** — Denoising autoencoder handling clean, salt-and-pepper, blur, and occlusion
2. **Hard-Routed Restoration** — Corruption classifier → specialist autoencoders
3. **Soft Mixture-of-Experts Restoration** — Differentiable gating network + weighted expert blend
4. **Face-to-Sketch Generator** — Style-conditioned conditional GAN (FS2K dataset)

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd GenAI-A1

# 2. Download models (or train from scratch — see below)
python scripts/download_models.py

# 3. Start the application
docker compose up --build

# 4. Open in browser
# http://localhost:3000
```

## Project Structure

```
├── src/                    # Core ML source code
│   ├── data/               # Dataset classes, corruptions, manifests
│   ├── models/             # Model architectures (task1–task4)
│   ├── training/           # Training loops
│   ├── evaluation/         # Evaluation scripts & metrics
│   ├── utils/              # Shared utilities (SSIM, checkpointing, etc.)
│   └── export/             # ONNX export scripts
├── configs/                # YAML configuration files
├── scripts/                # Runnable entry points
├── backend/                # FastAPI backend
│   └── app/
│       ├── main.py
│       ├── routers/        # API endpoints
│       └── services/       # ONNX inference services
├── frontend/               # React + Tailwind frontend
│   └── src/
│       ├── components/
│       └── pages/
├── docker/                 # Dockerfiles
├── manifests/              # Deterministic corruption manifests
├── checkpoints/            # Trained models & ONNX exports
├── optuna_studies/         # Optuna SQLite databases
├── notebooks/              # Exploration notebooks
├── reports/                # LaTeX report & figures
├── tests/                  # Unit & integration tests
└── docker-compose.yml
```

## Training from Scratch

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets
python scripts/download_datasets.py

# Generate deterministic manifests
python scripts/generate_manifests.py

# Train each task
python scripts/train_task1.py --config configs/task1.yaml
python scripts/train_task2.py --config configs/task2.yaml
python scripts/train_task3.py --config configs/task3.yaml
python scripts/train_task4.py --config configs/task4.yaml

# Export to ONNX
python scripts/export_all.py
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | PyTorch |
| HPO | Optuna |
| Experiment Tracking | MLflow / Weights & Biases |
| Inference | ONNX Runtime |
| Backend | FastAPI |
| Frontend | React + Tailwind CSS |
| Deployment | Docker Compose |
