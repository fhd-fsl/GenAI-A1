# Application Architecture Rules

## Backend (FastAPI)
- Entry point: `backend/app/main.py`
- Routers in `backend/app/routers/` — one per task + health
- Services in `backend/app/services/` — ONNX inference logic
- ONNX sessions loaded ONCE at startup (not per request)
- All endpoints return JSON with timing info (`inference_time_ms`)
- File uploads validated: accept only image types, max 10MB
- Preprocessing (resize, normalize) done server-side before inference

## Frontend (React + Tailwind)
- 4 workspace pages matching assignment naming:
  1. "Universal Restoration"
  2. "Hard-Routed Restoration"
  3. "Soft Mixture-of-Experts Restoration"
  4. "Face-to-Sketch Generator"
- Shared components: ImageUploader, ResultDisplay, MetricsCard
- Must support webcam capture for Task 4

## Docker
- `docker/Dockerfile.backend` — Python 3.12, onnxruntime, FastAPI
- `docker/Dockerfile.frontend` — Node build → nginx
- `docker-compose.yml` at project root
- ONNX models mounted as volume or downloaded via script
