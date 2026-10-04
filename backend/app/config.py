import os
from typing import Dict, List

# Get paths from env var (Docker compatibility) or use relative paths
ONNX_MODEL_DIR = os.getenv("ONNX_MODEL_DIR", "../checkpoints/onnx")

ONNX_MODEL_PATHS: Dict[str, str] = {
    "universal_ae": os.path.join(ONNX_MODEL_DIR, "universal_ae.onnx"),
    "task2_classifier": os.path.join(ONNX_MODEL_DIR, "task2_classifier.onnx"),
    "task2_specialist_sp": os.path.join(ONNX_MODEL_DIR, "task2_specialist_sp.onnx"),
    "task2_specialist_blur": os.path.join(ONNX_MODEL_DIR, "task2_specialist_blur.onnx"),
    "task2_specialist_occ": os.path.join(ONNX_MODEL_DIR, "task2_specialist_occ.onnx"),
    "task3_soft_moe": os.path.join(ONNX_MODEL_DIR, "task3_soft_moe.onnx"),
    "generator": os.path.join(ONNX_MODEL_DIR, "generator.onnx"),
}

IMAGE_SIZE = (128, 128)
CORRUPTION_LABELS: List[str] = ["clean", "salt_and_pepper", "gaussian_blur", "rectangular_occlusion"]
STYLE_LABELS: List[str] = ["Style 1", "Style 2", "Style 3"]
