import onnxruntime as ort
import os
from app.config import ONNX_MODEL_PATHS

class ModelManager:
    def __init__(self):
        self.sessions = {}

    def load_models(self):
        print("Loading ONNX models...")
        for name, path in ONNX_MODEL_PATHS.items():
            if not os.path.exists(path):
                print(f"[WARNING] Model not found at {path}")
                continue
            
            print(f"Loading {name} from {path}...")
            # Use CPUExecutionProvider for simplicity and cross-platform compatibility without specific CUDA setups
            self.sessions[name] = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        print(f"Loaded {len(self.sessions)} models.")

    def get_session(self, name: str) -> ort.InferenceSession:
        if name not in self.sessions:
            raise KeyError(f"Model {name} not loaded. Available models: {list(self.sessions.keys())}")
        return self.sessions[name]

# Global instance
model_manager = ModelManager()
