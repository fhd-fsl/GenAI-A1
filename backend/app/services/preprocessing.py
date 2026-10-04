import numpy as np
from PIL import Image
import io
from app.config import IMAGE_SIZE

def decode_image(image_bytes: bytes) -> Image.Image:
    """Decodes raw bytes into a PIL Image, forcing RGB."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def preprocess_image(image: Image.Image, normalize_to_minus_one_one: bool = False) -> np.ndarray:
    """
    Resizes image to 128x128 and converts it to a standard NCHW numpy array float32.
    If normalize_to_minus_one_one is True, scales to [-1, 1]. Otherwise [0, 1].
    """
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    
    # Convert to numpy and normalize to [0, 1]
    img_np = np.array(image, dtype=np.float32) / 255.0
    
    if normalize_to_minus_one_one:
        # Scale to [-1, 1] via (x - 0.5) / 0.5
        img_np = (img_np - 0.5) / 0.5
        
    # HWC to CHW
    img_np = np.transpose(img_np, (2, 0, 1))
    
    # Add batch dimension -> NCHW
    img_np = np.expand_dims(img_np, axis=0)
    
    return img_np
