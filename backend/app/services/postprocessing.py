import numpy as np
from PIL import Image
import io
import base64

def postprocess_image(tensor: np.ndarray, is_tanh: bool = False) -> str:
    """
    Converts a standard NCHW numpy float32 array back to a base64 encoded PNG string.
    If is_tanh is True, it expects the tensor to be in [-1, 1] and rescales to [0, 1].
    """
    # Remove batch dimension -> CHW
    img_np = np.squeeze(tensor, axis=0)
    
    # CHW to HWC
    img_np = np.transpose(img_np, (1, 2, 0))
    
    if is_tanh:
        # Rescale [-1, 1] -> [0, 1]
        img_np = (img_np + 1.0) / 2.0
        
    # Clamp to [0, 1] just in case
    img_np = np.clip(img_np, 0.0, 1.0)
    
    # Scale to [0, 255] and convert to uint8
    img_np = (img_np * 255.0).astype(np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(img_np)
    
    # Encode as PNG base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return f"data:image/png;base64,{img_str}"
