import numpy as np
import scipy.ndimage as ndimage
import random

def apply_salt_and_pepper(image: np.ndarray, prob: float) -> np.ndarray:
    """
    Apply salt and pepper noise to a numpy image (shape NCHW or CHW, [0,1]).
    Expects input to be [0,1].
    """
    c, h, w = image.shape[-3:]
    corrupted = image.copy()
    
    # Generate random mask
    rand_mask = np.random.rand(h, w)
    
    salt_mask = rand_mask < (prob / 2.0)
    pepper_mask = (rand_mask >= (prob / 2.0)) & (rand_mask < prob)
    
    # Apply to all channels
    if corrupted.ndim == 4:
        for i in range(c):
            corrupted[0, i, salt_mask] = 1.0
            corrupted[0, i, pepper_mask] = 0.0
    else:
        for i in range(c):
            corrupted[i, salt_mask] = 1.0
            corrupted[i, pepper_mask] = 0.0
            
    return corrupted

def apply_gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur to a numpy image.
    Uses scipy.ndimage.gaussian_filter on each channel independently.
    """
    # Create a copy
    corrupted = image.copy()
    
    # Since ndimage.gaussian_filter applies to the whole array, 
    # we need to be careful with spatial vs channel dimensions.
    if corrupted.ndim == 4:
        # shape (B, C, H, W)
        for b in range(corrupted.shape[0]):
            for c in range(corrupted.shape[1]):
                corrupted[b, c] = ndimage.gaussian_filter(corrupted[b, c], sigma=sigma, radius=kernel_size//2)
    else:
        # shape (C, H, W)
        for c in range(corrupted.shape[0]):
            corrupted[c] = ndimage.gaussian_filter(corrupted[c], sigma=sigma, radius=kernel_size//2)
            
    return corrupted

def apply_rectangular_occlusion(image: np.ndarray, num_rects: int, target_area_pct: float) -> np.ndarray:
    """
    Apply black rectangular occlusions to a numpy image.
    """
    h, w = image.shape[-2:]
    corrupted = image.copy()
    
    total_area = h * w
    target_occluded_area = total_area * target_area_pct
    area_per_rect = target_occluded_area / num_rects
    
    for _ in range(num_rects):
        aspect_ratio = random.uniform(0.5, 2.0)
        
        rect_w = int(np.sqrt(area_per_rect * aspect_ratio))
        rect_h = int(np.sqrt(area_per_rect / aspect_ratio))
        
        rect_w = min(max(1, rect_w), w)
        rect_h = min(max(1, rect_h), h)
        
        x1 = random.randint(0, w - rect_w)
        y1 = random.randint(0, h - rect_h)
        x2 = x1 + rect_w
        y2 = y1 + rect_h
        
        if corrupted.ndim == 4:
            corrupted[:, :, y1:y2, x1:x2] = 0.0
        else:
            corrupted[:, y1:y2, x1:x2] = 0.0
            
    return corrupted

def apply_deterministic_corruption(image: np.ndarray, corruption_type: str, params: dict) -> np.ndarray:
    """
    Applies the requested corruption using the provided parameters.
    """
    if corruption_type == "clean" or not corruption_type or corruption_type == "none":
        return image.copy()
        
    elif corruption_type == "salt_and_pepper":
        prob = float(params.get("probability", 0.05))
        return apply_salt_and_pepper(image, prob)
        
    elif corruption_type == "gaussian_blur":
        kernel_size = int(params.get("kernel_size", 5))
        sigma = float(params.get("sigma", 1.5))
        return apply_gaussian_blur(image, kernel_size, sigma)
        
    elif corruption_type == "rectangular_occlusion":
        num_rects = int(params.get("num_rects", 2))
        area_pct = float(params.get("area_pct", 0.2))
        return apply_rectangular_occlusion(image, num_rects, area_pct)
        
    return image.copy()
