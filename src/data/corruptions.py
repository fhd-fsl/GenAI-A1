import torch
import torchvision.transforms.functional as F
import random
import numpy as np

def apply_salt_and_pepper(image: torch.Tensor, prob: float) -> torch.Tensor:
    """
    Apply salt and pepper noise to an image tensor.
    
    Args:
        image: Tensor of shape (C, H, W) with values in [0, 1].
        prob: Probability of replacing a pixel with salt (1) or pepper (0).
        
    Returns:
        Corrupted image tensor of shape (C, H, W).
    """
    c, h, w = image.shape
    corrupted = image.clone()
    
    # Generate a random mask for the entire image (applied to all channels equally)
    # 0 = keep original, 1 = salt, 2 = pepper
    rand_mask = torch.rand(h, w, device=image.device)
    
    salt_mask = rand_mask < (prob / 2.0)
    pepper_mask = (rand_mask >= (prob / 2.0)) & (rand_mask < prob)
    
    # Expand masks to match channels
    salt_mask = salt_mask.unsqueeze(0).expand(c, -1, -1)
    pepper_mask = pepper_mask.unsqueeze(0).expand(c, -1, -1)
    
    # Apply noise
    corrupted[salt_mask] = 1.0
    corrupted[pepper_mask] = 0.0
    
    return corrupted

def apply_gaussian_blur(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Apply Gaussian blur to an image tensor.
    
    Args:
        image: Tensor of shape (C, H, W).
        kernel_size: Size of the Gaussian kernel (e.g., 3, 5, 7).
        sigma: Standard deviation of the Gaussian kernel.
        
    Returns:
        Corrupted image tensor of shape (C, H, W).
    """
    return F.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

def apply_rectangular_occlusion(image: torch.Tensor, num_rects: int, target_area_pct: float) -> torch.Tensor:
    """
    Apply rectangular occlusion (black boxes) to an image tensor.
    
    Args:
        image: Tensor of shape (C, H, W).
        num_rects: Number of rectangles to draw (1 to 3).
        target_area_pct: Total target area percentage to occlude (0.10 to 0.35).
        
    Returns:
        Corrupted image tensor of shape (C, H, W) and list of rect params for tracking if needed.
    """
    c, h, w = image.shape
    corrupted = image.clone()
    total_area = h * w
    target_occluded_area = total_area * target_area_pct
    
    # Area per rectangle
    area_per_rect = target_occluded_area / num_rects
    
    rect_params = []
    
    for _ in range(num_rects):
        # Determine aspect ratio randomly to calculate w and h of the rectangle
        aspect_ratio = random.uniform(0.5, 2.0)
        
        # area = rect_w * rect_h = rect_w * (rect_w / aspect_ratio)
        rect_w = int(np.sqrt(area_per_rect * aspect_ratio))
        rect_h = int(np.sqrt(area_per_rect / aspect_ratio))
        
        # Ensure it fits in the image
        rect_w = min(max(1, rect_w), w)
        rect_h = min(max(1, rect_h), h)
        
        # Random top-left corner
        x1 = random.randint(0, w - rect_w)
        y1 = random.randint(0, h - rect_h)
        
        x2 = x1 + rect_w
        y2 = y1 + rect_h
        
        # Apply black occlusion (0.0)
        corrupted[:, y1:y2, x1:x2] = 0.0
        
        rect_params.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
        
    return corrupted, rect_params

def get_random_corruption(image: torch.Tensor):
    """
    Applies one of the four corruptions randomly (used for training).
    Returns the corrupted image, the integer label, and the parameters used.
    
    Labels:
    0 = Clean
    1 = Salt and pepper
    2 = Gaussian blur
    3 = Rectangular occlusion
    """
    label = random.randint(0, 3)
    
    if label == 0:
        return image.clone(), label, {}
        
    elif label == 1:
        prob = random.uniform(0.02, 0.15)
        return apply_salt_and_pepper(image, prob), label, {"probability": prob}
        
    elif label == 2:
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.5, 2.5)
        return apply_gaussian_blur(image, kernel_size, sigma), label, {"kernel_size": kernel_size, "sigma": sigma}
        
    elif label == 3:
        num_rects = random.randint(1, 3)
        target_area_pct = random.uniform(0.10, 0.35)
        corrupted, rect_params = apply_rectangular_occlusion(image, num_rects, target_area_pct)
        return corrupted, label, {"rects": rect_params}
    
    raise ValueError("Invalid corruption label generated")

def apply_deterministic_corruption(image: torch.Tensor, corruption_type: str, params: dict):
    """
    Applies a specific corruption using provided params (used for val/test manifests).
    """
    if corruption_type == "clean":
        return image.clone(), 0
        
    elif corruption_type == "salt_and_pepper":
        return apply_salt_and_pepper(image, params["probability"]), 1
        
    elif corruption_type == "gaussian_blur":
        return apply_gaussian_blur(image, params["kernel_size"], params["sigma"]), 2
        
    elif corruption_type == "rectangular_occlusion":
        corrupted = image.clone()
        for rect in params["rects"]:
            x1, y1, x2, y2 = rect["x1"], rect["y1"], rect["x2"], rect["y2"]
            corrupted[:, y1:y2, x1:x2] = 0.0
        return corrupted, 3
        
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
