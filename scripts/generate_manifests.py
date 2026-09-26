import os
import json
import random
import numpy as np
from typing import List, Dict

def read_annotation_file(filepath: str) -> List[str]:
    """Reads the annotation file and returns a list of image filenames."""
    filenames = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                filenames.append(parts[0] + '.jpg')
    return filenames

def generate_rects_for_occlusion(w: int, h: int, num_rects: int, target_area_pct: float) -> List[Dict[str, int]]:
    """Generates deterministic rect coordinates for occlusion."""
    total_area = h * w
    target_occluded_area = total_area * target_area_pct
    area_per_rect = target_occluded_area / num_rects
    
    rect_params = []
    for _ in range(num_rects):
        aspect_ratio = random.uniform(0.5, 2.0)
        rect_w = int(np.sqrt(area_per_rect * aspect_ratio))
        rect_h = int(np.sqrt(area_per_rect / aspect_ratio))
        
        rect_w = min(max(1, rect_w), w)
        rect_h = min(max(1, rect_h), h)
        
        x1 = random.randint(0, max(0, w - rect_w))
        y1 = random.randint(0, max(0, h - rect_h))
        
        rect_params.append({'x1': x1, 'y1': y1, 'x2': x1 + rect_w, 'y2': y1 + rect_h})
    return rect_params

def create_val_manifest(val_images: List[str], output_path: str):
    """Creates a manifest for the validation set (1 random corruption per image)."""
    manifest = {}
    random.seed(42)  # Ensure deterministic generation
    
    for img in val_images:
        label = random.randint(0, 3)
        if label == 0:
            ctype, params = "clean", {}
        elif label == 1:
            ctype, params = "salt_and_pepper", {"probability": random.uniform(0.02, 0.15)}
        elif label == 2:
            ctype, params = "gaussian_blur", {"kernel_size": random.choice([3, 5, 7]), "sigma": random.uniform(0.5, 2.5)}
        elif label == 3:
            num_rects = random.randint(1, 3)
            area = random.uniform(0.10, 0.35)
            # All images are 128x128 when loaded in the dataset pipeline
            rects = generate_rects_for_occlusion(128, 128, num_rects, area)
            ctype, params = "rectangular_occlusion", {"rects": rects}
            
        manifest[img] = [{
            "corruption_type": ctype,
            "label": label,
            "params": params
        }]
        
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Validation manifest saved to {output_path} ({len(val_images)} images).")

def create_test_manifest(test_images: List[str], output_path: str):
    """Creates a manifest for the test set (10 corruptions per image: clean + 3 severities * 3 types)."""
    manifest = {}
    random.seed(99)  # Independent seed for test set generation
    
    for img in test_images:
        evaluations = [
            {"corruption_type": "clean", "severity": "none", "label": 0, "params": {}}
        ]
        
        # Salt & Pepper (Low, Medium, High)
        evaluations.append({"corruption_type": "salt_and_pepper", "severity": "low", "label": 1, "params": {"probability": 0.05}})
        evaluations.append({"corruption_type": "salt_and_pepper", "severity": "medium", "label": 1, "params": {"probability": 0.10}})
        evaluations.append({"corruption_type": "salt_and_pepper", "severity": "high", "label": 1, "params": {"probability": 0.15}})
        
        # Gaussian Blur (Low, Medium, High)
        evaluations.append({"corruption_type": "gaussian_blur", "severity": "low", "label": 2, "params": {"kernel_size": 3, "sigma": 1.0}})
        evaluations.append({"corruption_type": "gaussian_blur", "severity": "medium", "label": 2, "params": {"kernel_size": 5, "sigma": 1.5}})
        evaluations.append({"corruption_type": "gaussian_blur", "severity": "high", "label": 2, "params": {"kernel_size": 7, "sigma": 2.5}})
        
        # Rectangular Occlusion (Low, Medium, High)
        # All images are 128x128 when loaded in the dataset pipeline
        rects_low = generate_rects_for_occlusion(128, 128, num_rects=1, target_area_pct=0.10)
        rects_med = generate_rects_for_occlusion(128, 128, num_rects=2, target_area_pct=0.20)
        rects_high = generate_rects_for_occlusion(128, 128, num_rects=3, target_area_pct=0.35)
        
        evaluations.append({"corruption_type": "rectangular_occlusion", "severity": "low", "label": 3, "params": {"rects": rects_low}})
        evaluations.append({"corruption_type": "rectangular_occlusion", "severity": "medium", "label": 3, "params": {"rects": rects_med}})
        evaluations.append({"corruption_type": "rectangular_occlusion", "severity": "high", "label": 3, "params": {"rects": rects_high}})

        manifest[img] = evaluations

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Test manifest saved to {output_path} ({len(test_images)} images, 10 eval tasks each).")

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils.config import load_config

    config = load_config()
    base_data_dir = config.get("data", {}).get("dir", "./data")
    data_dir = os.path.join(base_data_dir, "oxford-iiit-pet")
    manifests_dir = config.get("data", {}).get("manifests_dir", "./manifests")
    
    os.makedirs(manifests_dir, exist_ok=True)
    
    trainval_path = os.path.join(data_dir, "annotations", "trainval.txt")
    test_path = os.path.join(data_dir, "annotations", "test.txt")
    
    if not os.path.exists(trainval_path) or not os.path.exists(test_path):
        print("Error: Dataset annotation files not found. Did you run the download script?")
        exit(1)
        
    # Process Train/Val split
    trainval_images = read_annotation_file(trainval_path)
    # Sort to guarantee determinism before shuffle
    trainval_images.sort()
    
    random.seed(42)
    random.shuffle(trainval_images)
    
    split_idx = int(len(trainval_images) * 0.8)
    train_images = trainval_images[:split_idx]
    val_images = trainval_images[split_idx:]
    
    print(f"Total trainval images: {len(trainval_images)}")
    print(f" -> Training split (80%): {len(train_images)} images (No manifest needed)")
    print(f" -> Validation split (20%): {len(val_images)} images")
    
    create_val_manifest(val_images, os.path.join(manifests_dir, "val_manifest.json"))
    
    # Process Test
    test_images = read_annotation_file(test_path)
    test_images.sort()
    create_test_manifest(test_images, os.path.join(manifests_dir, "test_manifest.json"))
    print("Manifest generation complete.")
