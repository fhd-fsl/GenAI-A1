import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import List, Dict, Tuple

from src.data.corruptions import (
    apply_salt_and_pepper,
    apply_gaussian_blur,
    apply_rectangular_occlusion,
    apply_deterministic_corruption
)

def get_train_images(data_dir: str) -> List[str]:
    """Recalculate the training split (80% of trainval.txt) deterministically."""
    trainval_path = os.path.join(data_dir, "annotations", "trainval.txt")
    filenames = []
    with open(trainval_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                filenames.append(parts[0] + '.jpg')
    filenames.sort()
    random.seed(42)
    random.shuffle(filenames)
    split_idx = int(len(filenames) * 0.8)
    return filenames[:split_idx]

class OxfordPetDataset(Dataset):
    """
    Dataset for Oxford-IIIT Pet images.
    - If split='train': Returns clean tensors. The collate_fn handles applying balanced corruptions.
    - If split='val' or 'test': Returns (corrupted_tensor, clean_tensor, label) directly from manifest.
    """
    def __init__(self, data_dir: str, split: str = "train", manifests_dir: str = "./manifests", transform=None):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        self.items = [] # Will hold (filename, corruption_dict) or just filename for train
        
        if split == "train":
            self.items = get_train_images(self.data_dir)
        else:
            manifest_name = "val_manifest.json" if split == "val" else "test_manifest.json"
            manifest_path = os.path.join(manifests_dir, manifest_name)
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                
            # Flatten the manifest: image -> list of evals becomes a flat list of (image, eval)
            for img_name, evals in manifest.items():
                for eval_dict in evals:
                    self.items.append((img_name, eval_dict))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        if self.split == "train":
            img_name = self.items[idx]
            eval_dict = None
        else:
            img_name, eval_dict = self.items[idx]
            
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        clean_tensor = self.transform(image)
        
        if self.split == "train":
            # Just return the clean tensor. The collate_fn will handle the rest.
            return clean_tensor
        else:
            # Apply exact deterministic corruption using the helper in corruptions.py
            corrupted_tensor, label = apply_deterministic_corruption(
                clean_tensor, 
                eval_dict["corruption_type"], 
                eval_dict["params"]
            )
            # Instead of just the integer label, return the full metadata dict
            metadata = {
                "corruption": eval_dict["corruption_type"],
                "severity": eval_dict.get("severity", "none"),
                "label": eval_dict["label"]
            }
            return corrupted_tensor, clean_tensor, metadata


def balanced_corruption_collate_fn(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for training. 
    Takes a batch of N clean tensors and applies exactly balanced corruptions on the fly.
    """
    clean_images = torch.stack(batch, dim=0)
    batch_size = clean_images.size(0)
    
    corrupted_images = []
    labels = []
    
    # Cycle through labels 0, 1, 2, 3 to ensure perfect batch balance
    for i in range(batch_size):
        label = i % 4
        clean_tensor = clean_images[i]
        
        if label == 0:
            corr = clean_tensor.clone()
        elif label == 1:
            p = random.uniform(0.02, 0.15)
            corr = apply_salt_and_pepper(clean_tensor, prob=p)
        elif label == 2:
            kernel = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 2.5)
            corr = apply_gaussian_blur(clean_tensor, kernel_size=kernel, sigma=sigma)
        elif label == 3:
            num_rects = random.randint(1, 3)
            area_pct = random.uniform(0.10, 0.35)
            # apply_rectangular_occlusion returns (corrupted_tensor, rect_params)
            corr, _ = apply_rectangular_occlusion(clean_tensor, num_rects=num_rects, target_area_pct=area_pct)
            
        corrupted_images.append(corr)
        labels.append(label)
        
    corrupted_batch = torch.stack(corrupted_images, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Shuffle the batch so the labels aren't strictly predictable 0,1,2,3...
    indices = torch.randperm(batch_size)
    corrupted_batch = corrupted_batch[indices]
    clean_images = clean_images[indices]
    labels_tensor = labels_tensor[indices]
    
    return corrupted_batch, clean_images, labels_tensor
