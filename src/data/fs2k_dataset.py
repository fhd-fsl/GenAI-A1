import os
import json
import random
from typing import Tuple, Dict, List
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from sklearn.model_selection import train_test_split

class FS2KDataset(Dataset):
    """
    Dataset for FS2K Facial Sketch Synthesis.
    Loads paired photographs and sketches, with style conditions (0, 1, 2).
    """
    def __init__(self, data_dir: str = "data/FS2K", split: str = "train"):
        self.data_dir = data_dir
        self.split = split
        
        # Load appropriate JSON
        if split in ["train", "val"]:
            anno_path = os.path.join(data_dir, "anno_train.json")
        elif split == "test":
            anno_path = os.path.join(data_dir, "anno_test.json")
        else:
            raise ValueError(f"Unknown split: {split}")
            
        with open(anno_path, 'r') as f:
            annotations = json.load(f)
            
        if split in ["train", "val"]:
            # Perform stratified split (85/15)
            styles = [item["style"] for item in annotations]
            train_items, val_items = train_test_split(
                annotations, test_size=0.15, random_state=42, stratify=styles
            )
            self.items = train_items if split == "train" else val_items
        else:
            self.items = annotations

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        item = self.items[idx]
        image_name = item["image_name"]
        style_label = item["style"]
        
        # Construct paths
        # image_name looks like "photo1/image0110"
        photo_path = os.path.join(self.data_dir, "photo", f"{image_name}.jpg")
        
        # Replace 'photo' with 'sketch' and 'image' with 'sketch' 
        # (e.g. 'photo1/image0110' -> 'sketch1/sketch0110')
        sketch_name = image_name.replace("photo", "sketch").replace("image", "sketch")
        sketch_path = os.path.join(self.data_dir, "sketch", f"{sketch_name}.jpg")
        if not os.path.exists(sketch_path):
            sketch_path = os.path.join(self.data_dir, "sketch", f"{sketch_name}.png")
        
        # Load images
        photo = Image.open(photo_path).convert("RGB")
        sketch = Image.open(sketch_path).convert("RGB")
        
        # Paired Data Augmentation (only for training)
        if self.split == "train":
            # Resize to 144 then random crop to 128
            photo = F.resize(photo, (144, 144))
            sketch = F.resize(sketch, (144, 144))
            
            i, j, h, w = transforms.RandomCrop.get_params(photo, output_size=(128, 128))
            photo = F.crop(photo, i, j, h, w)
            sketch = F.crop(sketch, i, j, h, w)
            
            # Horizontal flip
            if random.random() > 0.5:
                photo = F.hflip(photo)
                sketch = F.hflip(sketch)
            
            # Slight rotation (-10 to 10 degrees)
            angle = random.uniform(-10, 10)
            photo = F.rotate(photo, angle)
            sketch = F.rotate(sketch, angle)
        else:
            # Resize
            photo = F.resize(photo, (128, 128))
            sketch = F.resize(sketch, (128, 128))
            
        # Convert to Tensor [0, 1]
        photo_tensor = F.to_tensor(photo)
        sketch_tensor = F.to_tensor(sketch)
        
        # Normalize to [-1, 1] for Tanh compatibility
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        photo_tensor = normalize(photo_tensor)
        sketch_tensor = normalize(sketch_tensor)
        
        return photo_tensor, sketch_tensor, style_label
