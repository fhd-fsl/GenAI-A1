import torch
import pytest
from src.data.pet_dataset import balanced_corruption_collate_fn

def test_balanced_collate_fn_perfect_balance():
    # Create 16 dummy clean tensors (simulating batch of 16)
    batch = [torch.ones(3, 128, 128) for _ in range(16)]
    
    corrupted, clean, labels = balanced_corruption_collate_fn(batch)
    
    assert corrupted.shape == (16, 3, 128, 128)
    assert clean.shape == (16, 3, 128, 128)
    assert labels.shape == (16,)
    
    # Check that exactly 4 of each corruption label (0, 1, 2, 3) are present
    counts = torch.bincount(labels)
    assert counts.tolist() == [4, 4, 4, 4], "Batch must be perfectly balanced across 4 corruptions"

def test_balanced_collate_fn_imperfect_batch_size():
    # Test with a batch size not divisible by 4 (e.g. 15)
    batch = [torch.ones(3, 128, 128) for _ in range(15)]
    
    _, _, labels = balanced_corruption_collate_fn(batch)
    counts = torch.bincount(labels)
    
    # Should distribute as evenly as mathematically possible: 4, 4, 4, 3
    assert counts.tolist() == [4, 4, 4, 3], "Collate fn should handle uneven batch sizes gracefully"
