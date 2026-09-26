import torch
from src.utils.loss import CombinedL1SSIMLoss

def test_combined_loss_shape_and_range():
    loss_fn = CombinedL1SSIMLoss(alpha=0.5)
    
    # Create dummy tensors (N, C, H, W)
    preds = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)
    
    loss = loss_fn(preds, target)
    
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

def test_combined_loss_perfect_match():
    loss_fn = CombinedL1SSIMLoss(alpha=0.8)
    
    preds = torch.ones(2, 3, 64, 64)
    target = torch.ones(2, 3, 64, 64)
    
    loss = loss_fn(preds, target)
    
    # Perfect match should have exactly 0 loss
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5), f"Loss for identical tensors should be 0, got {loss.item()}"
