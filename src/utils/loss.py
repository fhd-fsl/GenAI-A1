import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure

class CombinedL1SSIMLoss(nn.Module):
    """
    Combined L1 (MAE) and SSIM loss function.
    Loss = alpha * L1 + (1 - alpha) * (1 - SSIM)
    
    The SSIM implementation is from torchmetrics and operates on (N, C, H, W) tensors.
    """
    def __init__(self, alpha: float = 0.8, data_range: float = 1.0):
        """
        Args:
            alpha: Weighting factor between L1 and SSIM. 
                   alpha=1.0 is pure L1, alpha=0.0 is pure SSIM.
            data_range: Maximum value range of images (default 1.0 for [0, 1] normalized images).
        """
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Reconstructed images (N, C, H, W)
            target: Ground truth original images (N, C, H, W)
            
        Returns:
            Scalar combined loss tensor.
        """
        loss_l1 = self.l1_loss(preds, target)
        
        # SSIM returns a value between -1 and 1, where 1 is perfect similarity.
        # We convert it to a loss (where 0 is perfect) by taking (1 - ssim).
        loss_ssim = 1.0 - self.ssim(preds, target)
        
        return self.alpha * loss_l1 + (1.0 - self.alpha) * loss_ssim
