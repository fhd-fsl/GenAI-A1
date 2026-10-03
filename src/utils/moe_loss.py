import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure


class MoEJointLoss(nn.Module):
    """
    Joint loss for Soft Mixture-of-Experts training.
    
    L_joint = lambda_recon * L1(x_hat, x_clean)
            + lambda_ssim * (1 - SSIM(x_hat, x_clean))
            + lambda_ce * CrossEntropy(logits, true_label)
            + lambda_balance * L_balance(weights)
    
    The balance loss prevents routing collapse by penalizing deviation
    from a uniform 1/4 distribution across the 4 branches.
    """
    
    def __init__(self, lambda_recon: float = 0.8, lambda_ssim: float = 0.2,
                 lambda_ce: float = 0.1, lambda_balance: float = 0.01):
        """
        Args:
            lambda_recon: Weight for L1 reconstruction loss.
            lambda_ssim: Weight for (1 - SSIM) structural loss.
            lambda_ce: Weight for cross-entropy classification loss.
            lambda_balance: Weight for routing balance regularizer.
        """
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_ssim = lambda_ssim
        self.lambda_ce = lambda_ce
        self.lambda_balance = lambda_balance
        
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    
    def forward(self, x_hat, x_clean, logits, weights, true_labels):
        """
        Args:
            x_hat: Reconstructed image [B, 3, 128, 128]
            x_clean: Clean target image [B, 3, 128, 128]
            logits: Raw gate logits [B, 4] (for CE loss)
            weights: Routing weights [B, 4] (for balance loss)
            true_labels: Ground-truth corruption class indices [B] (int)
            
        Returns:
            total_loss: Scalar combined loss
            loss_dict: Dictionary of individual loss components (for logging)
        """
        # 1. Reconstruction loss (L1)
        loss_l1 = self.l1_loss(x_hat, x_clean)
        
        # 2. Structural similarity loss
        loss_ssim = 1.0 - self.ssim(x_hat, x_clean)
        
        # 3. Cross-entropy classification loss
        loss_ce = self.ce_loss(logits, true_labels)
        
        # 4. Balance loss: penalize deviation from uniform 1/4 per branch
        w_bar = weights.mean(dim=0)  # [4] - average weight per branch across batch
        loss_balance = ((w_bar - 0.25) ** 2).sum()
        
        # Combined
        total_loss = (self.lambda_recon * loss_l1 +
                      self.lambda_ssim * loss_ssim +
                      self.lambda_ce * loss_ce +
                      self.lambda_balance * loss_balance)
        
        loss_dict = {
            "loss_l1": loss_l1.item(),
            "loss_ssim": loss_ssim.item(),
            "loss_ce": loss_ce.item(),
            "loss_balance": loss_balance.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_dict
