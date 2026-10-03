import torch
import torch.nn as nn
import optuna

from src.models.task2.classifier import CorruptionClassifier
from src.models.task1.model import UniversalAutoencoder


class SoftMoE(nn.Module):
    """
    Task 3: Soft Mixture-of-Experts restoration model.
    
    Combines a gating network (initialized from the Task 2 classifier) with
    three specialist autoencoders (initialized from Task 2 specialists) and
    an identity bypass branch for clean images.
    
    The gating network produces 4 continuous weights via temperature-scaled
    softmax. The final output is a weighted sum of all branch outputs.
    """
    
    def __init__(self, gate_base_channels: int, gate_dropout: float,
                 expert_base_channels: int, expert_bottleneck_dim: int,
                 tau: float = 1.0):
        """
        Args:
            gate_base_channels: Base channel count for the gating network (from classifier HPO).
            gate_dropout: Dropout rate for the gating network (from classifier HPO).
            expert_base_channels: Base channel count for all expert AEs (from specialist HPO).
            expert_bottleneck_dim: Bottleneck dimension for all expert AEs (from specialist HPO).
            tau: Temperature for softmax routing. Higher = softer/more uniform weights.
        """
        super().__init__()
        
        # Temperature is a buffer, NOT a learnable parameter.
        # Optuna controls it externally to prevent backprop from collapsing it to 0.
        self.register_buffer('tau', torch.tensor(tau, dtype=torch.float32))
        
        # Gating network (reuses CorruptionClassifier architecture)
        self.gate = CorruptionClassifier(
            base_channels=gate_base_channels,
            dropout_rate=gate_dropout
        )
        
        # 3 Expert branches (reuse UniversalAutoencoder architecture)
        self.expert_sp = UniversalAutoencoder(
            base_channels=expert_base_channels,
            bottleneck_dim=expert_bottleneck_dim
        )
        self.expert_blur = UniversalAutoencoder(
            base_channels=expert_base_channels,
            bottleneck_dim=expert_bottleneck_dim
        )
        self.expert_occ = UniversalAutoencoder(
            base_channels=expert_base_channels,
            bottleneck_dim=expert_bottleneck_dim
        )
        # Identity branch has no parameters (just passes input through)
    
    def forward(self, x):
        """
        Args:
            x: Input image tensor [B, 3, 128, 128]
            
        Returns:
            reconstructed: Soft-fused output [B, 3, 128, 128]
            weights: Routing weights [B, 4] (clean, sp, blur, occ)
            logits: Raw gate logits [B, 4] (for CE loss)
        """
        # 1. Compute gate logits and temperature-scaled softmax weights
        logits = self.gate(x)                              # [B, 4]
        weights = torch.softmax(logits / self.tau, dim=1)  # [B, 4]
        
        # 2. Run all expert branches
        out_sp = self.expert_sp(x)    # [B, 3, 128, 128]
        out_blur = self.expert_blur(x)  # [B, 3, 128, 128]
        out_occ = self.expert_occ(x)   # [B, 3, 128, 128]
        
        # 3. Weighted fusion
        # Reshape weights for broadcasting: [B, 1, 1, 1] per expert
        w_clean = weights[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        w_sp    = weights[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        w_blur  = weights[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        w_occ   = weights[:, 3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        reconstructed = (w_clean * x +
                         w_sp * out_sp +
                         w_blur * out_blur +
                         w_occ * out_occ)
        
        return reconstructed, weights, logits
    
    @staticmethod
    def from_pretrained(device='cpu'):
        """
        Factory method that loads the optimal architectures from the Optuna
        databases and initializes all weights from the Task 2 checkpoints.
        
        Returns:
            model: SoftMoE instance with pre-trained weights loaded.
            gate_params: dict of classifier HPO params (for reference).
            expert_params: dict of specialist HPO params (for reference).
        """
        # 1. Load classifier architecture params
        cls_study = optuna.load_study(
            study_name="task2_classifier_hpo",
            storage="sqlite:///optuna_studies/task2_classifier.db"
        )
        cls_params = cls_study.best_trial.params
        
        # 2. Load specialist architecture params
        spec_study = optuna.load_study(
            study_name="task2_specialists_hpo",
            storage="sqlite:///optuna_studies/task2_specialists.db"
        )
        spec_params = spec_study.best_trial.params
        
        # 3. Instantiate the model
        model = SoftMoE(
            gate_base_channels=cls_params["base_channels"],
            gate_dropout=cls_params.get("dropout", 0.0),
            expert_base_channels=spec_params["base_channels"],
            expert_bottleneck_dim=spec_params["bottleneck_dim"],
            tau=1.0  # Default; Optuna will override
        )
        
        # 4. Load pre-trained weights
        cls_ckpt = f"checkpoints/task2_classifier/trial_{cls_study.best_trial.number}/best_model.pt"
        model.gate.load_state_dict(
            torch.load(cls_ckpt, map_location=device, weights_only=True)
        )
        
        for name, folder in [("expert_sp", "salt_and_pepper"),
                             ("expert_blur", "gaussian_blur"),
                             ("expert_occ", "rectangular_occlusion")]:
            ckpt = f"checkpoints/task2_specialists/{folder}/best_model.pt"
            getattr(model, name).load_state_dict(
                torch.load(ckpt, map_location=device, weights_only=True)
            )
        
        return model.to(device), cls_params, spec_params
