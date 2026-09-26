import torch
import torch.nn as nn
from src.models.task1.encoder import Encoder
from src.models.task1.decoder import Decoder

class UniversalAutoencoder(nn.Module):
    """
    Task 1: Universal Denoising Autoencoder.
    Contains a strict bottleneck (no unrestricted skip connections) to ensure 
    the model learns a universal compressed representation for all corruptions.
    """
    def __init__(self, base_channels: int = 32, bottleneck_dim: int = 256, dropout_rate: float = 0.2):
        super().__init__()
        
        self.encoder = Encoder(
            base_channels=base_channels, 
            bottleneck_dim=bottleneck_dim, 
            dropout_rate=dropout_rate
        )
        
        self.decoder = Decoder(
            base_channels=base_channels, 
            bottleneck_dim=bottleneck_dim
        )
        
    def forward(self, x):
        # Forward pass through encoder to get the 1D latent vector
        latent = self.encoder(x)
        
        # Forward pass through decoder to reconstruct the 128x128 RGB image
        reconstructed = self.decoder(latent)
        
        return reconstructed
