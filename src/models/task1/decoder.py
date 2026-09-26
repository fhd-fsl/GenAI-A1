import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, base_channels: int, bottleneck_dim: int):
        super().__init__()
        
        self.base_channels = base_channels
        self.flatten_dim = base_channels * 8 * 8 * 8
        
        # Project from 1D latent bottleneck back to a spatial feature map
        self.fc = nn.Linear(bottleneck_dim, self.flatten_dim)
        
        # Deconvolutions (ConvTranspose2d) blow the image back up in reverse order
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1), # -> (B, base_channels*4, 16, 16)
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1), # -> (B, base_channels*2, 32, 32)
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1), # -> (B, base_channels, 64, 64)
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 3, kernel_size=4, stride=2, padding=1), # -> (B, 3, 128, 128)
            # Sigmoid because we expect image pixels to be normalized in the [0, 1] range
            nn.Sigmoid()
        )
        
    def forward(self, latent):
        x = self.fc(latent)
        # Reshape back to (Batch, Channels, Height, Width)
        x = x.view(-1, self.base_channels * 8, 8, 8)
        
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        
        return x
