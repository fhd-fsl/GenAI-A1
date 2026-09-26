import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, base_channels: int, bottleneck_dim: int, dropout_rate: float):
        super().__init__()
        
        # Input: (Batch, 3, 128, 128)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=4, stride=2, padding=1), # -> (B, base_channels, 64, 64)
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1), # -> (B, base_channels*2, 32, 32)
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1), # -> (B, base_channels*4, 16, 16)
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1), # -> (B, base_channels*8, 8, 8)
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 8 * 8 spatial grid * (base_channels * 8) channels
        self.flatten_dim = base_channels * 8 * 8 * 8
        self.fc = nn.Linear(self.flatten_dim, bottleneck_dim)
        
    def forward(self, x):
        # We explicitly do NOT return skip connections. The prompt mandates a genuine bottleneck.
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        latent = self.fc(x)
        
        return latent
