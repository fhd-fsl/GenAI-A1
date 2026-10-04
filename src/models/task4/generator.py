import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    """
    U-Net Generator for Style-Conditioned Face-to-Sketch generation.
    Incorporates a categorical style embedding concatenated to the input.
    """
    def __init__(self, base_channels: int = 64, embed_dim: int = 16, num_styles: int = 3, dropout: float = 0.5):
        super().__init__()
        
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.style_embed = nn.Embedding(num_styles, embed_dim)
        
        # Input is 3 (RGB) + embed_dim
        in_channels = 3 + embed_dim
        
        # Encoder
        # 128x128 -> 64x64
        self.e1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 64x64 -> 32x32
        self.e2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 32x32 -> 16x16
        self.e3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 16x16 -> 8x8
        self.e4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 8x8 -> 4x4
        self.e5 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 4x4 -> 2x2 (Bottleneck)
        self.e6 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (with skip connections)
        # 2x2 -> 4x4
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.Dropout(self.dropout),
            nn.ReLU(inplace=True)
        )
        
        # 4x4 -> 8x8 (input is 16*base_channels due to skip from e5)
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.Dropout(self.dropout),
            nn.ReLU(inplace=True)
        )
        
        # 8x8 -> 16x16 (input is 16*base_channels due to skip from e4)
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 16, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        # 16x16 -> 32x32 (input is 8*base_channels due to skip from e3)
        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # 32x32 -> 64x64 (input is 4*base_channels due to skip from e2)
        self.d5 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 64x64 -> 128x128 (input is 2*base_channels due to skip from e1)
        # Output is 3 channels (RGB)
        self.d6 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Output in range [-1, 1]
        )

    def forward(self, x, style_idx):
        B, _, H, W = x.shape
        
        # Embed style condition
        style_vec = self.style_embed(style_idx) # [B, embed_dim]
        style_spatial = style_vec.view(B, self.embed_dim, 1, 1).expand(B, self.embed_dim, H, W)
        
        # Concatenate condition
        x_in = torch.cat([x, style_spatial], dim=1) # [B, 3 + embed_dim, H, W]
        
        # Encoder passes
        e1_out = self.e1(x_in)
        e2_out = self.e2(e1_out)
        e3_out = self.e3(e2_out)
        e4_out = self.e4(e3_out)
        e5_out = self.e5(e4_out)
        e6_out = self.e6(e5_out)
        
        # Decoder passes with skip connections
        d1_out = self.d1(e6_out)
        d1_cat = torch.cat([d1_out, e5_out], dim=1)
        
        d2_out = self.d2(d1_cat)
        d2_cat = torch.cat([d2_out, e4_out], dim=1)
        
        d3_out = self.d3(d2_cat)
        d3_cat = torch.cat([d3_out, e3_out], dim=1)
        
        d4_out = self.d4(d3_cat)
        d4_cat = torch.cat([d4_out, e2_out], dim=1)
        
        d5_out = self.d5(d4_cat)
        d5_cat = torch.cat([d5_out, e1_out], dim=1)
        
        d6_out = self.d6(d5_cat)
        
        return d6_out
