import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Style-Conditioned Face-to-Sketch generation.
    Takes the original photo, a sketch (real or generated), and a style condition.
    Outputs a grid of values representing the probability of patches being real.
    """
    def __init__(self, base_channels: int = 64, embed_dim: int = 16, num_styles: int = 3):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.style_embed = nn.Embedding(num_styles, embed_dim)
        
        # Input channels: Photo (3) + Sketch (3) + Style (embed_dim)
        in_channels = 3 + 3 + embed_dim
        
        # 128x128 -> 64x64
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 64x64 -> 32x32
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 32x32 -> 16x16
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 16x16 -> 15x15 (stride 1)
        self.layer4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 15x15 -> 14x14 (stride 1)
        # 1-channel output (logits)
        self.layer5 = nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, photo, sketch, style_idx):
        B, _, H, W = photo.shape
        
        # Embed style condition
        style_vec = self.style_embed(style_idx)
        style_spatial = style_vec.view(B, self.embed_dim, 1, 1).expand(B, self.embed_dim, H, W)
        
        # Concatenate inputs along channel dimension
        x_in = torch.cat([photo, sketch, style_spatial], dim=1)
        
        out = self.layer1(x_in)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        return out
