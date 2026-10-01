import torch
import torch.nn as nn

class CorruptionClassifier(nn.Module):
    def __init__(self, base_channels=32, dropout_rate=0.2):
        """
        A convolutional classifier to predict the corruption type applied to an image.
        Outputs 4 logits corresponding to: [clean, salt_and_pepper, gaussian_blur, rectangular_occlusion]
        """
        super().__init__()
        
        # Spatial size: 128x128
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, base_channels, kernel_size=3, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2),
            
            # Block 2
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
            
            # Block 3
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2),
            
            # Block 4
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2),
            
            # Block 5
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=2, padding=1), # 4x4
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(base_channels * 8, 4) # 4 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
