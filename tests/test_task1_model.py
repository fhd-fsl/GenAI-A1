import torch
import pytest
from src.models.task1.encoder import Encoder
from src.models.task1.decoder import Decoder
from src.models.task1.model import UniversalAutoencoder

@pytest.mark.parametrize("base_channels", [32, 48, 64])
@pytest.mark.parametrize("bottleneck_dim", [128, 256, 512])
def test_encoder_shape(base_channels, bottleneck_dim):
    encoder = Encoder(base_channels=base_channels, bottleneck_dim=bottleneck_dim, dropout_rate=0.0)
    # Batch size 2, 3 channels, 128x128 image
    dummy_input = torch.randn(2, 3, 128, 128)
    
    latent = encoder(dummy_input)
    
    assert latent.shape == (2, bottleneck_dim), f"Encoder latent shape mismatch for {base_channels}, {bottleneck_dim}"

@pytest.mark.parametrize("base_channels", [32, 48, 64])
@pytest.mark.parametrize("bottleneck_dim", [128, 256, 512])
def test_decoder_shape(base_channels, bottleneck_dim):
    decoder = Decoder(base_channels=base_channels, bottleneck_dim=bottleneck_dim)
    # Batch size 2, 1D latent vector
    dummy_latent = torch.randn(2, bottleneck_dim)
    
    output = decoder(dummy_latent)
    
    assert output.shape == (2, 3, 128, 128), f"Decoder output shape mismatch for {base_channels}, {bottleneck_dim}"

@pytest.mark.parametrize("base_channels", [32, 64])
def test_universal_autoencoder_end_to_end(base_channels):
    model = UniversalAutoencoder(base_channels=base_channels, bottleneck_dim=256, dropout_rate=0.1)
    dummy_input = torch.randn(2, 3, 128, 128)
    
    output = model(dummy_input)
    
    assert output.shape == (2, 3, 128, 128), "End-to-end Autoencoder shape mismatch"
    
    # Assert values are roughly in [0, 1] range due to Sigmoid
    assert torch.min(output) >= 0.0, "Decoder output contains values below 0.0 (Sigmoid failed?)"
    assert torch.max(output) <= 1.0, "Decoder output contains values above 1.0 (Sigmoid failed?)"
