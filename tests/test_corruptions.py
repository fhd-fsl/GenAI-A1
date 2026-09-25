import torch
import pytest
from src.data.corruptions import (
    apply_salt_and_pepper,
    apply_gaussian_blur,
    apply_rectangular_occlusion,
    get_random_corruption,
    apply_deterministic_corruption
)

@pytest.fixture
def dummy_image():
    # 3 channels, 128x128, all values 0.5
    return torch.ones(3, 128, 128) * 0.5

def test_apply_salt_and_pepper(dummy_image):
    prob = 0.1
    corrupted = apply_salt_and_pepper(dummy_image, prob=prob)
    
    assert corrupted.shape == dummy_image.shape
    # Check that values are only 0.0, 0.5, or 1.0
    unique_vals = torch.unique(corrupted)
    for val in unique_vals:
        assert val.item() in [0.0, 0.5, 1.0]

def test_apply_gaussian_blur(dummy_image):
    corrupted = apply_gaussian_blur(dummy_image, kernel_size=5, sigma=1.5)
    
    assert corrupted.shape == dummy_image.shape
    # Blurring a constant image should leave it mostly unchanged
    assert torch.allclose(corrupted, dummy_image, atol=1e-3)

def test_apply_rectangular_occlusion(dummy_image):
    num_rects = 2
    target_area_pct = 0.20
    corrupted, rect_params = apply_rectangular_occlusion(dummy_image, num_rects, target_area_pct)
    
    assert corrupted.shape == dummy_image.shape
    assert len(rect_params) == num_rects
    
    # Check that at least some pixels were zeroed out
    assert (corrupted == 0.0).any()
    
    # Verify the area is roughly 20%
    zero_pixels_per_channel = (corrupted[0] == 0.0).sum().item()
    total_pixels = 128 * 128
    actual_area_pct = zero_pixels_per_channel / total_pixels
    # Should be close to 0.20, but might be slightly less if rects overlap or clip
    assert 0.0 < actual_area_pct <= 0.25

def test_get_random_corruption(dummy_image):
    corrupted, label, params = get_random_corruption(dummy_image)
    
    assert corrupted.shape == dummy_image.shape
    assert 0 <= label <= 3
    assert isinstance(params, dict)

def test_apply_deterministic_corruption(dummy_image):
    # Test S&P
    corrupted, label = apply_deterministic_corruption(
        dummy_image, "salt_and_pepper", {"probability": 0.1}
    )
    assert label == 1
    assert corrupted.shape == dummy_image.shape

    # Test Blur
    corrupted, label = apply_deterministic_corruption(
        dummy_image, "gaussian_blur", {"kernel_size": 3, "sigma": 1.0}
    )
    assert label == 2
    assert corrupted.shape == dummy_image.shape

    # Test Occlusion
    rects = [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}]
    corrupted, label = apply_deterministic_corruption(
        dummy_image, "rectangular_occlusion", {"rects": rects}
    )
    assert label == 3
    assert corrupted.shape == dummy_image.shape
    assert (corrupted[:, 10:50, 10:50] == 0.0).all()
