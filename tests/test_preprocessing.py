import pytest
import numpy as np
from src.preprocessing import load_image, convert_to_gray, convert_to_hsv, convert_to_rgb

TEST_IMAGE_PATH= "./notebooks/sample.jpeg"
INVALID_IMAGE_PATH = "./notebooks/note_exist.jpeg"

def test_load_image_valid():
    """Tests if load_image correctly returns a 3-channel NumPy array"""
    img = load_image(TEST_IMAGE_PATH)
    assert isinstance(img, np.ndarray)
    assert len(img.shape)== 3
    assert img.shape[2]== 3

def test_load_invalid_image():
    """Tests if the function raises a ValueError for an invalid file path"""
    with pytest.raises(ValueError):
        load_image(INVALID_IMAGE_PATH)

def test_convert_to_gray():
    """Tests if the Grayscale image is correctly reduced to 2 dimensions."""
    img = load_image(TEST_IMAGE_PATH)
    gray = convert_to_gray(img)
    assert len(gray.shape) == 2

def test_convert_to_rgb():
    """Tests if the RGB image retains the original shape."""
    img = load_image(TEST_IMAGE_PATH)
    rgb = convert_to_rgb(img)
    assert rgb.shape == img.shape

def test_convert_to_hsv():
    """Tests if the HSV image retains the original shape."""
    img = load_image(TEST_IMAGE_PATH)
    hsv = convert_to_hsv(img)
    assert hsv.shape == img.shape
