
import cv2
import numpy as np
import pytest

from src.preprocessing import (
    load_image,
    gaussian_blur,
    apply_threshold,
    detect_edges,
    erode_image,
    dilate_image,
    morphological_opening,
    morphological_closing,
    _get_kernel,
)

# Load image

def test_load_image_rgb(tmp_path):
    """Test loading an image in RGB format."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    image[:, :, 2] = 255  # Red channel in BGR format

    image_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(image_path), image)

    result = load_image(str(image_path))

    assert result.shape == (50, 50, 3)
    assert result.dtype == np.uint8
    assert result[0, 0, 0] > result[0, 0, 2]


def test_load_image_grayscale(tmp_path):
    """Test loading an image in grayscale format."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)

    image_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(image_path), image)

    result = load_image(str(image_path), as_gray=True)

    assert result.shape == (50, 50)
    assert result.dtype == np.uint8


def test_load_image_invalid_path():
    """Test loading an image from a non-existent path."""
    with pytest.raises(ValueError):
        load_image("not_exist.jpg")

# Gaussian blur

def test_gaussian_blur():
    """Test Gaussian blur output dimensions."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image[50, 50] = 255

    result = gaussian_blur(image)

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert result[50, 50] < 255


def test_gaussian_blur_invalid_kernel():
    """Test Gaussian blur with an invalid kernel size."""
    image = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(ValueError):
        gaussian_blur(image, kernel_size=(4, 4))

# Thresholding

def test_apply_threshold():
    """Test binary thresholding."""
    image = np.array([
        [50, 100],
        [150, 200]
    ], dtype=np.uint8)

    result = apply_threshold(image, threshold_value=127)

    assert result.shape == image.shape
    assert set(np.unique(result)).issubset({0, 255})
    assert result[0, 0] == 0
    assert result[1, 0] == 255


def test_apply_threshold_invalid_input():
    """Test thresholding with a color image."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_threshold(image)


# Edge detection

def test_detect_edges():
    """Test Canny edge detection."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image[30:70, 30:70] = 255

    result = detect_edges(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.any(result > 0)


def test_detect_edges_invalid_input():
    """Test Canny with a color image."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        detect_edges(image)


def test_detect_edges_invalid_thresholds():
    """Test Canny with invalid threshold values."""
    image = np.zeros((50, 50), dtype=np.uint8)

    with pytest.raises(ValueError):
        detect_edges(image, threshold1=150, threshold2=50)

# Kernel

def test_get_kernel():
    """Test structuring element dimensions."""
    kernel = _get_kernel((3, 5))

    assert kernel.shape == (3, 5)
    assert kernel.dtype == np.uint8
    assert np.all(kernel == 1)


def test_get_kernel_invalid_size():
    """Test kernel creation with invalid dimensions."""
    with pytest.raises(ValueError):
        _get_kernel((0, 3))

# Erosion and dilation

def test_erode_image():
    """Test that erosion reduces foreground regions."""
    image = np.zeros((20, 20), dtype=np.uint8)
    image[5:15, 5:15] = 255

    result = erode_image(image)

    assert result.shape == image.shape
    assert np.count_nonzero(result) < np.count_nonzero(image)


def test_dilate_image():
    """Test that dilation expands foreground regions."""
    image = np.zeros((20, 20), dtype=np.uint8)
    image[8:12, 8:12] = 255

    result = dilate_image(image)

    assert result.shape == image.shape
    assert np.count_nonzero(result) > np.count_nonzero(image)


def test_erode_image_invalid_iterations():
    """Test erosion with an invalid number of iterations."""
    image = np.zeros((20, 20), dtype=np.uint8)

    with pytest.raises(ValueError):
        erode_image(image, iterations=0)


def test_dilate_image_invalid_iterations():
    """Test dilation with an invalid number of iterations."""
    image = np.zeros((20, 20), dtype=np.uint8)

    with pytest.raises(ValueError):
        dilate_image(image, iterations=0)

# Morphological opening and closing

def test_morphological_opening():
    """Test that opening removes small foreground noise."""
    image = np.zeros((20, 20), dtype=np.uint8)
    image[5:15, 5:15] = 255
    image[1, 1] = 255  # Small isolated noise

    result = morphological_opening(image, kernel_size=(3, 3))

    assert result.shape == image.shape
    assert result[1, 1] == 0
    assert np.count_nonzero(result) < np.count_nonzero(image)


def test_morphological_closing():
    """Test that closing fills small holes in foreground regions."""
    image = np.zeros((20, 20), dtype=np.uint8)
    image[5:15, 5:15] = 255
    image[9, 9] = 0  # Small hole

    result = morphological_closing(image, kernel_size=(3, 3))

    assert result.shape == image.shape
    assert result[9, 9] == 255
    assert np.count_nonzero(result) > np.count_nonzero(image)
