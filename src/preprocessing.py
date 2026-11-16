
import cv2
import numpy as np
from pathlib import Path

def load_image(image_path: str, as_gray: bool = False) -> np.ndarray:
    """Load an image from disk in RGB or grayscale format."""

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image at: {image_path}")
        
    if as_gray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def gaussian_blur(image: np.ndarray,kernel_size: tuple = (5, 5)) -> np.ndarray:
    """Smooth an image using a Gaussian filter."""
    if image is None:
        raise ValueError("Input image cannot be None.")

    if (
        len(kernel_size) != 2
        or any(not isinstance(k, int) or k <= 0 or k % 2 == 0 for k in kernel_size)
    ):
        raise ValueError("Kernel dimensions must be positive odd integers.")

    return cv2.GaussianBlur(image, kernel_size, 0)


def apply_threshold(image: np.ndarray,threshold_value: int = 127,max_value: int = 255) -> np.ndarray:
    """Convert a grayscale image into a binary image."""
    if image is None or image.ndim != 2:
        raise ValueError("Input must be a grayscale image.")

    if not (0 <= threshold_value <= 255 and 0 <= max_value <= 255):
        raise ValueError("Threshold and max values must be between 0 and 255.")

    _, result = cv2.threshold( image, threshold_value, max_value, cv2.THRESH_BINARY)
    return result


def detect_edges(image: np.ndarray,threshold1: int = 50,threshold2: int = 150) -> np.ndarray:
    """Detect edges in a grayscale image using Canny."""
    if image is None or image.ndim != 2:
        raise ValueError("Input must be a grayscale image.")

    if threshold1 < 0 or threshold2 < threshold1:
        raise ValueError("Invalid Canny thresholds.")

    return cv2.Canny(image, threshold1, threshold2)


def _get_kernel(kernel_size: tuple) -> np.ndarray:
    """Create a rectangular structuring element."""
    if (
        len(kernel_size) != 2
        or any(not isinstance(k, int) or k <= 0 for k in kernel_size)
    ):
        raise ValueError("Kernel dimensions must be positive integers.")

    return np.ones(kernel_size, dtype=np.uint8)


def erode_image(image: np.ndarray,kernel_size: tuple = (3, 3),iterations: int = 1) -> np.ndarray:
    """Shrink foreground regions in a binary image."""
    if image is None:
        raise ValueError("Input image cannot be None.")
    if iterations < 1:
        raise ValueError("Iterations must be at least 1.")

    kernel = _get_kernel(kernel_size)
    return cv2.erode(image, kernel, iterations=iterations)


def dilate_image(image: np.ndarray,kernel_size: tuple = (3, 3),iterations: int = 1) -> np.ndarray:
    """Expand foreground regions in a binary image."""
    if image is None:
        raise ValueError("Input image cannot be None.")
    if iterations < 1:
        raise ValueError("Iterations must be at least 1.")

    kernel = _get_kernel(kernel_size)
    return cv2.dilate(image, kernel, iterations=iterations)


def morphological_opening(image: np.ndarray,kernel_size: tuple = (3, 3)) -> np.ndarray:
    """Remove small foreground objects using morphological opening."""
    if image is None:
        raise ValueError("Input image cannot be None.")

    kernel = _get_kernel(kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def morphological_closing(image: np.ndarray,kernel_size: tuple = (3, 3)) -> np.ndarray:
    """Fill small gaps in foreground regions using morphological closing."""
    if image is None:
        raise ValueError("Input image cannot be None.")

    kernel = _get_kernel(kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def resize_image(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize an image to (size, size)."""
    if image is None:
        raise ValueError("Input image cannot be None.")
    if size <= 0:
        raise ValueError("Size must be positive.")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def preprocess_image(image: np.ndarray, size: int = 224,blur: bool = False,blur_kernel: tuple = (3, 3)) -> np.ndarray:
    """Single preprocessing pipeline shared by training and inference.

    Input : RGB uint8 image (H, W, 3)
    Output: RGB uint8 image (size, size, 3)
    """
    if image is None or image.ndim != 3:
        raise ValueError("Input must be an RGB image with shape (H, W, 3).")

    if blur:
        image = gaussian_blur(image, blur_kernel)
    return resize_image(image, size)
