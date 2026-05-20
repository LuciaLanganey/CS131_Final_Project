"""Image preprocessing helpers."""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np


def conv(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).

    Adapted from Project 1 Option B.
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0, pad_width0), (pad_width1, pad_width1))
    padded = np.pad(image, pad_width, mode="edge")

    flipped_kernel = np.flip(kernel)
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(flipped_kernel * padded[i : i + Hk, j : j + Wk])

    return out


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).

    Adapted from Project 1 Option B.
    """

    kernel = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            numerator = (i - size // 2) ** 2 + (j - size // 2) ** 2
            denominator = 2 * sigma ** 2
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                -numerator / denominator
            )
    return kernel


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale luminance in [0, 1]."""
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if rgb.dtype == np.uint8:
        lum = lum / 255.0
    else:
        lum = np.clip(lum, 0.0, 1.0)
    return lum


def gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.4,
) -> np.ndarray:
    """
    Apply Gaussian smoothing using conv() with a Gaussian kernel. 
    Adapted from Project 1 Option B.
    """
    if image.ndim == 3:
        gray = _luminance(image)
    else:
        gray = image.astype(np.float64)
        if np.issubdtype(image.dtype, np.integer):
            gray = gray / 255.0

    k = gaussian_kernel(kernel_size, sigma)
    k /= np.sum(k)
    blurred = conv(gray.astype(np.float64), k)
    blurred = np.clip(blurred, 0.0, 1.0)
    return blurred.astype(np.float32)


def normalize_brightness(image: np.ndarray) -> np.ndarray:
    """Global histogram equalization."""
    if image.ndim == 3:
        values = (_luminance(image) * 255.0).round().clip(0, 255).astype(np.uint8)
    else:
        if np.issubdtype(image.dtype, np.integer):
            values = np.clip(image, 0, 255).astype(np.uint8)
        else:
            values = (
                np.clip(image.astype(np.float64), 0.0, 1.0) * 255.0
            ).round().astype(np.uint8)

    pixels = values.ravel()
    hist = np.bincount(pixels.astype(np.int64), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    cdf_scaled = cdf / cdf[-1] if cdf[-1] > 0 else cdf
    equalized_flat = cdf_scaled[pixels.astype(np.int64)]
    shaped = equalized_flat.reshape(values.shape).astype(np.float32)
    return np.clip(shaped, 0.0, 1.0).astype(np.float32)


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Resize an image using area."""
    width, height = target_size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def preprocess(
    image_path: str,
    target_size: Tuple[int, int] = (256, 256),
    equalize: bool = True,
) -> dict[str, np.ndarray]:
    """Run preprocessing on an image path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from path: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_rgb = resize_image(rgb, target_size)

    resized_lum01 = _luminance(resized_rgb).astype(np.float32)

    if equalize:
        normalized_gray = normalize_brightness(resized_rgb)
    else:
        normalized_gray = np.clip(resized_lum01, 0.0, 1.0).astype(np.float32)

    gray_blurred = gaussian_blur(normalized_gray, kernel_size=5, sigma=1.4)

    return {
        "original": resized_rgb.astype(np.uint8),
        "gray": gray_blurred,
        "normalized": normalized_gray,
    }


if __name__ == "__main__":
    demo_path = "test_images/sample.jpg"
    artifacts = preprocess(demo_path)
    for key, arr in artifacts.items():
        print(f"{key}: shape={tuple(arr.shape)} dtype={arr.dtype}")
