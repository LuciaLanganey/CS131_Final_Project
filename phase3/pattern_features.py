from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from skimage.feature import hog
except ImportError as exc:
    raise ImportError(
        "This file needs scikit-image for HOG features. "
        "Install it with: pip install scikit-image"
    ) from exc



# Make imports work from the main project folder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase2.segmentation import segment_garment, to_uint8



# Basic helpers

def get_mask_bbox(mask: np.ndarray, padding: int = 8) -> Tuple[int, int, int, int]:
    """
    Get the bounding box around the white garment region in the mask.

    Returns:
        x1, y1, x2, y2

    If the mask is empty, we return the whole image.
    """
    ys, xs = np.where(mask > 0)

    height, width = mask.shape

    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, width, height

    x1 = max(xs.min() - padding, 0)
    y1 = max(ys.min() - padding, 0)
    x2 = min(xs.max() + padding + 1, width)
    y2 = min(ys.max() + padding + 1, height)

    return x1, y1, x2, y2


def crop_to_garment(
    rgb: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
    padding: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop RGB image, grayscale image, and mask to the garment bounding box.
    """
    x1, y1, x2, y2 = get_mask_bbox(mask, padding=padding)

    rgb_crop = rgb[y1:y2, x1:x2]
    gray_crop = gray[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    return rgb_crop, gray_crop, mask_crop


def resize_for_features(
    rgb_crop: np.ndarray,
    gray_crop: np.ndarray,
    mask_crop: np.ndarray,
    size: Tuple[int, int] = (128, 128),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resize crop and mask to fixed size so HOG/FFT features are comparable.
    """
    width, height = size

    rgb_resized = cv2.resize(rgb_crop, (width, height), interpolation=cv2.INTER_AREA)
    gray_resized = cv2.resize(gray_crop, (width, height), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask_crop, (width, height), interpolation=cv2.INTER_NEAREST)

    return rgb_resized, gray_resized, mask_resized


def apply_gray_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply the binary mask to a grayscale image.

    Instead of setting outside-mask pixels to black, set them to the mean
    garment intensity. This avoids creating artificial hard edges that
    dominate FFT/Gabor/HOG features.
    """
    gray_float = gray.astype(np.float32)

    if gray_float.max() > 1.0:
        gray_float = gray_float / 255.0

    mask_bool = mask > 0

    if np.sum(mask_bool) == 0:
        return gray_float

    garment_mean = float(np.mean(gray_float[mask_bool]))

    masked = np.full_like(gray_float, garment_mean)
    masked[mask_bool] = gray_float[mask_bool]

    return masked
    
    
def erode_mask(mask: np.ndarray, erosion_size: int = 9) -> np.ndarray:
    """
    Shrink the garment mask so texture features are computed inside the garment,
    away from the silhouette boundary.

    This prevents FFT, Gabor, and HOG from being dominated by the garment outline.
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)

    # If erosion removes too much, fall back to original mask.
    if np.mean(eroded > 0) < 0.05:
        return mask_uint8

    return eroded



# Feature group 1: HSV color variance

def hsv_variance_features(rgb: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Extract HSV color variance features from garment pixels only.

    Patterned clothing tends to have higher variance.
    Solid clothing tends to have lower variance.
    """
    rgb_uint8 = rgb.astype(np.uint8)
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)

    garment_pixels = hsv[mask > 0]

    if garment_pixels.size == 0:
        return {
            "h_mean": 0.0,
            "s_mean": 0.0,
            "v_mean": 0.0,
            "h_var": 0.0,
            "s_var": 0.0,
            "v_var": 0.0,
            "hsv_total_var": 0.0,
        }

    h = garment_pixels[:, 0].astype(np.float32)
    s = garment_pixels[:, 1].astype(np.float32)
    v = garment_pixels[:, 2].astype(np.float32)

    h_var = float(np.var(h))
    s_var = float(np.var(s))
    v_var = float(np.var(v))

    return {
        "h_mean": float(np.mean(h)),
        "s_mean": float(np.mean(s)),
        "v_mean": float(np.mean(v)),
        "h_var": h_var,
        "s_var": s_var,
        "v_var": v_var,
        "hsv_total_var": h_var + s_var + v_var,
    }



# Feature group 2: Gabor filter features

def gabor_features(gray: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Apply Gabor filters at several orientations.

    Striped or directional patterns usually create stronger responses
    at specific orientations.
    """
    gray_masked = apply_gray_mask(gray, mask)
    
    garment_values = gray_masked[mask > 0]
    contrast = float(np.std(garment_values)) + 1e-6

    orientations = {
        "0": 0,
        "45": np.pi / 4,
        "90": np.pi / 2,
        "135": 3 * np.pi / 4,
    }

    features: Dict[str, float] = {}
    energies = []

    for name, theta in orientations.items():
        kernel = cv2.getGaborKernel(
            ksize=(21, 21),
            sigma=4.0,
            theta=theta,
            lambd=10.0,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F,
        )

        response = cv2.filter2D(gray_masked, cv2.CV_32F, kernel)

        garment_response = response[mask > 0]

        if garment_response.size == 0:
            mean_abs = 0.0
            energy = 0.0
            std = 0.0
        else:
            mean_abs = float(np.mean(np.abs(garment_response)) / contrast)
            energy = float(np.mean(garment_response ** 2) / (contrast ** 2))
            std = float(np.std(garment_response))

        features[f"gabor_{name}_mean_abs"] = mean_abs
        features[f"gabor_{name}_energy"] = energy
        features[f"gabor_{name}_std"] = std

        energies.append(energy)

    energies_array = np.array(energies, dtype=np.float32)

    max_energy = float(np.max(energies_array))
    mean_energy = float(np.mean(energies_array))
    min_energy = float(np.min(energies_array))

    # If one orientation dominates, that is a good sign of directional texture.
    orientation_contrast = max_energy / (mean_energy + 1e-6)

    features["gabor_max_energy"] = max_energy
    features["gabor_mean_energy"] = mean_energy
    features["gabor_min_energy"] = min_energy
    features["gabor_orientation_contrast"] = float(orientation_contrast)

    return features



# Feature group 3: HOG features

def hog_features(gray: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Extract summary statistics from HOG.

    HOG captures local gradient directions, which are useful for stripes,
    dots, checks, and other repeated edge structures.
    """
    gray_masked = apply_gray_mask(gray, mask)

    # HOG expects a 2D image. We use a fixed 128x128 garment crop.
    hog_vector = hog(
        gray_masked,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    if hog_vector.size == 0:
        return {
            "hog_mean": 0.0,
            "hog_std": 0.0,
            "hog_max": 0.0,
            "hog_energy": 0.0,
            "hog_nonzero_fraction": 0.0,
        }

    return {
        "hog_mean": float(np.mean(hog_vector)),
        "hog_std": float(np.std(hog_vector)),
        "hog_max": float(np.max(hog_vector)),
        "hog_energy": float(np.mean(hog_vector ** 2)),
        "hog_nonzero_fraction": float(np.mean(hog_vector > 1e-4)),
    }



# Feature group 4: FFT frequency features

def fft_features(gray: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Extract frequency-domain features using FFT.

    Periodic patterns like stripes and checks often produce strong frequency
    peaks away from the center of the Fourier spectrum.
    """
    gray_masked = apply_gray_mask(gray, mask)

    # Remove mean from garment pixels to reduce the DC component.
    garment_values = gray_masked[mask > 0]

    if garment_values.size == 0:
        return {
            "fft_high_freq_ratio": 0.0,
            "fft_peak_ratio": 0.0,
            "fft_energy": 0.0,
            "fft_frequency_concentration": 0.0,
        }

    centered = gray_masked.copy()
    centered[mask > 0] = centered[mask > 0] - np.mean(garment_values)

    # Windowing reduces edge artifacts from the crop boundary.
    h, w = centered.shape
    window_y = np.hanning(h)
    window_x = np.hanning(w)
    window = np.outer(window_y, window_x)

    windowed = centered * window

    fft = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)

    total_energy = float(np.sum(magnitude ** 2))

    if total_energy == 0:
        return {
            "fft_high_freq_ratio": 0.0,
            "fft_peak_ratio": 0.0,
            "fft_energy": 0.0,
            "fft_frequency_concentration": 0.0,
        }

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    low_freq = radius < 8
    mid_freq = (radius >= 8) & (radius <= 45)
    high_freq = radius > 45
    
    mid_freq_energy = float(np.sum(magnitude[mid_freq] ** 2))
    mid_freq_ratio = mid_freq_energy / total_energy

    high_freq_energy = float(np.sum(magnitude[high_freq] ** 2))
    high_freq_ratio = high_freq_energy / total_energy

    # Ignore center/DC region when finding periodic peaks.
    # For pattern detection, mid frequencies are usually more useful than
    # very low shape frequencies or extreme high-frequency noise.
    magnitude_mid = np.zeros_like(magnitude)
    magnitude_mid[mid_freq] = magnitude[mid_freq]

    mid_peak = float(np.max(magnitude_mid))
    mean_mid = float(np.mean(magnitude[mid_freq]))
    mid_peak_ratio = mid_peak / (mean_mid + 1e-6)

    # Keep a non-center version for frequency concentration.
    magnitude_no_center = magnitude.copy()
    magnitude_no_center[low_freq] = 0

    # Concentration is high if a small number of frequencies dominate.
    sorted_mag = np.sort(magnitude_no_center.ravel())[::-1]
    top_k = max(1, int(0.01 * sorted_mag.size))
    top_energy = float(np.sum(sorted_mag[:top_k] ** 2))
    frequency_concentration = top_energy / total_energy

    return {
        "fft_high_freq_ratio": float(high_freq_ratio),
        "fft_mid_freq_ratio": float(mid_freq_ratio),
        "fft_mid_peak_ratio": float(mid_peak_ratio),
        "fft_energy": total_energy,
        "fft_frequency_concentration": float(frequency_concentration),
    }



# Full feature extraction

def extract_pattern_features(
    image_path: str,
    segmentation_method: str = "hybrid",
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Extract all Partner B pattern/texture features for one image.

    Returns:
        features:
            Dictionary of numeric features.

        artifacts:
            Images useful for visualization/debugging.
    """
    segmentation_results = segment_garment(
        image_path=image_path,
        method=segmentation_method,
    )

    rgb = segmentation_results["rgb"]
    gray = segmentation_results["gray"]
    mask = segmentation_results["mask"]

    # First crop to the garment bounding box.
    rgb_crop, gray_crop, mask_crop = crop_to_garment(
        rgb,
        gray,
        mask,
        padding=8,
    )

    # Then resize the crop to a fixed size.
    rgb_crop, gray_crop, mask_crop = resize_for_features(
        rgb_crop,
        gray_crop,
        mask_crop,
        size=(128, 128),
    )

    # Use a slightly eroded mask for texture features so Gabor/HOG/FFT
    # are not dominated by the garment silhouette boundary.
    texture_mask = erode_mask(mask_crop, erosion_size=9)

    features: Dict[str, float] = {}

    features.update(hsv_variance_features(rgb_crop, texture_mask))
    features.update(gabor_features(gray_crop, texture_mask))
    features.update(hog_features(gray_crop, texture_mask))
    features.update(fft_features(gray_crop, texture_mask))

    # Added a few basic mask features. These help debug whether a
    # feature vector came from bad segmentation.
    features["mask_area_fraction"] = float(np.mean(mask_crop > 0))
    features["segmentation_method_hybrid_used"] = 1.0 if segmentation_method == "hybrid" else 0.0

    artifacts = {
        "rgb": rgb,
        "gray": gray,
        "mask": mask,
        "rgb_crop": rgb_crop,
        "gray_crop": gray_crop,
        "mask_crop": mask_crop,
        "texture_mask": texture_mask,
        "masked_gray_crop": apply_gray_mask(gray_crop, texture_mask),
    }

    return features, artifacts



# Saving and visualization

def save_features_csv(
    features: Dict[str, float],
    image_path: str,
    output_csv: str,
) -> None:
    """
    Append one row of features to a CSV file.

    If the existing CSV has a different header from the current feature set,
    stop and ask the user to regenerate the CSV. This prevents misaligned rows.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    row = {"image_path": image_path}
    row.update(features)

    current_fieldnames = list(row.keys())

    if os.path.exists(output_csv):
        with open(output_csv, mode="r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)

        if existing_header != current_fieldnames:
            raise ValueError(
                "The existing CSV header does not match the current feature set. "
                "Delete the old CSV and regenerate it:\n"
                f"rm {output_csv}"
            )

    file_exists = os.path.exists(output_csv)

    with open(output_csv, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=current_fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(f"Saved features to: {output_csv}")


def visualize_pattern_artifacts(artifacts: Dict[str, np.ndarray]) -> None:
    """
    Visualize the garment crop and mask used for feature extraction.
    """
    titles = [
        "RGB image",
        "Phase 2 mask",
        "Garment crop",
        "Crop mask",
        "Texture mask",
        "Masked grayscale crop",
    ]

    images = [
        artifacts["rgb"],
        artifacts["mask"],
        artifacts["rgb_crop"],
        artifacts["mask_crop"],
        artifacts["texture_mask"],
        artifacts["masked_gray_crop"],
    ]

    cmaps = [
        None,
        "gray",
        None,
        "gray",
        "gray",
        "gray",
    ]

    fig, axes = plt.subplots(1, 6, figsize=(18, 4))

    for ax, image, title, cmap in zip(axes, images, titles, cmaps):
        if cmap == "gray":
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image.astype(np.uint8))

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()



# Command-line interface

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 Partner B: extract pattern/texture features.",
    )

    parser.add_argument(
        "image_path",
        help="Path to the clothing image.",
    )

    parser.add_argument(
        "--method",
        default="hybrid",
        choices=["hybrid", "edge_fill", "kmeans", "grabcut", "threshold"],
        help="Segmentation method from Phase 2.",
    )

    parser.add_argument(
        "--save_csv",
        default=None,
        help="Optional path to save/append features as CSV.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show visualization of the crop/mask used for features.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    features, artifacts = extract_pattern_features(
        image_path=args.image_path,
        segmentation_method=args.method,
    )

    print("\nExtracted pattern/texture features:")
    for name, value in features.items():
        print(f"  {name}: {value:.6f}")

    if args.save_csv is not None:
        save_features_csv(
            features=features,
            image_path=args.image_path,
            output_csv=args.save_csv,
        )

    if args.show:
        visualize_pattern_artifacts(artifacts)


if __name__ == "__main__":
    main()
