from __future__ import annotations

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase2.segmentation import get_largest_contour, segment_garment


WIDTH_SLICES = [0.10, 0.25, 0.50, 0.75, 0.90]


def get_row_width(mask: np.ndarray, row_y: int) -> float:
    """Count how wide the garment is on one horizontal row."""
    xs = np.where(mask[row_y, :] > 0)[0]

    if len(xs) == 0:
        return 0.0

    return float(xs.max() - xs.min() + 1)


def get_width_profile(mask: np.ndarray, x: int, y: int, w: int, h: int) -> list[float]:
    """Measure garment width."""
    widths = []

    for fraction in WIDTH_SLICES:
        row_y = y + int(round(fraction * (h - 1)))
        row_y = max(0, min(row_y, mask.shape[0] - 1))
        widths.append(get_row_width(mask, row_y))

    max_width = max(widths)
    if max_width == 0:
        return [0.0] * len(WIDTH_SLICES)

    # Normalize so the widest slice is 1.0
    normalized = [width / max_width for width in widths]
    return normalized


def get_edge_density_in_half(
    mask: np.ndarray,
    edges: np.ndarray,
    mid_y: int,
    use_upper_half: bool,
) -> float:
    """Count how many edge pixels appear in the upper or lower half."""
    garment_pixels = 0
    edge_pixels = 0

    for row in range(mask.shape[0]):
        in_half = row < mid_y if use_upper_half else row >= mid_y

        for col in range(mask.shape[1]):
            if mask[row, col] == 0:
                continue

            if not in_half:
                continue

            garment_pixels += 1

            if edges[row, col] > 0:
                edge_pixels += 1

    if garment_pixels == 0:
        return 0.0

    return edge_pixels / garment_pixels


def extract_silhouette_features(mask: np.ndarray, edges: np.ndarray) -> dict:
    """Extract simple silhouette features from a garment mask and edge map."""
    mask = (mask > 0).astype(np.uint8) * 255
    edges = (edges > 0).astype(np.uint8) * 255

    contour = get_largest_contour(mask)
    if contour is None:
        return {
            "contour_area_fraction": 0.0,
            "bbox_aspect_ratio": 0.0,
            "solidity": 0.0,
            "width_profile": [0.0] * len(WIDTH_SLICES),
            "width_taper_ratio": 0.0,
            "edge_density_upper": 0.0,
            "edge_density_lower": 0.0,
            "edge_density_ratio": 0.0,
        }

    image_height, image_width = mask.shape
    image_area = image_height * image_width

    # Basic contour shape info
    x, y, w, h = cv2.boundingRect(contour)

    moments = cv2.moments(contour)
    area = moments["m00"]
    contour_area_fraction = area / image_area
    bbox_aspect_ratio = w / h

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Width profile across the garment
    width_profile = get_width_profile(mask, x, y, w, h)
    top_width = width_profile[0]
    bottom_width = width_profile[-1]

    if top_width > 0:
        width_taper_ratio = bottom_width / top_width
    else:
        width_taper_ratio = 0.0

    # Compare edge density in top half vs bottom half
    mid_y = y + h // 2
    edge_density_upper = get_edge_density_in_half(mask, edges, mid_y, use_upper_half=True)
    edge_density_lower = get_edge_density_in_half(mask, edges, mid_y, use_upper_half=False)

    if edge_density_lower > 0:
        edge_density_ratio = edge_density_upper / edge_density_lower
    else:
        edge_density_ratio = edge_density_upper

    return {
        "contour_area_fraction": contour_area_fraction,
        "bbox_aspect_ratio": bbox_aspect_ratio,
        "solidity": solidity,
        "width_profile": width_profile,
        "width_taper_ratio": width_taper_ratio,
        "edge_density_upper": edge_density_upper,
        "edge_density_lower": edge_density_lower,
        "edge_density_ratio": edge_density_ratio,
    }


def visualize_features(mask: np.ndarray, edges: np.ndarray, features: dict) -> None:
    """Show the mask, edges, and width profile."""
    contour = get_largest_contour(mask)
    if contour is None:
        return

    x, y, w, h = cv2.boundingRect(contour)
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Draw the rows where we measured width
    for fraction in WIDTH_SLICES:
        row_y = y + int(round(fraction * (h - 1)))
        row_y = max(0, min(row_y, mask.shape[0] - 1))
        cv2.line(overlay, (0, row_y), (mask.shape[1] - 1, row_y), (0, 255, 0), 1)

    # Draw the split between upper and lower half
    mid_y = y + h // 2
    cv2.line(overlay, (0, mid_y), (mask.shape[1] - 1, mid_y), (255, 0, 0), 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Mask with slice lines")
    axes[0].axis("off")

    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Canny edges")
    axes[1].axis("off")

    axes[2].bar([str(value) for value in WIDTH_SLICES], features["width_profile"])
    axes[2].set_xlabel("Vertical slice")
    axes[2].set_ylabel("Normalized width")
    axes[2].set_title("Width profile")

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: extract silhouette features.")
    parser.add_argument("image_path", help="Path to the clothing image.")
    args = parser.parse_args()

    segmentation = segment_garment(args.image_path)
    features = extract_silhouette_features(segmentation["mask"], segmentation["edges"])

    print("Silhouette features:")
    for key, value in features.items():
        print(f"  {key}: {value}")

    visualize_features(segmentation["mask"], segmentation["edges"], features)


if __name__ == "__main__":
    main()
