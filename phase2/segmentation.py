from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np



# Import Phase 1 preprocessing

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase1.preprocessing import preprocess



# Helper 1: Convert grayscale image to uint8

def to_uint8(gray: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale image to uint8 format.

    Phase 1 code returns grayscale images as floats in [0, 1].
    OpenCV functions like thresholding and Canny expect uint8 values
    in [0, 255], so we convert here.
    """
    if gray.dtype == np.uint8:
        return gray

    gray = np.clip(gray, 0.0, 1.0)
    gray_uint8 = (gray * 255).astype(np.uint8)

    return gray_uint8


# Helper 2: Clean binary masks

def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean a binary mask using morphological operations.

    The mask should have:
    - 255 for garment pixels
    - 0 for background pixels

    Opening removes tiny white noise.
    Closing fills small black holes and connects nearby regions.
    """
    mask = mask.astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)

    # Remove small noise.
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fill small holes and connect nearby garment regions.
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    return cleaned



# Helper 3: Find largest contour

def get_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the largest connected contour in the mask.

    We assume the main garment is the largest foreground object.
    If no contours are found, return None.
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour



# Method 1: Thresholding baseline

def threshold_segmentation(gray: np.ndarray) -> np.ndarray:
    """
    Segment the garment using Otsu thresholding.

    This is our simple baseline method.

    Since some garments are dark on light backgrounds and others are
    light on dark backgrounds, we try both:
    - regular binary thresholding
    - inverse binary thresholding

    Then we choose the result whose largest contour looks most reasonable.
    """
    gray_uint8 = to_uint8(gray)

    # Regular Otsu threshold.
    _, mask_binary = cv2.threshold(
        gray_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Inverse Otsu threshold.
    _, mask_inverse = cv2.threshold(
        gray_uint8,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    candidates = [mask_binary, mask_inverse]

    height, width = gray_uint8.shape
    image_area = height * width

    best_mask = None
    best_score = -1

    for candidate in candidates:
        cleaned = clean_mask(candidate)
        contour = get_largest_contour(cleaned)

        if contour is None:
            continue

        contour_area = cv2.contourArea(contour)
        area_fraction = contour_area / image_area

        # The garment should not be tiny and should not cover the whole image.
        # This avoids picking masks that are obviously background.
        if 0.05 <= area_fraction <= 0.90:
            score = contour_area
        else:
            score = 0

        if score > best_score:
            best_score = score
            best_mask = cleaned

    # Fallback if something unnatural happens.
    if best_mask is None:
        best_mask = clean_mask(mask_inverse)

    return best_mask



# Method 2: GrabCut segmentation

def keep_best_garment_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep the foreground component that looks most like the main garment.

    It scores each component by:
    - area: garment should be large
    - centrality: garment should be near the image center
    - vertical extent: garment should take up meaningful height
    - width: garment should take up meaningful width
    """
    height, width = mask.shape
    image_area = height * width

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if num_labels <= 1:
        return mask

    image_center_x = width / 2
    image_center_y = height / 2
    max_center_distance = np.sqrt(image_center_x ** 2 + image_center_y ** 2)

    best_label = None
    best_score = -1

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        cx, cy = centroids[label]

        area_fraction = area / image_area
        height_fraction = h / height
        width_fraction = w / width

        # Ignore tiny pieces.
        if area_fraction < 0.03:
            continue

        # Ignore huge background-like regions.
        if area_fraction > 0.90:
            continue

        center_distance = np.sqrt(
            (cx - image_center_x) ** 2 + (cy - image_center_y) ** 2
        )
        centrality = 1.0 - (center_distance / max_center_distance)

        # Garments should usually have meaningful height and width.
        shape_score = height_fraction + width_fraction

        # Weighted score.
        score = (
            3.0 * area_fraction
            + 2.0 * centrality
            + 1.5 * shape_score
        )

        if score > best_score:
            best_score = score
            best_label = label

    output = np.zeros_like(mask)

    if best_label is not None:
        output[labels == best_label] = 255
    else:
        output = mask

    return output


def grabcut_segmentation(rgb: np.ndarray) -> np.ndarray:
    """
    Segment the garment using GrabCut with a centered garment prior.

    This is the best practical method for our project because our dataset
    consists of single-garment clothing images where the garment is the main
    centered object.

    It gives GrabCut a reasonable prior: borders are background, and the central garment region is foreground.
    """
    height, width = rgb.shape[:2]

    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Start with everything as probable background.
    mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)

    # The image border is very likely background.
    border = int(0.04 * min(height, width))
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    # Broad central area: probable foreground.
    # This captures the shirt/body even when it is similar to the background.
    pr_fg_top = int(0.12 * height)
    pr_fg_bottom = int(0.92 * height)
    pr_fg_left = int(0.18 * width)
    pr_fg_right = int(0.82 * width)
    mask[pr_fg_top:pr_fg_bottom, pr_fg_left:pr_fg_right] = cv2.GC_PR_FGD

    # Core torso area: sure foreground.
    # This is the strongest cue that the centered garment should be kept.
    fg_top = int(0.28 * height)
    fg_bottom = int(0.76 * height)
    fg_left = int(0.34 * width)
    fg_right = int(0.66 * width)
    mask[fg_top:fg_bottom, fg_left:fg_right] = cv2.GC_FGD

    # Keep the border as definite background.
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        bgr,
        mask,
        None,
        background_model,
        foreground_model,
        8,
        cv2.GC_INIT_WITH_MASK,
    )

    foreground_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    foreground_mask = clean_mask(foreground_mask)
    foreground_mask = keep_best_garment_component(foreground_mask)

    # Final cleanup after component selection.
    foreground_mask = clean_mask(foreground_mask)

    return foreground_mask



# Canny edge detection

def canny_edges(gray: np.ndarray) -> np.ndarray:
    """
    Compute Canny edges from the blurred grayscale image.

    This is not the segmentation mask itself.
    It is an edge map that helps us visualize garment boundaries
    and later compute edge-based features.
    """
    gray_uint8 = to_uint8(gray)

    edges = cv2.Canny(
        gray_uint8,
        threshold1=50,
        threshold2=150,
    )

    return edges



# Draw contour

def draw_largest_contour(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Draw the largest garment contour on the RGB image.
    """
    contour_overlay = rgb.copy()
    contour = get_largest_contour(mask)

    if contour is not None:
        # Red contour in RGB format.
        cv2.drawContours(contour_overlay, [contour], -1, (255, 0, 0), 2)

    return contour_overlay



# Apply mask

def apply_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Use the mask to keep only the garment.

    Background pixels become black.
    """
    segmented = np.zeros_like(rgb)
    garment_pixels = mask > 0
    segmented[garment_pixels] = rgb[garment_pixels]

    return segmented


def remove_long_thin_components(mask: np.ndarray) -> np.ndarray:
    """
    Remove long, thin components from a binary mask.

    This helps remove artifacts like clothing racks, table edges, or
    horizontal background lines without assuming a fixed image location.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    output = np.zeros_like(mask)

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        if h == 0:
            continue

        aspect_ratio = w / h
        fill_ratio = area / (w * h)

        # Remove line-like artifacts:
        # very wide, very short, and sparse/thin.
        is_long_thin = aspect_ratio > 5.0 and h < 0.12 * mask.shape[0]

        if not is_long_thin:
            output[labels == label] = 255

    return output


def remove_horizontal_edge_lines(edges: np.ndarray) -> np.ndarray:
    """
    Remove long horizontal edge lines before contour filling.

    This targets rack/table/background lines using shape, not fixed position.
    """
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))

    horizontal_lines = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1,
    )

    cleaned_edges = cv2.subtract(edges, horizontal_lines)

    return cleaned_edges


def remove_thin_top_protrusion(mask: np.ndarray) -> np.ndarray:
    """
    Remove thin protrusions above the main garment body.

    This helps with artifacts like hanger hooks, small stems, or rack pieces
    that get attached to the garment mask.

    It looks at the mask's width profile row by row and removes rows above the shoulder/body region if they are much thinner than the garment.
    """
    cleaned = mask.copy()

    rows = np.where(cleaned > 0)[0]
    if len(rows) == 0:
        return cleaned

    top = rows.min()
    bottom = rows.max()

    widths = []
    row_indices = []

    for y in range(top, bottom + 1):
        xs = np.where(cleaned[y, :] > 0)[0]
        if len(xs) > 0:
            width = xs.max() - xs.min() + 1
        else:
            width = 0

        widths.append(width)
        row_indices.append(y)

    widths = np.array(widths)
    row_indices = np.array(row_indices)

    max_width = widths.max()

    if max_width == 0:
        return cleaned

    # A row becomes garment-like once it reaches a meaningful fraction
    # of the full garment width. This usually corresponds to shoulder/body area.
    shoulder_threshold = 0.35 * max_width

    shoulder_candidates = row_indices[widths >= shoulder_threshold]

    if len(shoulder_candidates) == 0:
        return cleaned

    shoulder_y = shoulder_candidates.min()

    # Remove only thin material above the shoulder/body region.
    # This removes hanger-like protrusions while keeping the garment body.
    for y, width in zip(row_indices, widths):
        if y < shoulder_y and width < shoulder_threshold:
            cleaned[y, :] = 0

    return cleaned
    
    
def smooth_mask_contour(mask: np.ndarray) -> np.ndarray:
    """
    Smooth the final garment mask.

    This makes the silhouette cleaner by applying a small morphological
    closing and opening.
    """
    kernel_close = np.ones((5, 5), np.uint8)
    kernel_open = np.ones((3, 3), np.uint8)

    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    return smoothed


def edge_fill_segmentation(gray: np.ndarray) -> np.ndarray:
    """
    Segment the garment by using Canny edges, removing obvious line artifacts,
    closing gaps, and filling the best garment-like contour.

    This works well for our project because silhouette information is more
    important than perfect pixel-level segmentation.
    """
    gray_uint8 = to_uint8(gray)

    # Step 1: Find edges.
    edges = cv2.Canny(gray_uint8, threshold1=35, threshold2=120)

    # Step 2: Remove long straight horizontal artifacts such as racks or table edges.
    edges = remove_horizontal_edge_lines(edges)

    # Step 3: Thicken edges so broken garment boundaries connect.
    dilate_kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, dilate_kernel, iterations=1)

    # Step 4: Close gaps in the outline.
    close_kernel = np.ones((9, 9), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # Step 5: Find contours from the closed edge map.
    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = gray_uint8.shape
    image_area = height * width

    best_contour = None
    best_score = -1

    image_center_x = width / 2
    image_center_y = height / 2
    max_center_distance = np.sqrt(image_center_x ** 2 + image_center_y ** 2)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area <= 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        area_fraction = area / image_area
        height_fraction = h / height
        width_fraction = w / width

        # Ignore tiny fragments.
        if area_fraction < 0.02:
            continue

        # Ignore huge background-like shapes.
        if area_fraction > 0.90:
            continue

        # Ignore very long, thin objects.
        aspect_ratio = w / max(h, 1)
        if aspect_ratio > 5.0 and h < 0.15 * height:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx = x + w / 2
            cy = y + h / 2

        center_distance = np.sqrt(
            (cx - image_center_x) ** 2 + (cy - image_center_y) ** 2
        )
        centrality = 1.0 - (center_distance / max_center_distance)

        score = (
            3.0 * area_fraction
            + 2.0 * centrality
            + 1.5 * height_fraction
            + 1.0 * width_fraction
        )

        if score > best_score:
            best_score = score
            best_contour = contour

    mask = np.zeros_like(gray_uint8)

    if best_contour is not None:
        cv2.drawContours(mask, [best_contour], -1, 255, thickness=cv2.FILLED)

    # Final cleanup.
    mask = clean_mask(mask)

    # Remove rack/table/background line artifacts.
    mask = remove_long_thin_components(mask)

    # Keep the most garment-like component.
    mask = keep_best_garment_component(mask)

    # Remove narrow top protrusions, like hanger stems or small attached artifacts.
    mask = remove_thin_top_protrusion(mask)

    # Smooth the final mask.
    mask = smooth_mask_contour(mask)

    return mask


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Get the boundary pixels of a binary mask.

    We use this to check whether the proposed mask boundary agrees
    with the Canny edge map.
    """
    mask = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)

    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = cv2.subtract(mask, eroded)

    return boundary


def edge_agreement_score(mask: np.ndarray, edges: np.ndarray) -> float:
    """
    Score how well the mask boundary lines up with Canny edges.

    A good segmentation mask should have a boundary that overlaps with
    strong image edges.
    """
    boundary = mask_boundary(mask)

    if np.sum(boundary > 0) == 0:
        return 0.0

    # Dilate edges slightly so near-matches still count.
    edge_kernel = np.ones((5, 5), np.uint8)
    thick_edges = cv2.dilate(edges, edge_kernel, iterations=1)

    overlap = np.logical_and(boundary > 0, thick_edges > 0)
    agreement = np.sum(overlap) / np.sum(boundary > 0)

    return float(agreement)


def count_border_contact(mask: np.ndarray) -> float:
    """
    Measure how much of the mask touches the image border.

    A garment can touch the border sometimes, but if a mask touches too much
    border, it is often background leakage.
    """
    h, w = mask.shape

    border_pixels = (
        np.sum(mask[0, :] > 0)
        + np.sum(mask[-1, :] > 0)
        + np.sum(mask[:, 0] > 0)
        + np.sum(mask[:, -1] > 0)
    )

    total_border = 2 * h + 2 * w
    return border_pixels / total_border


def score_mask(mask: np.ndarray, edges: np.ndarray) -> float:
    """
    Score a candidate garment mask.

    A good garment mask should:
    - have reasonable area
    - be near the center
    - have meaningful height and width
    - not touch the image border too much
    - have a boundary that agrees with Canny edges
    - not look like an oversized rectangular/blob background region
    """
    mask = (mask > 0).astype(np.uint8) * 255

    contour = get_largest_contour(mask)

    if contour is None:
        return -1.0

    h_img, w_img = mask.shape
    image_area = h_img * w_img

    area = cv2.contourArea(contour)
    area_fraction = area / image_area

    if area_fraction < 0.025:
        return -1.0

    if area_fraction > 0.85:
        return -1.0

    x, y, w, h = cv2.boundingRect(contour)

    width_fraction = w / w_img
    height_fraction = h / h_img

    if width_fraction < 0.12 or height_fraction < 0.20:
        return -1.0

    moments = cv2.moments(contour)

    if moments["m00"] != 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx = x + w / 2
        cy = y + h / 2

    image_center_x = w_img / 2
    image_center_y = h_img / 2

    max_distance = np.sqrt(image_center_x ** 2 + image_center_y ** 2)
    center_distance = np.sqrt(
        (cx - image_center_x) ** 2 + (cy - image_center_y) ** 2
    )

    centrality = 1.0 - (center_distance / max_distance)

    bbox_area = w * h
    fill_ratio = area / bbox_area if bbox_area > 0 else 0.0

    edge_score = edge_agreement_score(mask, edges)
    border_contact = count_border_contact(mask)

    # Solidity compares contour area to convex hull area.
    # Clothing is usually not perfectly convex, but if a mask is too convex/blob-like,
    # it may be overfilled. This lightly penalizes overly blobby masks.
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Rectangularity compares mask area to bounding box area.
    # Very high rectangularity means the mask may be a big blocky fill.
    rectangularity = fill_ratio

    # A good clothing silhouette should have some boundary complexity.
    # Arc length normalized by image size gives a simple contour-complexity cue.
    perimeter = cv2.arcLength(contour, closed=True)
    normalized_perimeter = perimeter / (2 * (h_img + w_img))

    # Penalize masks that are too boxy/filled.
    boxy_penalty = 0.0
    if rectangularity > 0.78:
        boxy_penalty += rectangularity - 0.78

    if solidity > 0.95:
        boxy_penalty += solidity - 0.95

    # Final weighted score.
    # Edge agreement matters a lot because this project is about silhouette.
    score = (
        2.0 * area_fraction
        + 1.5 * centrality
        + 1.2 * height_fraction
        + 0.8 * width_fraction
        + 5.0 * edge_score
        + 0.8 * normalized_perimeter
        - 3.0 * border_contact
        - 4.0 * boxy_penalty
    )

    return float(score)


def prepare_candidate_mask(mask: np.ndarray) -> np.ndarray:
    """
    Apply conservative cleanup to every candidate mask.

    This makes the comparison fair without over-editing the shape.
    """
    mask = (mask > 0).astype(np.uint8) * 255

    mask = clean_mask(mask)
    mask = remove_long_thin_components(mask)
    mask = keep_best_garment_component(mask)
    mask = smooth_mask_contour(mask)

    return mask


def hybrid_segmentation(rgb: np.ndarray, gray: np.ndarray, verbose: bool = True) -> np.ndarray:
    """
    Try multiple segmentation methods and automatically choose the best mask.

    Since our project focuses on silhouette, edge_fill gets priority when its
    score is close to the best score. This avoids choosing a larger but less
    visually accurate GrabCut mask when edge_fill better follows the clothing
    boundary.
    """
    edges = canny_edges(gray)

    raw_candidates = {
        "threshold": threshold_segmentation(gray),
        "grabcut": grabcut_segmentation(rgb),
        "edge_fill": edge_fill_segmentation(gray),
    }

    prepared_candidates = {}
    scores = {}

    if verbose:
        print("\nHybrid segmentation scores:")

    for name, raw_mask in raw_candidates.items():
        candidate_mask = prepare_candidate_mask(raw_mask)
        candidate_score = score_mask(candidate_mask, edges)

        prepared_candidates[name] = candidate_mask
        scores[name] = candidate_score

        if verbose:
            print(f"  {name}: {candidate_score:.4f}")

    # Standard best method by score.
    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]

    # Silhouette-aware tie-breaker:
    # If edge_fill is close to the best method, prefer edge_fill.
    # This is because our downstream task depends on garment outline.
    edge_fill_score = scores.get("edge_fill", -1.0)

    if edge_fill_score >= best_score - 0.50:
        best_name = "edge_fill"

    best_mask = prepared_candidates[best_name]

    if verbose:
        print(f"Selected method: {best_name}\n")

    # Final cleanup after selection.
    best_mask = keep_best_garment_component(best_mask)
    best_mask = remove_thin_top_protrusion(best_mask)
    best_mask = smooth_mask_contour(best_mask)

    return best_mask


# Main Phase 2 pipeline

def segment_garment(image_path: str, method: str = "hybrid") -> dict[str, np.ndarray]:
    """
    Full Phase 2 pipeline.

    Input:
        image_path:
            Path to the raw clothing image.

        method:
            'grabcut' or 'threshold'.

    Output:
        Dictionary containing:
        - rgb: Phase 1 resized RGB image
        - gray: Phase 1 blurred grayscale image
        - mask: binary garment mask
        - edges: Canny edge map
        - contour_overlay: largest contour drawn on RGB image
        - segmented: image with background removed
    """
    # Use Phase 1 code.
    artifacts = preprocess(
        image_path,
        target_size=(256, 256),
        equalize=True,
    )

    rgb = artifacts["original"]
    gray = artifacts["gray"]

    # Choose segmentation method.
    if method == "hybrid":
        mask = hybrid_segmentation(rgb, gray)
    elif method == "edge_fill":
        mask = edge_fill_segmentation(gray)
    elif method == "grabcut":
        mask = grabcut_segmentation(rgb)
    elif method == "threshold":
        mask = threshold_segmentation(gray)
    else:
        raise ValueError(
            "method must be 'hybrid', 'edge_fill', 'grabcut', or 'threshold'"
        )

    edges = canny_edges(gray)
    contour_overlay = draw_largest_contour(rgb, mask)
    segmented = apply_mask(rgb, mask)

    results = {
        "rgb": rgb,
        "gray": gray,
        "mask": mask,
        "edges": edges,
        "contour_overlay": contour_overlay,
        "segmented": segmented,
    }

    return results



# Display outputs

def visualize_results(results: dict[str, np.ndarray]) -> None:
    """
    Display all major Phase 2 outputs in one figure.
    """
    titles = [
        "Phase 1 RGB",
        "Phase 1 blurred grayscale",
        "Garment mask",
        "Canny edges",
        "Largest contour",
        "Segmented garment",
    ]

    images = [
        results["rgb"],
        results["gray"],
        results["mask"],
        results["edges"],
        results["contour_overlay"],
        results["segmented"],
    ]

    cmaps = [
        None,
        "gray",
        "gray",
        "gray",
        None,
        None,
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for ax, image, title, cmap in zip(axes.ravel(), images, titles, cmaps):
        if cmap == "gray":
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image.astype(np.uint8))

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()



# Save outputs

def save_results(
    results: dict[str, np.ndarray],
    output_dir: str,
    image_path: str,
) -> None:
    """
    Save the Phase 2 outputs to a folder.
    """
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    edges_path = os.path.join(output_dir, f"{base_name}_edges.png")
    contour_path = os.path.join(output_dir, f"{base_name}_contour.png")
    segmented_path = os.path.join(output_dir, f"{base_name}_segmented.png")

    # Mask and edges are grayscale, so OpenCV can save them directly.
    cv2.imwrite(mask_path, results["mask"])
    cv2.imwrite(edges_path, results["edges"])

    # These are RGB, but OpenCV saves in BGR, so convert before saving.
    contour_bgr = cv2.cvtColor(
        results["contour_overlay"].astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )

    segmented_bgr = cv2.cvtColor(
        results["segmented"].astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )

    cv2.imwrite(contour_path, contour_bgr)
    cv2.imwrite(segmented_path, segmented_bgr)

    print("Saved outputs:")
    print(f"  Mask: {mask_path}")
    print(f"  Edges: {edges_path}")
    print(f"  Contour: {contour_path}")
    print(f"  Segmented garment: {segmented_path}")



# Command-line interface

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Phase 2: segment a clothing item from an image.",
    )

    parser.add_argument(
        "image_path",
        help="Path to the clothing image.",
    )

    parser.add_argument(
        "--method",
        choices=["hybrid", "edge_fill", "grabcut", "threshold"],
        default="hybrid",
        help="Segmentation method to use. Default: hybrid.",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output images.",
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/segmentation",
        help="Folder where output images should be saved.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run Phase 2 from the command line.
    """
    args = parse_args()

    results = segment_garment(
        image_path=args.image_path,
        method=args.method,
    )

    visualize_results(results)

    if args.save:
        save_results(
            results=results,
            output_dir=args.output_dir,
            image_path=args.image_path,
        )


if __name__ == "__main__":
    main()
