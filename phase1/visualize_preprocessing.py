from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from preprocessing import conv, gaussian_kernel, preprocess


def partial_x(img: np.ndarray) -> np.ndarray:
    """Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.

    Adapted from Project 1 Option B.
    """

    kernel = np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]], dtype=np.float64)
    out = conv(np.asarray(img, dtype=np.float64), kernel)

    return out


def partial_y(img: np.ndarray) -> np.ndarray:
    """Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    
    Adapted from Project 1 Option B.
    """

    kernel = np.array([[0, 0.5, 0], [0, 0, 0], [0, -0.5, 0]], dtype=np.float64)
    out = conv(np.asarray(img, dtype=np.float64), kernel)

    return out


def gradient(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    
    Adapted from Project 1 Option B.
    """
    G = np.zeros(img.shape, dtype=np.float64)
    theta = np.zeros(img.shape, dtype=np.float64)

    p_x = partial_x(img)
    p_y = partial_y(img)

    G = np.sqrt(p_x**2 + p_y**2)
    theta = np.degrees(np.arctan2(p_y, p_x)) % 360.0

    return G, theta


def non_maximum_suppression(G: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.

    Adapted from Project 1 Option B.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    theta = (theta % 360.0).astype(np.int32)

    padded_G = np.pad(G, 1, mode="constant", constant_values=0)

    for i in range(H):
        for j in range(W):
            angle = theta[i, j]
            pi = i + 1
            pj = j + 1

            # Horizontal
            if angle == 0 or angle == 180:
                p1 = padded_G[pi, pj + 1]
                p2 = padded_G[pi, pj - 1]

            # 45 Degrees
            elif angle == 45 or angle == 225:
                p1 = padded_G[pi - 1, pj - 1]
                p2 = padded_G[pi + 1, pj + 1]

            # Vertical
            elif angle == 90 or angle == 270:
                p1 = padded_G[pi - 1, pj]
                p2 = padded_G[pi + 1, pj]

            # 135 Degrees
            elif angle == 135 or angle == 315:
                p1 = padded_G[pi - 1, pj + 1]
                p2 = padded_G[pi + 1, pj - 1]

            else:
                p1 = 0
                p2 = 0

            if (G[i, j] >= p1) and (G[i, j] >= p2):
                out[i, j] = G[i, j]

    return out


def double_thresholding(
    img: np.ndarray,
    high: float,
    low: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.

    Adapted from Project 1 Option B.
    """

    strong_edges = img > high
    weak_edges = (img > low) & (img <= high)

    return strong_edges, weak_edges


def get_neighbors(y: int, x: int, H: int, W: int) -> list[tuple[int, int]]:
    """Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].

    Adapted from Project 1 Option B.
    """
    neighbors: list[tuple[int, int]] = []

    for i in (y - 1, y, y + 1):
        for j in (x - 1, x, x + 1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if i == y and j == x:
                    continue
                neighbors.append((i, j))

    return neighbors


def link_edges(strong_edges: np.ndarray, weak_edges: np.ndarray) -> np.ndarray:
    """Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).

    Returns:
        edges: numpy boolean array of shape(H, W).

    Adapted from Project 1 Option B.
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges).astype(bool)
    edges = np.copy(strong_edges).astype(bool)

    queue: list[tuple[int, ...]] = list(map(tuple, indices))
    visited = set(queue)

    while queue:
        y, x = queue.pop(0)
        for i, j in get_neighbors(y, x, H, W):
            if (i, j) not in visited and weak_edges[i, j]:
                edges[i, j] = True
                weak_edges[i, j] = False
                visited.add((i, j))
                queue.append((i, j))

    return edges


def canny(
    img: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.4,
    high: float = 20.0,
    low: float = 15.0,
) -> np.ndarray:
    """Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).

    Adapted from Project 1 Option B.
    """

    working = img.astype(np.float64)
    if np.max(working) <= 1.0 + 1e-6 and working.dtype.kind == "f":
        working = working * 255.0

    kernel = gaussian_kernel(kernel_size, sigma)
    conv_img = conv(working, kernel)
    G, theta = gradient(conv_img)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)

    return edge.astype(np.uint8)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize preprocessing stages for a clothing image.",
    )
    parser.add_argument("image_path", help="Filesystem path to a clothing photo.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    artifacts = preprocess(args.image_path, target_size=(256, 256), equalize=True)
    blurred = artifacts["gray"]
    blurred_01 = np.clip(blurred.astype(np.float32), 0.0, 1.0)
    edges = canny(blurred_01, kernel_size=5, sigma=1.4, high=20.0, low=15.0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    originals = artifacts["original"]
    equalized_gray = artifacts["normalized"]

    titles = ["Resized RGB", "Equalized (gray)", "Grayscale blurred", "Canny preview"]

    visuals = (
        originals,
        equalized_gray,
        blurred,
        edges,
    )
    cmap = [None, "gray", "gray", "gray"]

    for ax, viz, cmap_name, title in zip(axes, visuals, cmap, titles):
        if cmap_name:
            ax.imshow(viz, cmap=cmap_name, vmin=viz.min(), vmax=viz.max())
        else:
            ax.imshow(viz.astype(np.uint8))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
