from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from preprocessing import preprocess


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

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    originals = artifacts["original"]
    equalized_gray = artifacts["normalized"]

    titles = ["Resized RGB", "Equalized (gray)", "Grayscale blurred"]

    visuals = (
        originals,
        equalized_gray,
        blurred,
    )
    cmap = [None, "gray", "gray"]

    for ax, viz, cmap_name, title in zip(axes, visuals, cmap, titles):
        if cmap_name:
            ax.imshow(viz, cmap=cmap_name, vmin=viz.min(), vmax=viz.max())
        else:
            ax.imshow(viz.astype("uint8"))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
