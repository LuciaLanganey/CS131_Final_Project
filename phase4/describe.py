from __future__ import annotations

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np


# Make imports work from the main project folder.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase4.classifier import extract_features_for_image


LABEL_NAMES = ["pattern", "sleeve", "silhouette", "season", "garment_type"]


def load_models(model_dir="outputs/classification/models/"):
    """
    Load trained classifiers and label encoders from disk.

    Args:
        model_dir: Directory containing saved classifier and encoder .pkl files.

    Returns:
        classifiers: Dictionary mapping label names to trained classifiers.
        encoders: Dictionary mapping label names to fitted LabelEncoders.
    """
    classifiers = {}
    encoders = {}

    for label in LABEL_NAMES:
        clf_path = os.path.join(model_dir, f"{label}_clf.pkl")
        enc_path = os.path.join(model_dir, f"{label}_enc.pkl")

        if not os.path.exists(clf_path):
            raise FileNotFoundError(
                f"Missing classifier file: {clf_path}. "
                "Run python3 phase4/classifier.py first."
            )

        if not os.path.exists(enc_path):
            raise FileNotFoundError(
                f"Missing encoder file: {enc_path}. "
                "Run python3 phase4/classifier.py first."
            )

        classifiers[label] = joblib.load(clf_path)
        encoders[label] = joblib.load(enc_path)

        print(f"Loaded {label} classifier and encoder")

    return classifiers, encoders


def format_sleeve_phrase(sleeve):
    """
    Convert a sleeve label into a natural language phrase.

    Args:
        sleeve: Predicted sleeve label as a string.

    Returns:
        Natural language sleeve phrase.
    """
    sleeve = str(sleeve).strip().lower()

    if sleeve == "sleeveless":
        return "sleeveless"

    if sleeve == "short":
        return "short-sleeved"

    if sleeve == "long":
        return "long-sleeved"

    if sleeve == "spaghetti_strap":
        return "spaghetti-strap"

    if sleeve == "none":
        return ""

    return f"{sleeve}-sleeved"


def describe_image(image_path, classifiers, encoders):
    """
    Predict clothing attributes for one image and create a description.

    Args:
        image_path: Path to the clothing image.
        classifiers: Dictionary mapping label names to trained classifiers.
        encoders: Dictionary mapping label names to fitted LabelEncoders.

    Returns:
        description: Human-readable clothing description.
        predictions: Dictionary of decoded label predictions.
    """
    features = extract_features_for_image(image_path)

    if features is None:
        raise ValueError(f"Could not extract features from image: {image_path}")

    features = np.array(features).reshape(1, -1)

    predictions = {}

    for label in LABEL_NAMES:
        classifier = classifiers[label]
        encoder = encoders[label]

        encoded_prediction = classifier.predict(features)
        decoded_prediction = encoder.inverse_transform(encoded_prediction)[0]

        predictions[label] = decoded_prediction

    pattern = predictions["pattern"]
    silhouette = predictions["silhouette"]
    season = predictions["season"]
    garment_type = predictions["garment_type"]

    garment_type_clean = str(garment_type).strip().lower()

    if garment_type_clean == "pants":
        description = (
            f"A {pattern} {silhouette} pair of pants, "
            f"suited for {season} wear."
        )
    else:
        sleeve = predictions["sleeve"]
        sleeve_phrase = format_sleeve_phrase(sleeve)

        if sleeve_phrase:
            description = (
                f"A {sleeve_phrase} {pattern} {silhouette} {garment_type}, "
                f"suited for {season} wear."
            )
        else:
            description = (
                f"A {pattern} {silhouette} {garment_type}, "
                f"suited for {season} wear."
            )

    return description, predictions


def display_result(image_path, description):
    """
    Display and save the image with the generated description.

    Args:
        image_path: Path to the clothing image.
        description: Human-readable clothing description.

    Returns:
        output_path: Path where the result image was saved.
    """
    image = plt.imread(image_path)

    os.makedirs("outputs/classification", exist_ok=True)

    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(
        "outputs/classification",
        f"{image_stem}_result.png",
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(description, fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()

    print("\nGenerated description:")
    print(description)
    print(f"\nSaved result to: {output_path}")

    return output_path


def main():
    """
    Run the clothing description demo from the command line.

    Args:
        None directly. Uses sys.argv for an optional image path.

    Returns:
        None.
    """
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "test_images/sample.jpg"

    classifiers, encoders = load_models()
    description, predictions = describe_image(image_path, classifiers, encoders)

    print("\nRaw predictions:")
    for label, prediction in predictions.items():
        print(f"  {label}: {prediction}")

    display_result(image_path, description)


if __name__ == "__main__":
    main()
