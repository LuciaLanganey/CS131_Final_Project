import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase4.classifier import (
    build_feature_matrix,
    load_labels,
    label_training_mask,
    encode_labels,
)

LABELS = ["pattern", "sleeve", "silhouette", "season", "garment_type"]


def get_test_data(X, y):
    """Get the same test split that classifier.py uses during training."""
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(unique_classes)
    n_samples = len(y)
    test_size = 0.2
    n_test = int(np.ceil(test_size * n_samples))

    can_stratify = (
        n_classes > 1
        and np.all(class_counts >= 2)
        and n_test >= n_classes
    )

    if can_stratify:
        stratify_arg = y
    else:
        stratify_arg = None

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=stratify_arg,
    )
    return X_test, y_test


def load_models(model_dir="outputs/classification/models/"):
    """Load saved classifiers and label encoders from .pkl files."""
    classifiers = {}
    encoders = {}

    for label in LABELS:
        clf_path = os.path.join(model_dir, label + "_clf.pkl")
        enc_path = os.path.join(model_dir, label + "_enc.pkl")

        classifiers[label] = joblib.load(clf_path)
        encoders[label] = joblib.load(enc_path)

        print("Loaded classifier:", clf_path)
        print("Loaded encoder:", enc_path)

    print("\nLoaded", len(classifiers), "classifiers and", len(encoders), "encoders")
    return classifiers, encoders


def evaluate_all(classifiers, encoders, X, y_encoded, garment_types=None):
    """Evaluate each saved classifier on test data."""
    out_dir = "outputs/classification/confusion_matrices/"
    os.makedirs(out_dir, exist_ok=True)

    summary = {}

    for label in LABELS:
        clf = classifiers[label]
        encoder = encoders[label]

        if garment_types is not None:
            mask = label_training_mask(label, garment_types)
            X_use = X[mask]
            y_use = y_encoded[label][mask]

            if label != "garment_type":
                print(f"\nEvaluating {label} on {len(y_use)} applicable clothing samples")
        else:
            X_use = X
            y_use = y_encoded[label]

        X_test, y_test = get_test_data(X_use, y_use)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        class_names = list(encoder.classes_)
        cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))

        print("\n" + label + " accuracy:", f"{acc:.4f}")
        print(label + " confusion matrix:")
        print("             ", end="")
        for name in class_names:
            print(name.rjust(12), end="")
        print()
        for i in range(len(class_names)):
            print(class_names[i].rjust(12), end="")
            for j in range(len(class_names)):
                print(str(cm[i, j]).rjust(12), end="")
            print()

        # Save confusion matrix plot
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(label + " Confusion Matrix")

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        save_path = os.path.join(out_dir, label + "_confusion.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print("Saved confusion matrix:", save_path)

        # Get model name from saved classifier object
        model = clf
        if hasattr(model, "steps"):
            model = model.steps[-1][1]
        model_name = type(model).__name__
        if model_name == "RandomForestClassifier":
            model_name = "RandomForest"
        elif model_name == "SVC":
            model_name = "SVC"
        elif model_name == "LogisticRegression":
            model_name = "LogisticRegression"

        summary[label] = {
            "accuracy": float(acc),
            "best_model": model_name,
        }

    return summary


def print_report(summary):
    """Print a summary table of evaluation results."""
    print("\nEvaluation Summary")
    print("Label          | Accuracy")
    print("---------------|----------")
    for label in LABELS:
        acc = summary[label]["accuracy"]
        print(f"{label:<14} | {acc:.2f}")


if __name__ == "__main__":
    df = load_labels("data/labels.csv")

    for col in ["pattern", "sleeve", "silhouette", "season", "garment_type"]:
        df[col] = df[col].astype(str).str.strip()

    X, y = build_feature_matrix(df)

    if len(X) == 0:
        raise ValueError("No feature rows were built.")

    classifiers, encoders = load_models()
    
    y_encoded, _ = encode_labels(
        y,
        garment_types=y["garment_type"],
    )

    summary = evaluate_all(
        classifiers,
        encoders,
        X,
        y_encoded,
        garment_types=y["garment_type"],
    )

    print_report(summary)
