import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase3 import pattern_features
from phase3.features import extract_silhouette_features, segment_garment

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_labels(csv_path):
    """Load clothing image labels from a CSV file."""
    df = pd.read_csv(csv_path)
    df = df[["filename", "garment_type", "pattern", "sleeve", "silhouette", "season"]]

    print("Loaded", len(df), "samples")
    return df


def extract_features_for_image(image_path):
    """Extract silhouette + pattern features for one image."""
    try:
        seg = segment_garment(image_path)

        silhouette_feats = extract_silhouette_features(seg["mask"], seg["edges"])
        pattern_feats, _ = pattern_features.extract_pattern_features(image_path)

        # Combine silhouette and pattern features
        all_feats = {}
        all_feats.update(silhouette_feats)
        all_feats.update(pattern_feats)

        values = []
        for key in sorted(all_feats.keys()):
            val = all_feats[key]
            if isinstance(val, list):
                for v in val:
                    values.append(float(v))
            else:
                values.append(float(val))

        return np.array(values)

    except Exception:
        print("Warning: could not extract features for", image_path)
        return None


def build_feature_matrix(df, image_dir="test_images/"):
    """Build X (features) and y (labels) from the labeled dataframe."""
    X_rows = []
    pattern_labels = []
    sleeve_labels = []
    silhouette_labels = []
    season_labels = []
    garment_type_labels = []

    for i in range(len(df)):
        row = df.iloc[i]
        image_path = os.path.join(image_dir, row["filename"])

        features = extract_features_for_image(image_path)
        if features is None:
            continue

        X_rows.append(features)
        pattern_labels.append(row["pattern"])
        sleeve_labels.append(row["sleeve"])
        silhouette_labels.append(row["silhouette"])
        season_labels.append(row["season"])
        garment_type_labels.append(row["garment_type"])

    X = np.array(X_rows)
    y = {
        "pattern": np.array(pattern_labels),
        "sleeve": np.array(sleeve_labels),
        "silhouette": np.array(silhouette_labels),
        "season": np.array(season_labels),
        "garment_type": np.array(garment_type_labels),
    }

    print("Final feature matrix shape:", X.shape)
    return X, y


def sleeve_training_mask(garment_types):
    """Return True for rows that should be used when training sleeve labels."""
    garment_types = np.array(garment_types)
    return garment_types != "pants"

    
def encode_labels(y_dict, garment_types=None):
    """
    Encode string labels into integer labels for classifier training.

    Args:
        y_dict: Dictionary mapping label names to numpy arrays of string labels.
            Expected keys are pattern, sleeve, silhouette, season, and garment_type.

    Returns:
        y_encoded: Dictionary mapping each label name to a numpy array of
            encoded integer labels.
        encoders: Dictionary mapping each label name to its fitted LabelEncoder.
    """
    y_encoded = {}
    encoders = {}

    for label_name, labels in y_dict.items():
        encoder = LabelEncoder()

        if label_name == "sleeve" and garment_types is not None:
            mask = sleeve_training_mask(garment_types)
            encoder.fit(labels[mask])
            encoded_labels = np.full(len(labels), -1, dtype=int)
            encoded_labels[mask] = encoder.transform(labels[mask])
            print(
                f"{label_name} classes:",
                list(encoder.classes_),
                f"(trained on {mask.sum()} non-pants samples)",
            )
        else:
            encoded_labels = encoder.fit_transform(labels)
            print(f"{label_name} classes:", list(encoder.classes_))

        y_encoded[label_name] = encoded_labels
        encoders[label_name] = encoder

    return y_encoded, encoders


def train_classifiers(X, y_encoded, garment_types=None):
    """
    Train and select the best classifier for each clothing label.

    For each label type, this function trains three classifiers:
    Random Forest, SVM, and Logistic Regression. The model with the highest
    validation accuracy is selected as the best model for that label.

    Args:
        X: Numpy array of shape (n_samples, n_features), containing extracted
            image features.
        y_encoded: Dictionary mapping label names to encoded integer label arrays.

    Returns:
        best_classifiers: Dictionary mapping each label name to its best fitted
            classifier.
        results: Dictionary mapping each label name to a dictionary containing
            the winning model name, validation accuracy, held-out X_test, and
            held-out y_test.
    """
    best_classifiers = {}
    results = {}

    for label_name, y in y_encoded.items():
        print(f"\nTraining classifiers for: {label_name}")

        X_use = X
        y_use = y

        if label_name == "sleeve" and garment_types is not None:
            mask = sleeve_training_mask(garment_types)
            X_use = X[mask]
            y_use = y[mask]
            print(f"  Skipping pants: training sleeve on {len(y_use)} samples")

        # Stratify only works if each class has at least 2 samples.
        unique_classes, class_counts = np.unique(y_use, return_counts=True)
        n_classes = len(unique_classes)
        n_samples = len(y_use)
        test_size = 0.2
        n_test = int(np.ceil(test_size * n_samples))

        can_stratify = (
            n_classes > 1
            and np.all(class_counts >= 2)
            and n_test >= n_classes
        )

        if can_stratify:
            stratify_arg = y_use
        else:
            stratify_arg = None
            print(
                f"Warning: not using stratify for {label_name} because the dataset is too small "
                f"for a stratified split. n_samples={n_samples}, n_test={n_test}, "
                f"n_classes={n_classes}, class_counts={class_counts.tolist()}"
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X_use,
            y_use,
            test_size=0.3,
            random_state=42,
            stratify=stratify_arg,
        )

        candidate_models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
            "SVC": make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", random_state=42),
            ),
            "LogisticRegression": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, random_state=42),
            ),
        }

        best_model_name = None
        best_model = None
        best_accuracy = -1.0

        for model_name, model in candidate_models.items():
            try:
                model.fit(X_train, y_train)

                train_pred = model.predict(X_train)
                train_accuracy = accuracy_score(y_train, train_pred)

                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                print(f"  {model_name} train accuracy: {train_accuracy:.4f}")
                print(f"  {model_name} validation accuracy: {accuracy:.4f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = model_name
                    best_model = model

            except Exception as exc:
                print(f"  Warning: {model_name} failed for {label_name}: {exc}")

        if best_model is None:
            print(f"Warning: no classifier could be trained for {label_name}")
            continue

        best_classifiers[label_name] = best_model
        results[label_name] = {
            "best_model": best_model_name,
            "accuracy": float(best_accuracy),
            "X_test": X_test,
            "y_test": y_test,
        }

        print(
            f"Best model for {label_name}: "
            f"{best_model_name} with accuracy {best_accuracy:.4f}"
        )

    return best_classifiers, results


def save_classifiers(
    best_classifiers,
    encoders,
    out_dir="outputs/classification/models/",
):
    """
    Save trained classifiers and label encoders as .pkl files.

    Args:
        best_classifiers: Dictionary mapping label names to fitted classifiers.
        encoders: Dictionary mapping label names to fitted LabelEncoders.
        out_dir: Directory where model and encoder files should be saved.

    Returns:
        None.
    """
    os.makedirs(out_dir, exist_ok=True)

    for label_name, classifier in best_classifiers.items():
        classifier_path = os.path.join(out_dir, f"{label_name}_clf.pkl")
        joblib.dump(classifier, classifier_path)
        print(f"Saved classifier: {classifier_path}")

    for label_name, encoder in encoders.items():
        encoder_path = os.path.join(out_dir, f"{label_name}_enc.pkl")
        joblib.dump(encoder, encoder_path)
        print(f"Saved encoder: {encoder_path}")


if __name__ == "__main__":
    df = load_labels("data/labels.csv")
    X, y = build_feature_matrix(df)
    
    if len(X) == 0:
        raise ValueError(
            "No feature rows were built."
        )

    print("X.shape:", X.shape)
    if len(X) > 0:
        print("First row of X:", X[0])

    print("\npattern counts:")
    print(pd.Series(y["pattern"]).value_counts())

    print("\nsleeve counts (non-pants only):")
    sleeve_mask = sleeve_training_mask(y["garment_type"])
    print(pd.Series(y["sleeve"][sleeve_mask]).value_counts())

    print("\nsilhouette counts:")
    print(pd.Series(y["silhouette"]).value_counts())

    print("\nseason counts:")
    print(pd.Series(y["season"]).value_counts())

    print("\ngarment_type counts:")
    print(pd.Series(y["garment_type"]).value_counts())

    y_encoded, encoders = encode_labels(y, garment_types=y["garment_type"])
    best_classifiers, results = train_classifiers(
        X, y_encoded, garment_types=y["garment_type"]
    )
    save_classifiers(best_classifiers, encoders)

    print("\nFinal training summary:")
    for label_name, result in results.items():
        print(
            f"{label_name}: "
            f"{result['best_model']} "
            f"accuracy={result['accuracy']:.4f}"
        )
