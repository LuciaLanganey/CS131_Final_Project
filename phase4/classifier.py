import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase3 import pattern_features
from phase3.features import extract_silhouette_features, segment_garment


def load_labels(csv_path):
    """Load clothing image labels from a CSV file."""
    df = pd.read_csv(csv_path)
    df = df[["filename", "pattern", "sleeve", "silhouette", "season"]]

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

    X = np.array(X_rows)
    y = {
        "pattern": np.array(pattern_labels),
        "sleeve": np.array(sleeve_labels),
        "silhouette": np.array(silhouette_labels),
        "season": np.array(season_labels),
    }

    print("Final feature matrix shape:", X.shape)
    return X, y


if __name__ == "__main__":
    df = load_labels("data/labels.csv")
    X, y = build_feature_matrix(df)

    print("X.shape:", X.shape)
    if len(X) > 0:
        print("First row of X:", X[0])

    print("\npattern counts:")
    print(pd.Series(y["pattern"]).value_counts())

    print("\nsleeve counts:")
    print(pd.Series(y["sleeve"]).value_counts())

    print("\nsilhouette counts:")
    print(pd.Series(y["silhouette"]).value_counts())

    print("\nseason counts:")
    print(pd.Series(y["season"]).value_counts())
