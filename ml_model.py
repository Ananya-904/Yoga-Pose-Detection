"""
Machine Learning Pose Classifier (Random Forest, CSV-based)
-----------------------------------------------------------

This module trains a RandomForest model on pose angle features stored in a CSV.

Expected CSV structure:
    - One row per example
    - Numeric columns for angle features
      (e.g. knee_left, knee_right, elbow_left, ..., spine)
    - A target column named 'label' holding the pose name

Example header:
    knee_left,knee_right,elbow_left,elbow_right,shoulder_left,shoulder_right,hip_left,hip_right,spine,label

Usage (from the `yoga-detection` folder):

    python ml_model.py --csv-path path/to/pose_data.csv

This will:
    - load features/labels from the CSV
    - train a RandomForest classifier
    - print train/test accuracy and a classification report
    - save the trained model bundle to `pose_classifier.joblib`
      (used automatically by `feedback_engine.py`)
"""

import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_csv_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load dataset from a CSV file.

    Returns:
        X: [n_samples, n_features] feature matrix
        y: [n_samples] numeric labels
        label_names: list of class names
        feature_order: list of feature column names (in order)
    """
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column for pose names.")

    feature_cols = [c for c in df.columns if c != "label"]
    if not feature_cols:
        raise ValueError("CSV must contain at least one feature column besides 'label'.")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df["label"].astype(str).to_numpy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    return X, y, list(encoder.classes_), feature_cols


def train_pose_classifier(
    csv_path: str,
    model_path: str = "pose_classifier.joblib",
) -> None:
    """
    Train a RandomForest-based pose classifier on a CSV dataset.
    """
    print(f"[INFO] Loading dataset from: {csv_path}")
    X, y, label_names, feature_order = load_csv_dataset(csv_path)
    print(f"[INFO] Loaded {len(X)} samples, {len(label_names)} classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    print("[INFO] Training RandomForest classifier...")
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Test accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=label_names))

    bundle = {
        "model": clf,
        "scaler": scaler,
        "classes": label_names,
        # Store feature names so runtime can build vectors in same order
        "feature_order": feature_order,
    }
    dump(bundle, model_path)
    print(f"[INFO] Saved trained model to: {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train yoga pose classifier from CSV.")
    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path to CSV file containing pose features and labels.",
    )
    parser.add_argument(
        "--model-path",
        default="pose_classifier.joblib",
        help="Output path for saved model bundle (joblib).",
    )

    args = parser.parse_args()
    train_pose_classifier(args.csv_path, args.model_path)


if __name__ == "__main__":
    main()


