"""Train the Iris RandomForest classifier and persist the model + dataset.

Data pipeline:
    sklearn.datasets.load_iris  ->  data/iris_features.csv, data/iris_target.csv
                                ->  train/test split (70/30, stratified, seeded)
                                ->  RandomForestClassifier
                                ->  model/iris_classifier_rf.joblib
                                ->  model/metrics.json

Usage:
    python train_iris_model.py

Paths can be overridden with the IRIS_MODEL_DIR and IRIS_DATA_DIR env vars.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("IRIS_MODEL_DIR", BASE_DIR / "model"))
DATA_DIR = Path(os.environ.get("IRIS_DATA_DIR", BASE_DIR / "data"))
MODEL_PATH = MODEL_DIR / "iris_classifier_rf.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

RANDOM_STATE = 42
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def train(model_dir: Path = MODEL_DIR, data_dir: Path = DATA_DIR) -> dict:
    """Train, evaluate and save the model. Returns the metrics dict."""
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")

    # Persist raw data so the pipeline is reproducible / inspectable.
    X.to_csv(data_dir / "iris_features.csv", index=False)
    y.to_csv(data_dir / "iris_target.csv", index=False, header=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    metrics = {
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "feature_names": FEATURE_NAMES,
        "target_names": list(iris.target_names),
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
    }

    joblib.dump(clf, model_dir / MODEL_PATH.name)
    (model_dir / METRICS_PATH.name).write_text(json.dumps(metrics, indent=2) + "\n")

    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to {model_dir / MODEL_PATH.name}")
    return metrics


if __name__ == "__main__":
    train()
