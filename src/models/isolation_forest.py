"""
Isolation Forest implementation used throughout the project.

The goal of this module is to keep everything related to the model
in one place—training, prediction, scoring, saving and loading.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from configs.config import (
    IF_CONTAMINATION,
    IF_N_ESTIMATORS,
    ISOLATION_FOREST_METRICS_PATH,
    ISOLATION_FOREST_MODEL_PATH,
    RANDOM_STATE,
)
from src.utils.helpers import save_json

MODEL_PATH = ISOLATION_FOREST_MODEL_PATH
METRICS_PATH = ISOLATION_FOREST_METRICS_PATH


def train(
    x_train: pd.DataFrame,
) -> IsolationForest:
    """
    Train an Isolation Forest model.
    """

    if not isinstance(x_train, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame for features, but received: {type(x_train)}")

    if x_train.empty:
        raise ValueError("Cannot train Isolation Forest on an empty DataFrame.")

    # Isolation Forest assumes anomalies are relatively rare.
    # The NSL-KDD dataset violates this assumption, so contamination
    # is capped at 0.5 due to sklearn limitations.
    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(x_train)

    return model


def predict(
    model: IsolationForest,
    x: pd.DataFrame,
) -> np.ndarray:
    """
    Return binary predictions.

    Isolation Forest outputs:
        1  -> normal
       -1  -> anomaly

    We convert them into:
        0 -> normal
        1 -> anomaly
    """

    if model is None:
        raise ValueError("Cannot predict: model is None.")

    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but received: {type(x)}")

    if x.empty:
        raise ValueError("Cannot predict on an empty DataFrame.")

    # We check if features align with the fitted model's features to prevent shape mismatch crashes.
    if x.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Input feature count mismatch. The Isolation Forest model expects {model.n_features_in_} features, "
            f"but received input with {x.shape[1]} features."
        )

    predictions = model.predict(x)

    return (predictions == -1).astype(int)


def anomaly_scores(
    model: IsolationForest,
    x: pd.DataFrame,
) -> np.ndarray:
    """
    Higher values indicate a greater chance of being an anomaly.
    """

    if model is None:
        raise ValueError("Cannot compute anomaly scores: model is None.")

    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but received: {type(x)}")

    if x.empty:
        raise ValueError("Cannot compute anomaly scores on an empty DataFrame.")

    if x.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Input feature count mismatch. The Isolation Forest model expects {model.n_features_in_} features, "
            f"but received input with {x.shape[1]} features."
        )

    return -model.decision_function(x)


def save_model(
    model: IsolationForest,
    path: Path = MODEL_PATH,
) -> None:
    """
    Save the trained model to disk.
    """

    if model is None:
        raise ValueError("Cannot save model: model is None.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)


def load_model(
    path: Path = MODEL_PATH,
) -> IsolationForest:
    """
    Load a previously trained model.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Isolation Forest model file not found at: {path}. "
            "Ensure the model has been trained and saved successfully."
        )

    return joblib.load(path)


def save_metrics(
    metrics: dict,
    path: Path = METRICS_PATH,
) -> None:
    """
    Save evaluation metrics as JSON.
    """

    if not isinstance(metrics, dict):
        raise TypeError(f"Expected a dictionary for metrics, but received: {type(metrics)}")

    save_json(metrics, path)


def get_metadata(
    model: IsolationForest | None = None,
    training_time: float | None = None,
    inference_time: float | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return standardized metadata for the Isolation Forest model.
    """

    metadata = {
        "Model Name": "Isolation Forest",
        "Training Time": training_time if training_time is not None else 0.0,
        "Inference Time": inference_time if inference_time is not None else 0.0,
        "Hyperparameters": {
            "n_estimators": IF_N_ESTIMATORS,
            "contamination": IF_CONTAMINATION,
            "random_state": RANDOM_STATE,
        },
        "Number of Features": model.n_features_in_ if model is not None else "N/A",
        "Evaluation Metrics": metrics if metrics is not None else {},
        "Model File Location": str(MODEL_PATH),
        "Version": "1.0.0",
    }

    return metadata