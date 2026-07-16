"""
DBSCAN implementation used throughout the project.

Unlike Isolation Forest, DBSCAN is a clustering algorithm and does not
learn a reusable model. Every new dataset must be clustered again.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from configs.config import (
    DBSCAN_EPS,
    DBSCAN_METRICS_PATH,
    DBSCAN_MIN_SAMPLES,
)
from src.utils.helpers import save_json

METRICS_PATH = DBSCAN_METRICS_PATH


def train(
    x: pd.DataFrame,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> tuple[DBSCAN, np.ndarray]:
    """
    Cluster the dataset using DBSCAN.

    Unlike Isolation Forest or Autoencoders, DBSCAN is a density-based clustering algorithm.
    It is highly effective at discovering arbitrary-shaped clusters and automatically
    identifying outliers (noise points, labeled as -1) without requiring labels.
    However, because it computes pairwise distances, it scales quadratically O(N^2) in memory/time
    and does not support predicting on unseen data, meaning the entire dataset must be clustered again.
    """

    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but received: {type(x)}")

    if x.empty:
        raise ValueError("Cannot run DBSCAN on an empty DataFrame.")

    if eps <= 0:
        raise ValueError(f"DBSCAN eps parameter must be positive. Got: {eps}")

    if min_samples <= 0:
        raise ValueError(f"DBSCAN min_samples parameter must be positive. Got: {min_samples}")

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        n_jobs=-1,
    )

    cluster_labels = model.fit_predict(x)

    return model, cluster_labels


def predict(
    cluster_labels: np.ndarray,
) -> np.ndarray:
    """
    Convert DBSCAN cluster labels into binary anomaly labels.

    Cluster '-1' represents noise points, which we treat as anomalies.
    """

    if cluster_labels is None or len(cluster_labels) == 0:
        raise ValueError("DBSCAN cluster labels array cannot be None or empty.")

    return (cluster_labels == -1).astype(int)


def cluster_summary(
    cluster_labels: np.ndarray,
) -> dict:
    """
    Return a quick summary of the generated clusters.
    """

    if cluster_labels is None or len(cluster_labels) == 0:
        raise ValueError("DBSCAN cluster labels array cannot be None or empty.")

    unique_clusters = np.unique(cluster_labels)

    noise_points = int((cluster_labels == -1).sum())

    return {
        "Total Clusters": len(unique_clusters) - (1 if -1 in unique_clusters else 0),
        "Noise Points": noise_points,
        "Cluster Labels": unique_clusters.tolist(),
    }


def save_metrics(
    metrics: dict,
    path: Path = METRICS_PATH,
) -> None:
    """
    Save evaluation metrics as a JSON file.
    """

    if not isinstance(metrics, dict):
        raise TypeError(f"Expected a dictionary for metrics, but received: {type(metrics)}")

    save_json(metrics, path)


def save_model(
    model: DBSCAN,
    path: Path | None = None,
) -> None:
    """
    Placeholder to maintain naming consistency.

    DBSCAN is a clustering algorithm and does not learn a reusable model
    that can predict on unseen data. Therefore, saving is not supported.
    """

    raise NotImplementedError(
        "DBSCAN does not support saving/loading a reusable model."
    )


def load_model(
    path: Path | None = None,
) -> DBSCAN:
    """
    Placeholder to maintain naming consistency.

    DBSCAN is a clustering algorithm and does not learn a reusable model
    that can predict on unseen data. Therefore, loading is not supported.
    """

    raise NotImplementedError(
        "DBSCAN does not support saving/loading a reusable model."
    )


def get_metadata(
    model: DBSCAN | None = None,
    training_time: float | None = None,
    inference_time: float | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return standardized metadata for the DBSCAN model.
    """

    metadata = {
        "Model Name": "DBSCAN",
        "Training Time": training_time if training_time is not None else 0.0,
        "Inference Time": inference_time if inference_time is not None else 0.0,
        "Hyperparameters": {
            "eps": DBSCAN_EPS,
            "min_samples": DBSCAN_MIN_SAMPLES,
        },
        "Number of Features": model.components_.shape[1] if (model is not None and hasattr(model, "components_")) else "N/A",
        "Evaluation Metrics": metrics if metrics is not None else {},
        "Model File Location": "N/A - DBSCAN does not support serialization",
        "Version": "1.0.0",
    }

    return metadata