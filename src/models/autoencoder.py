"""
Autoencoder implementation used throughout the project.

The autoencoder learns the normal behaviour of network traffic.
Samples with high reconstruction error are treated as anomalies.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from configs.config import (
    AE_BATCH_SIZE,
    AE_EPOCHS,
    AE_ENCODING_DIM,
    AE_LEARNING_RATE,
    AE_THRESHOLD_PERCENTILE,
    AE_HIDDEN_DIMS,
    AE_DROPOUT_RATE,
    AUTOENCODER_METRICS_PATH,
    AUTOENCODER_MODEL_PATH,
    NORMAL_LABEL,
    RANDOM_STATE,
)
from src.utils.helpers import save_json

MODEL_PATH = AUTOENCODER_MODEL_PATH
METRICS_PATH = AUTOENCODER_METRICS_PATH

# Makes training reproducible.
tf.keras.utils.set_random_seed(RANDOM_STATE)


def build_autoencoder(
    input_dimension: int,
) -> Model:
    """
    Build the autoencoder architecture.
    """

    if input_dimension <= 0:
        raise ValueError(
            f"Input dimension must be a positive integer. Got: {input_dimension}"
        )

    inputs = Input(shape=(input_dimension,))
    x = inputs

    # Encoder loop constructing dense and dropout layers dynamically
    for i, dim in enumerate(AE_HIDDEN_DIMS):
        x = Dense(dim, activation="relu")(x)
        if i == 0 and AE_DROPOUT_RATE > 0:
            x = Dropout(AE_DROPOUT_RATE)(x)

    # Latent space bottleneck representing compressed features
    latent = Dense(
        AE_ENCODING_DIM,
        activation="relu",
        name="latent_space",
    )(x)

    # Decoder loop reconstructing the compressed features back to input shape
    x = latent
    num_dims = len(AE_HIDDEN_DIMS)
    for i, dim in enumerate(reversed(AE_HIDDEN_DIMS)):
        x = Dense(dim, activation="relu")(x)
        if i == num_dims - 2 and AE_DROPOUT_RATE > 0:
            x = Dropout(AE_DROPOUT_RATE)(x)

    outputs = Dense(
        input_dimension,
        activation="linear",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="NetworkAutoencoder",
    )

    model.compile(
        optimizer=Adam(
            learning_rate=AE_LEARNING_RATE,
        ),
        loss="mse",
    )

    return model


def train(
    x_train: pd.DataFrame,
    labels: pd.Series,
) -> tuple[Model, tf.keras.callbacks.History]:
    """
    Train only on normal traffic.

    Autoencoders are unsupervised deep learning architectures designed to compress inputs into
    a lower-dimensional latent space and then reconstruct them. They are trained only on normal traffic baseline.
    During inference, attacks will fail to reconstruct well, producing a high mean squared reconstruction error (MSE).
    Their main strength is catching complex, non-linear relationships, but they are computationally heavier to train.
    """

    if not isinstance(x_train, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame for features, but received: {type(x_train)}")

    if not isinstance(labels, pd.Series):
        try:
            labels = pd.Series(labels)
        except Exception:
            raise TypeError(f"Expected a pandas Series for labels, but received: {type(labels)}")

    if x_train.empty:
        raise ValueError("Cannot train model on empty training features.")

    if len(x_train) != len(labels):
        raise ValueError(
            f"Dimension mismatch: x_train has {len(x_train)} samples, "
            f"but labels has {len(labels)} samples."
        )

    # We check if there are any normal samples to train on, otherwise the autoencoder has no baseline.
    if NORMAL_LABEL not in labels.values:
        raise ValueError(
            f"Cannot train autoencoder: normal label '{NORMAL_LABEL}' was not found in training labels. "
            "Autoencoders require normal traffic to learn baseline behavior."
        )

    normal_samples = x_train[
        labels == NORMAL_LABEL
    ]

    model = build_autoencoder(
        input_dimension=x_train.shape[1]
    )

    callbacks = [
        EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=2,
            verbose=0,
        ),
    ]

    history = model.fit(
        normal_samples,
        normal_samples,
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        shuffle=True,
        verbose=0,
        callbacks=callbacks,
    )

    return model, history


def reconstruction_error(
    model: Model,
    x: pd.DataFrame,
) -> np.ndarray:
    """
    Calculate reconstruction error for every sample.
    """

    if model is None:
        raise ValueError("Cannot compute reconstruction error: model is None.")

    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but received: {type(x)}")

    if x.empty:
        raise ValueError("Cannot compute reconstruction error on empty DataFrame.")

    # We check if feature counts align with what the autoencoder expects to avoid
    # downstream shape alignment errors.
    expected_dim = model.input_shape[1]
    if x.shape[1] != expected_dim:
        raise ValueError(
            f"Input feature count mismatch. The autoencoder was built for {expected_dim} features, "
            f"but received input with {x.shape[1]} features."
        )

    reconstructed = model.predict(
        x,
        verbose=0,
    )

    return np.mean(
        np.square(x - reconstructed),
        axis=1,
    )


def predict(
    reconstruction_errors: np.ndarray,
    threshold: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Convert reconstruction errors into binary predictions.
    """

    if reconstruction_errors is None or len(reconstruction_errors) == 0:
        raise ValueError("Reconstruction errors array cannot be None or empty.")

    if threshold is not None and threshold < 0:
        raise ValueError(f"Reconstruction threshold must be non-negative. Got: {threshold}")

    if threshold is None:
        threshold = np.percentile(
            reconstruction_errors,
            AE_THRESHOLD_PERCENTILE,
        )

    predictions = (
        reconstruction_errors > threshold
    ).astype(int)

    return predictions, threshold


def save_model(
    model: Model,
    path: Path = MODEL_PATH,
) -> None:
    """
    Save the trained autoencoder.
    """

    if model is None:
        raise ValueError("Cannot save model: model is None.")

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(path)


def load_model(
    path: Path = MODEL_PATH,
) -> Model:
    """
    Load a trained autoencoder.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Autoencoder model weights not found at: {path}. "
            "Ensure the model has been trained and saved successfully."
        )

    return tf.keras.models.load_model(path)


def save_metrics(
    metrics: dict,
    path: Path = METRICS_PATH,
) -> None:
    """
    Save evaluation metrics.
    """

    if not isinstance(metrics, dict):
        raise TypeError(f"Expected a dictionary for metrics, but received: {type(metrics)}")

    save_json(metrics, path)


def get_metadata(
    model: Model | None = None,
    training_time: float | None = None,
    inference_time: float | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return standardized metadata for the Autoencoder model.
    """

    metadata = {
        "Model Name": "Autoencoder",
        "Training Time": training_time if training_time is not None else 0.0,
        "Inference Time": inference_time if inference_time is not None else 0.0,
        "Hyperparameters": {
            "hidden_dims": AE_HIDDEN_DIMS,
            "dropout_rate": AE_DROPOUT_RATE,
            "encoding_dim": AE_ENCODING_DIM,
            "learning_rate": AE_LEARNING_RATE,
            "epochs": AE_EPOCHS,
            "batch_size": AE_BATCH_SIZE,
            "threshold_percentile": AE_THRESHOLD_PERCENTILE,
        },
        "Number of Features": model.input_shape[1] if model is not None else "N/A",
        "Evaluation Metrics": metrics if metrics is not None else {},
        "Model File Location": str(MODEL_PATH),
        "Version": "1.0.0",
    }

    return metadata