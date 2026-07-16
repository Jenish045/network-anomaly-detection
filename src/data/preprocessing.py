"""
Data preprocessing utilities used across the project.

This module is responsible for converting the raw NSL-KDD dataset into
a format suitable for machine learning models. The same preprocessing
pipeline is shared by notebooks, model training, and the Streamlit app.
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from configs.config import (
    CATEGORICAL_COLUMNS,
    ENCODER_PATH,
    NORMAL_LABEL,
    SCALER_PATH,
    SCALER_TYPE,
    TARGET_COLUMN,
)


def split_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature columns from the target column.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but received: {type(data)}")

    if TARGET_COLUMN not in data.columns:
        raise KeyError(
            f"Cannot split features: target column '{TARGET_COLUMN}' is missing from the dataset. "
            "Verify that the column has not been dropped or renamed."
        )

    x = data.drop(columns=TARGET_COLUMN)
    y = data[TARGET_COLUMN]

    return x, y


def encode_categorical_features(
    data: pd.DataFrame,
    save_encoder: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Encode all categorical columns using LabelEncoder.

    The fitted encoders are saved so that uploaded datasets
    go through exactly the same transformation.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but received: {type(data)}")

    # Ensure all categorical columns are actually present in the dataframe before fitting.
    missing_cols = [col for col in CATEGORICAL_COLUMNS if col not in data.columns]
    if missing_cols:
        raise KeyError(
            f"Categorical column(s) {missing_cols} not found in the DataFrame. "
            "Please check if they were renamed or dropped."
        )

    data = data.copy()
    encoders = {}

    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[column])
        encoders[column] = encoder

    if save_encoder:
        # We ensure the parent directory (usually data/processed) exists to avoid a crash.
        ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoders, ENCODER_PATH)

    return data, encoders


def transform_categorical_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply previously fitted encoders to new data.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but received: {type(data)}")

    # Ensure columns exist prior to encoding them.
    missing_cols = [col for col in CATEGORICAL_COLUMNS if col not in data.columns]
    if missing_cols:
        raise KeyError(
            f"Categorical column(s) {missing_cols} not found in the DataFrame. "
            "Cannot apply categorical transformations."
        )

    data = data.copy()

    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Label encoders file not found at: {ENCODER_PATH}. "
            "Run the preprocessing pipeline on training data first to fit encoders."
        )

    # Wrap the load function to intercept corrupt file reads.
    try:
        encoders = joblib.load(ENCODER_PATH)
    except Exception as err:
        raise IOError(
            f"Failed to load label encoders from {ENCODER_PATH}. The file might be corrupted. "
            f"Please re-run the preprocessing pipeline on training data. Error details: {err}"
        ) from err

    for column in CATEGORICAL_COLUMNS:
        if column not in encoders:
            raise KeyError(f"No fitted encoder found for column '{column}' in the saved encoders.")

        encoder = encoders[column]

        # When unseen categories (like an unseen attack service or protocol) are encountered,
        # sklearn's LabelEncoder throws a ValueError. We intercept it and display exactly what
        # the unseen values are, and what values the encoder was fitted on.
        try:
            data[column] = encoder.transform(data[column])
        except ValueError as err:
            unseen = set(data[column].unique()) - set(encoder.classes_)
            raise ValueError(
                f"Unseen categorical value(s) {unseen} encountered in column '{column}'. "
                f"The fitted encoder only knows about the following classes: {list(encoder.classes_)}."
            ) from err

    return data


def fit_scaler(
    x_train: pd.DataFrame,
    save_scaler: bool = True,
) -> tuple[Any, pd.DataFrame]:
    """
    Fit a scaler (Standard/MinMax/Robust) using the training data only.
    """

    if not isinstance(x_train, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame for scaling, but received: {type(x_train)}")

    if x_train.empty:
        raise ValueError("Cannot fit scaler on an empty DataFrame.")

    # A common source of scikit-learn failures is passing unencoded string categories to standard scaler.
    # We run a proactive scan here and throw a clear, meaningful exception.
    import numpy as np
    non_num = x_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        raise ValueError(
            f"Scaler cannot scale non-numerical columns: {non_num}. "
            "Please ensure categorical variables are encoded before fitting the scaler."
        )

    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    if SCALER_TYPE == "standard":
        scaler = StandardScaler()
    elif SCALER_TYPE == "minmax":
        scaler = MinMaxScaler()
    elif SCALER_TYPE == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(
            f"Unsupported SCALER_TYPE: '{SCALER_TYPE}'. "
            "Please configure it to 'standard', 'minmax', or 'robust' inside configs/config.py."
        )

    x_scaled = scaler.fit_transform(x_train)

    if save_scaler:
        SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)

    x_scaled = pd.DataFrame(
        x_scaled,
        columns=x_train.columns,
        index=x_train.index,
    )

    return scaler, x_scaled


def transform_features(
    x: pd.DataFrame,
    scaler: Any = None,
) -> pd.DataFrame:
    """
    Scale data using an already fitted scaler.
    """

    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame for scaling, but received: {type(x)}")

    if x.empty:
        raise ValueError("Cannot scale features on an empty DataFrame.")

    import numpy as np
    non_num = x.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        raise ValueError(
            f"Scaler cannot scale non-numerical columns: {non_num}. "
            "Please ensure categorical variables are encoded before applying the scaler."
        )

    if scaler is None:
        if not SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Scaler joblib not found at: {SCALER_PATH}. "
                "Fit and save the scaler on training data first."
            )
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as err:
            raise IOError(
                f"Failed to load scaler from {SCALER_PATH}. The file might be corrupted. "
                f"Please re-run the preprocessing pipeline on training data. Error details: {err}"
            ) from err

    x_scaled = scaler.transform(x)

    return pd.DataFrame(
        x_scaled,
        columns=x.columns,
        index=x.index,
    )


def prepare_training_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Complete preprocessing pipeline for model training.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but received: {type(data)}")

    if data.empty:
        raise ValueError("Input data is empty.")

    data, _ = encode_categorical_features(data)
    x, y = split_features_and_target(data)
    _, x = fit_scaler(x)

    return x, y


def prepare_inference_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the saved preprocessing pipeline to unseen data.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but received: {type(data)}")

    if data.empty:
        raise ValueError("Input data is empty.")

    data = transform_categorical_features(data)
    x, _ = split_features_and_target(data)

    return transform_features(x)


def get_binary_labels(
    labels: pd.Series,
) -> pd.Series:
    """
    Convert attack names into binary labels.

    normal -> 0
    attack -> 1
    """

    if not isinstance(labels, pd.Series):
        # We try to coerce it into a series if it is list-like, to keep it flexible
        # but raise a helpful TypeError if that fails.
        try:
            labels = pd.Series(labels)
        except Exception as err:
            raise TypeError(
                f"Expected a pandas Series or list-like object for labels, "
                f"but received: {type(labels)}. Error: {err}"
            )

    return (labels != NORMAL_LABEL).astype(int)