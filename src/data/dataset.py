"""
Utility functions for loading and validating the NSL-KDD dataset.

This module is intentionally kept free from preprocessing logic.
Its only responsibility is reading the dataset and performing a few
basic validation checks before handing it over to the preprocessing
pipeline.
"""

from pathlib import Path

import pandas as pd

from configs.config import (
    DROP_COLUMNS,
    TARGET_COLUMN,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
)

# Official NSL-KDD feature names
DATASET_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Load an NSL-KDD dataset from disk.

    Parameters
    ----------
    file_path : str | Path
        Path to the dataset file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    file_path = Path(file_path)

    # We do a hard file check here. If the file is missing, raising a FileNotFoundError
    # prevents downstream pandas errors that are harder to trace.
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # Sometimes files can be malformed, empty, or locked. Wrapping read_csv in a try-except
    # block lets us surface a clean, descriptive error for developers.
    try:
        data = pd.read_csv(
            file_path,
            names=DATASET_COLUMNS,
        )
    except Exception as err:
        raise IOError(f"Failed to read dataset file at {file_path}. Error details: {err}")

    # The difficulty column is only meant for evaluation in the original dataset.
    # We drop it right away so we don't accidentally treat it as an active ML feature.
    if all(col in data.columns for col in DROP_COLUMNS):
        data = data.drop(columns=DROP_COLUMNS)

    validate_dataset(data)

    return data


def load_train_data() -> pd.DataFrame:
    """Load the NSL-KDD training dataset."""

    return load_dataset(TRAIN_DATA_PATH)


def load_test_data() -> pd.DataFrame:
    """Load the NSL-KDD testing dataset."""

    return load_dataset(TEST_DATA_PATH)


def validate_dataset(data: pd.DataFrame) -> None:
    """
    Run a few sanity checks before the data enters the pipeline.
    """

    # Ensure we actually received a pandas DataFrame to avoid obscure AttributeError crashes.
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but received: {type(data)}")

    if data.empty:
        raise ValueError("Loaded dataset is empty.")

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing from the dataset. "
            "Please check if the file format matches the expected NSL-KDD schema."
        )


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """
    Return all feature columns.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but received: {type(data)}")

    return [column for column in data.columns if column != TARGET_COLUMN]


def get_target_column() -> str:
    """
    Return the target column name.

    Keeping this as a helper avoids hardcoding the label column
    throughout the project.
    """

    return TARGET_COLUMN


def dataset_summary(data: pd.DataFrame) -> dict:
    """
    Return a lightweight summary of the dataset.

    Useful for notebooks and the Streamlit dashboard.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but received: {type(data)}")

    if data.empty:
        raise ValueError("Cannot summarize an empty dataset.")

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing from the dataset. "
            "Summary cannot be computed without it."
        )

    return {
        "rows": len(data),
        "columns": data.shape[1],
        "missing_values": int(data.isnull().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "attack_classes": data[TARGET_COLUMN].nunique(),
    }