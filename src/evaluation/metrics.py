"""
Common evaluation utilities shared by every model.

Keeping all metric calculations here ensures that every notebook,
the Streamlit dashboard and future experiments report results
in exactly the same way.
"""

from typing import Any

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(
    y_true: Any,
    y_pred: Any,
) -> dict[str, float]:
    """
    Calculate the core evaluation metrics.

    Returns
    -------
    dict
        Precision, Recall, F1 Score and Accuracy.
    """

    return {
        "Precision": round(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            4,
        ),
        "Recall": round(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            4,
        ),
        "F1 Score": round(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            4,
        ),
        "Accuracy": round(
            accuracy_score(
                y_true,
                y_pred,
            ),
            4,
        ),
    }


def classification_report_dataframe(
    y_true: Any,
    y_pred: Any,
) -> pd.DataFrame:
    """
    Return the sklearn classification report
    as a nicely formatted DataFrame.
    """

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Attack"],
        zero_division=0,
        output_dict=True,
    )

    # We convert the classification report dict directly to a DataFrame and transpose it
    # so that classes (Normal, Attack) are rows and metrics (precision, recall, etc.)
    # are columns. This maps perfectly to pandas rendering in notebooks/dashboard.
    return (
        pd.DataFrame(report)
        .transpose()
        .round(4)
    )


def confusion_matrix_dataframe(
    y_true: Any,
    y_pred: Any,
) -> pd.DataFrame:
    """
    Return the confusion matrix as a DataFrame.

    This makes it easier to display inside notebooks
    and Streamlit.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    # Wrapping sklearn's raw confusion matrix array in a pandas DataFrame with explicit
    # labels ("Actual Normal", "Predicted Attack") makes it immediately readable and self-documenting.
    return pd.DataFrame(
        matrix,
        index=["Actual Normal", "Actual Attack"],
        columns=["Predicted Normal", "Predicted Attack"],
    )


def metrics_dataframe(
    metrics: dict[str, Any],
) -> pd.DataFrame:
    """
    Convert a metrics dictionary into a one-row DataFrame.
    """

    return pd.DataFrame([metrics])


def print_metrics(
    metrics: dict[str, float],
) -> None:
    """
    Pretty-print evaluation metrics.

    Mostly useful while experimenting in notebooks.
    """

    print()

    for metric, value in metrics.items():
        print(f"{metric:<12}: {value:.4f}")

    print()