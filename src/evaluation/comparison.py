"""
Utilities for comparing multiple anomaly detection models.

This module keeps model comparison logic separate from
the individual model implementations.
"""

import pandas as pd


def create_comparison_table(
    isolation_forest_metrics: dict | None = None,
    dbscan_metrics: dict | None = None,
    autoencoder_metrics: dict | None = None,
    model_metrics: dict[str, dict] | list[dict] | None = None,
) -> pd.DataFrame:
    """
    Build a comparison table for all models.

    This function supports both backward-compatible parameters and a generic
    model_metrics dictionary or list for future model extensibility (e.g. One-Class SVM, VAE).
    """

    rows = []

    # 1. Handle backward-compatible explicit parameters
    if isolation_forest_metrics is not None:
        rows.append({"Model": "Isolation Forest", **isolation_forest_metrics})
    if dbscan_metrics is not None:
        rows.append({"Model": "DBSCAN", **dbscan_metrics})
    if autoencoder_metrics is not None:
        rows.append({"Model": "Autoencoder", **autoencoder_metrics})

    # 2. Handle generic model_metrics input for extensible dynamic models
    if model_metrics is not None:
        if isinstance(model_metrics, dict):
            for model_name, metrics in model_metrics.items():
                rows.append({"Model": model_name, **metrics})
        elif isinstance(model_metrics, list):
            for item in model_metrics:
                if isinstance(item, dict) and "Model" in item:
                    rows.append(item)
                else:
                    raise ValueError("List items in model_metrics must be dictionaries containing a 'Model' key.")
        else:
            raise TypeError("model_metrics must be a dict or a list of dicts.")

    if not rows:
        raise ValueError("No model metrics were provided to build the comparison table.")

    comparison = pd.DataFrame(rows)

    # We round metrics to 4 decimal places because it balances scientific precision
    # with UI display readability when plotted or tabulated in the dashboard.
    return comparison.round(4)


def best_model(
    comparison: pd.DataFrame,
    metric: str = "F1 Score",
) -> tuple[str, float]:
    """
    Return the model with the highest score
    for the selected metric.
    """

    # Using pandas idxmax lets us grab the row index of the top-performing model
    # for our chosen metric without needing to sort the entire dataframe.
    best_row = comparison.loc[
        comparison[metric].idxmax()
    ]

    return (
        best_row["Model"],
        best_row[metric],
    )


def rank_models(
    comparison: pd.DataFrame,
    metric: str = "F1 Score",
) -> pd.DataFrame:
    """
    Rank all models based on a selected metric.
    """

    return (
        comparison
        .sort_values(
            metric,
            ascending=False,
        )
        .reset_index(drop=True)
    )