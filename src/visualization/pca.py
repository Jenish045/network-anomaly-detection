"""
Utilities for Principal Component Analysis (PCA).

The project uses PCA only for visualization, not for training.
Keeping it here avoids repeating the same code in multiple notebooks
and inside the Streamlit dashboard.
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from configs.config import RANDOM_STATE


def compute_pca(
    data: pd.DataFrame,
    n_components: int = 2,
) -> tuple[pd.DataFrame, PCA]:
    """
    Perform PCA on the given dataset.

    Returns
    -------
    tuple
        - DataFrame containing principal components
        - Fitted PCA object
    """

    # Setting the random_state is essential because the singular value decomposition (SVD)
    # solver inside sklearn's PCA can affect the sign/orientation of the principal component
    # axes, which would make visualizations inconsistent across multiple runs.
    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_STATE,
    )

    transformed = pca.fit_transform(data)

    columns = [
        f"PC{i + 1}"
        for i in range(n_components)
    ]

    transformed = pd.DataFrame(
        transformed,
        columns=columns,
        index=data.index,
    )

    return transformed, pca


def explained_variance(
    pca: PCA,
) -> pd.DataFrame:
    """
    Return the explained variance of every principal component.
    """

    variance = pd.DataFrame(
        {
            "Principal Component": [
                f"PC{i + 1}"
                for i in range(
                    len(
                        pca.explained_variance_ratio_
                    )
                )
            ],
            "Explained Variance": pca.explained_variance_ratio_,
            "Cumulative Variance": (
                pca.explained_variance_ratio_
            ).cumsum(),
        }
    )

    return variance.round(4)


def prepare_pca_dataframe(
    transformed_data: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    """
    Combine PCA coordinates with labels.

    This makes plotting much easier because everything
    lives inside one DataFrame.
    """

    pca_df = transformed_data.copy()

    if hasattr(labels, "values"):
        pca_df["Label"] = labels.values
    else:
        pca_df["Label"] = np.array(labels)

    return pca_df


def get_feature_contribution(
    pca: PCA,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Return feature loadings for each principal component.
    """

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[
            f"PC{i + 1}"
            for i in range(
                pca.n_components_
            )
        ],
        index=feature_names,
    )

    return loadings


def cumulative_variance(
    pca: PCA,
) -> float:
    """
    Return the total explained variance.
    """

    return round(
        pca.explained_variance_ratio_.sum(),
        4,
    )