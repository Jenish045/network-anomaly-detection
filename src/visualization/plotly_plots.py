"""
Reusable Plotly visualizations.

Every notebook and the Streamlit dashboard should use these
functions instead of creating figures manually.
"""

from pathlib import Path
from typing import Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from configs.config import PLOTS_DIR
from configs.config import (
    FIGURE_HEIGHT,
    FIGURE_WIDTH,
    PLOT_TEMPLATE,
)


def _apply_standard_layout(
    figure: go.Figure,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
) -> go.Figure:
    """
    Internal helper to apply standardized layout settings to Plotly figures.
    """
    layout_update = {
        "template": PLOT_TEMPLATE,
        "font": dict(family="Outfit, Inter, Roboto, sans-serif", size=12),
    }

    # Add title settings
    t_text = title if title else (figure.layout.title.text if figure.layout.title else None)
    if t_text:
        layout_update["title"] = dict(
            text=t_text,
            font=dict(size=18, family="Outfit, Inter, Roboto, sans-serif", color="#1E293B"),
            x=0.05,
            xanchor="left",
        )

    # Set consistent responsive/standard sizing
    if not figure.layout.width:
        layout_update["width"] = FIGURE_WIDTH
    if not figure.layout.height:
        layout_update["height"] = FIGURE_HEIGHT

    figure.update_layout(**layout_update)

    # Apply axis labels if specified
    if xaxis_title:
        figure.update_xaxes(title_text=xaxis_title)
    if yaxis_title:
        figure.update_yaxes(title_text=yaxis_title)

    return figure


def pca_2d_plot(
    data: pd.DataFrame,
    color: str = "Label",
    title: str = "PCA Projection",
) -> go.Figure:
    """
    Create an interactive 2D PCA scatter plot.
    """

    figure = px.scatter(
        data,
        x="PC1",
        y="PC2",
        color=color,
        title=title,
    )

    return _apply_standard_layout(
        figure,
        title=title,
        xaxis_title="Principal Component 1 (PC1)",
        yaxis_title="Principal Component 2 (PC2)",
    )


def pca_3d_plot(
    data: pd.DataFrame,
    color: str = "Label",
    title: str = "3D PCA Projection",
) -> go.Figure:
    """
    Create an interactive 3D PCA scatter plot.
    """

    figure = px.scatter_3d(
        data,
        x="PC1",
        y="PC2",
        z="PC3",
        color=color,
        title=title,
    )

    return _apply_standard_layout(
        figure,
        title=title,
    )


def explained_variance_plot(
    variance_df: pd.DataFrame,
) -> go.Figure:
    """
    Plot the explained variance of each principal component.
    """

    figure = px.bar(
        variance_df,
        x="Principal Component",
        y="Explained Variance",
        title="Explained Variance by Principal Component",
    )

    return _apply_standard_layout(
        figure,
        xaxis_title="Principal Components",
        yaxis_title="Explained Variance Ratio",
    )


def metrics_bar_chart(
    comparison_table: pd.DataFrame,
    metric: str,
) -> go.Figure:
    """
    Compare model performance for a selected metric.
    """

    figure = px.bar(
        comparison_table,
        x="Model",
        y=metric,
        color="Model",
        title=f"{metric} Comparison",
    )

    return _apply_standard_layout(
        figure,
        xaxis_title="Algorithm",
        yaxis_title=metric,
    )


def radar_chart(
    comparison_table: pd.DataFrame,
) -> go.Figure:
    """
    Compare all models using a radar chart.
    """

    metrics = [
        "Precision",
        "Recall",
        "F1 Score",
        "Accuracy",
    ]

    figure = go.Figure()

    for _, row in comparison_table.iterrows():
        figure.add_trace(
            go.Scatterpolar(
                r=[row[metric] for metric in metrics],
                theta=metrics,
                fill="toself",
                name=row["Model"],
            )
        )

    figure.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )
        ),
        showlegend=True,
    )

    return _apply_standard_layout(
        figure,
        title="Model Performance Comparison (Radar)",
    )


def save_plot(
    figure: go.Figure,
    filename: str,
    output_directory: Path = PLOTS_DIR,
) -> Path:
    """
    Save a Plotly figure to disk.

    This is mainly used when generating screenshots
    for the README or exporting figures from notebooks.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Always save the robust interactive HTML version (requires no dynamic headless dependencies)
    html_filename = Path(filename).with_suffix(".html")
    html_path = output_directory / html_filename
    figure.write_html(str(html_path))

    # 2. Try to write the static image (png/jpg) for README screenshots using Kaleido
    image_path = output_directory / filename
    try:
        pio.write_image(
            figure,
            str(image_path),
            scale=2,
        )
    except Exception as err:
        # Kaleido is sometimes missing or fails on headless Windows systems. We log the issue but don't crash
        # because the HTML figure was successfully written.
        print(
            f"Warning: Could not save static image to {image_path}. Dynamic HTML plot was saved successfully. "
            f"Error details: {err}"
        )

    return image_path