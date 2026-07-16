"""
Reusable Streamlit components.

Keeping these components separate helps keep app.py
focused on workflow instead of UI implementation.
"""

from typing import Any
import streamlit as st
import pandas as pd


def section_header(
    title: str,
    description: str | None = None,
) -> None:
    """
    Display a consistent, styled section header.

    Use this at the beginning of major page regions (e.g. data loader, results, evaluation).
    """

    st.markdown(
        f"<div class='section-title'>{title}</div>",
        unsafe_allow_html=True,
    )

    if description:
        st.markdown(
            f"<div class='section-description'>{description}</div>",
            unsafe_allow_html=True,
        )


def metric_card(
    label: str,
    value: Any,
) -> None:
    """
    Display a metric inside a styled card.

    Ideal for displaying core counts, prediction rates, and scores side-by-side.
    """

    st.markdown(
        f"""
        <div class="metric-card">
            <h4>{label}</h4>
            <h2>{value}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_box(
    message: str,
    title: str | None = None,
) -> None:
    """
    Display a styled informational message box.

    Use to provide context on dataset splits, sizing, or parameters.
    """
    header_html = f"<strong>{title}</strong><br>" if title else ""
    st.markdown(
        f'<div class="info-box">{header_html}{message}</div>',
        unsafe_allow_html=True,
    )


def success_box(
    message: str,
    title: str | None = None,
) -> None:
    """
    Display a styled success message box.

    Use for successful loading, training completion, or normal transaction checks.
    """
    header_html = f"<strong>{title}</strong><br>" if title else ""
    st.markdown(
        f'<div class="success-box">{header_html}{message}</div>',
        unsafe_allow_html=True,
    )


def warning_box(
    message: str,
    title: str | None = None,
) -> None:
    """
    Display a styled warning message box.

    Use to caution users about missing model configurations, high anomalies rates, or empty datasets.
    """
    header_html = f"<strong>{title}</strong><br>" if title else ""
    st.markdown(
        f'<div class="warning-box">{header_html}{message}</div>',
        unsafe_allow_html=True,
    )


def error_box(
    message: str,
    title: str | None = None,
) -> None:
    """
    Display a styled error message box.

    Use to print validation errors, shape mismatches, or file loading failures.
    """
    header_html = f"<strong>{title}</strong><br>" if title else ""
    st.markdown(
        f'<div class="error-box">{header_html}{message}</div>',
        unsafe_allow_html=True,
    )


def engineering_note(
    note: str,
    title: str = "Developer Engineering Note",
) -> None:
    """
    Display a technical note explaining engineering trade-offs or choices (such as model shapes or seeds).
    """
    st.markdown(
        f'<div class="engineering-note"><strong>🛠️ {title}</strong><br>{note}</div>',
        unsafe_allow_html=True,
    )


def model_summary_card(
    metadata: dict[str, Any],
) -> None:
    """
    Display a structured model metadata summary card.

    Includes details like training/inference times, hyperparameters, and location.
    """
    hparams_str = ""
    hparams = metadata.get("Hyperparameters", {})
    if isinstance(hparams, dict):
        hparams_str = ", ".join(f"{k}: {v}" for k, v in hparams.items())
    else:
        hparams_str = str(hparams)

    st.markdown(
        f"""
        <div class="metric-card" style="margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #0f172a; font-size: 1.25rem;">📋 {metadata.get('Model Name', 'Unknown Model')} Summary</h3>
            <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 12px 0;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 8px 0; font-weight: 600; color: #475569;">Training Time:</td>
                    <td style="padding: 8px 0; text-align: right; color: #0f172a;">{metadata.get('Training Time', 0.0):.4f}s</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 8px 0; font-weight: 600; color: #475569;">Inference Time:</td>
                    <td style="padding: 8px 0; text-align: right; color: #0f172a;">{metadata.get('Inference Time', 0.0):.4f}s</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 8px 0; font-weight: 600; color: #475569;">Input Features:</td>
                    <td style="padding: 8px 0; text-align: right; color: #0f172a;">{metadata.get('Number of Features', 'N/A')}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 8px 0; font-weight: 600; color: #475569;">Hyperparameters:</td>
                    <td style="padding: 8px 0; text-align: right; color: #0f172a; max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{hparams_str}">{hparams_str}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 8px 0; font-weight: 600; color: #475569;">File Location:</td>
                    <td style="padding: 8px 0; text-align: right; color: #0f172a; font-family: monospace; font-size: 0.75rem; word-break: break-all;">{metadata.get('Model File Location', 'N/A')}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; font-weight: 600; color: #475569;">Version:</td>
                    <td style="padding: 8px 0; text-align: right; color: #0f172a;">{metadata.get('Version', '1.0.0')}</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def comparison_table_view(
    df: pd.DataFrame,
) -> None:
    """
    Render a formatted, color-coded model comparison table inside Streamlit.
    """
    # Color-code F1 Score and Accuracy columns to quickly highlight the best-performing models.
    st.dataframe(
        df.style.background_gradient(cmap="Blues", subset=["F1 Score", "Accuracy"])
        .format(precision=4, subset=["Precision", "Recall", "F1 Score", "Accuracy"])
    )


def expandable_section(
    title: str,
    content: str,
) -> None:
    """
    Display additional details or walkthroughs inside an expandable section.
    """
    with st.expander(title):
        st.markdown(content)