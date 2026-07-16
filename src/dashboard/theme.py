"""
Shared styling utilities for the Streamlit dashboard.

Keeping the styling here avoids repeating CSS
inside app.py.
"""

import streamlit as st


def apply_theme() -> None:
    """
    Apply custom styling to the dashboard.
    """

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');

        /* Standardize typography for a premium design layout */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }

        /* Premium metric card styling with micro-interactions */
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 16px;
            padding: 22px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
        }

        .metric-card h4 {
            margin: 0 0 8px 0;
            color: #64748b;
            font-size: 0.95rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-card h2 {
            margin: 0;
            color: #0f172a;
            font-size: 2rem;
            font-weight: 700;
        }

        /* Section titles with bottom highlight line */
        .section-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.7rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 24px;
            margin-bottom: 8px;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 4px;
            display: inline-block;
        }

        .section-description {
            color: #475569;
            font-size: 0.95rem;
            margin-bottom: 22px;
        }

        /* Status and message boxes */
        .info-box {
            background: #eff6ff;
            color: #1e40af;
            border-left: 5px solid #3b82f6;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            font-size: 0.95rem;
        }

        .success-box {
            background: #f0fdf4;
            color: #166534;
            border-left: 5px solid #22c55e;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            font-size: 0.95rem;
        }

        .warning-box {
            background: #fffbec;
            color: #854d0e;
            border-left: 5px solid #eab308;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            font-size: 0.95rem;
        }

        .error-box {
            background: #fef2f2;
            color: #991b1b;
            border-left: 5px solid #ef4444;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            font-size: 0.95rem;
        }

        /* Technical developer engineering notes formatting */
        .engineering-note {
            background: #f8fafc;
            border: 1px dashed #cbd5e1;
            color: #334155;
            padding: 16px;
            border-radius: 12px;
            font-family: monospace;
            font-size: 0.85rem;
            margin-top: 12px;
            margin-bottom: 12px;
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )