import streamlit as st
import pandas as pd
import numpy as np
import json
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix
)
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import load_data, encode_features, split_features
from src.isolation_forest_model import train_isolation_forest, detect_anomalies
from src.autoencoder_model import build_autoencoder

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "KDDTest+.txt")
RES_DIR   = os.path.join(BASE_DIR, "results")

st.set_page_config(
    page_title="Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("🛡️ Controls")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Isolation Forest", "DBSCAN", "Autoencoder"],
    help="Choose the anomaly detection model to run."
)

st.sidebar.markdown("#### Autoencoder Settings")
ae_percentile = st.sidebar.slider(
    "Reconstruction Error Threshold (%)",
    min_value=50, max_value=99, value=90,
    help="Top X% of reconstruction errors flagged as anomalies."
)

st.sidebar.markdown("#### DBSCAN Settings")
eps         = st.sidebar.slider("eps",         0.5,  5.0, 3.0, step=0.1)
min_samples = st.sidebar.slider("min_samples",   5,   50,  10)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    "Upload Custom Dataset (optional)",
    type=["csv", "txt"],
    help="Upload your own NSL-KDD formatted file."
)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & preprocessing data...")
def load_and_process(file):
    if file:
        data = pd.read_csv(file)
    else:
        data = load_data(DATA_PATH)

    data     = encode_features(data)
    X, y     = split_features(data)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return data, X_scaled, y

data, X_scaled, y = load_and_process(uploaded_file)
true_labels = (y != "normal").astype(int)

# ─────────────────────────────────────────────
# MODEL RUNNERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training Isolation Forest...")
def get_isolation_forest(X):
    return train_isolation_forest(X)

@st.cache_resource(show_spinner="Training Autoencoder (this may take a minute)...")
def get_autoencoder(input_dim, _X, _y):
    model  = build_autoencoder(input_dim)
    normal = _X[_y == "normal"]
    model.fit(normal, normal, epochs=10, batch_size=256, verbose=0)
    return model

@st.cache_data(show_spinner="Running DBSCAN...")
def run_dbscan(X, _eps, _min_samples):
    return DBSCAN(eps=_eps, min_samples=_min_samples).fit_predict(X)

def get_predictions(model_name):
    """Returns (anomaly_labels 0/1, scores array)."""
    if model_name == "Isolation Forest":
        model  = get_isolation_forest(X_scaled)
        raw    = detect_anomalies(model, X_scaled)
        scores = -model.decision_function(X_scaled)
        preds  = (raw == -1).astype(int)

    elif model_name == "DBSCAN":
        labels = run_dbscan(X_scaled, eps, min_samples)
        scores = labels.astype(float)
        preds  = (labels == -1).astype(int)

    else:  # Autoencoder
        model  = get_autoencoder(X_scaled.shape[1], X_scaled, y)
        recon  = model.predict(X_scaled, verbose=0)
        scores = np.mean((X_scaled - recon) ** 2, axis=1)
        thresh = np.percentile(scores, ae_percentile)
        preds  = (scores > thresh).astype(int)

    return preds, scores

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🛡️ Network Anomaly Detection System")
st.caption("Unsupervised anomaly detection on NSL-KDD using Isolation Forest, DBSCAN, and Autoencoder.")
st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "📈 Visualizations",
    "📄 Data Explorer",
    "⚔️ Model Comparison"
])

# ══════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════
with tab1:
    st.subheader(f"Model: {model_choice}")

    with st.spinner(f"Running {model_choice}..."):
        preds, scores = get_predictions(model_choice)

    data["anomaly"] = preds
    data["score"]   = scores

    # ── Summary Metrics ──
    st.markdown("#### 🔢 Detection Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples",  f"{len(data):,}")
    c2.metric("Anomalies",      f"{int(preds.sum()):,}")
    c3.metric("Normal",         f"{int((preds == 0).sum()):,}")
    c4.metric("Attack Rate",    f"{preds.mean()*100:.1f}%")

    # ── Performance Metrics ──
    st.markdown("#### 🎯 Performance Metrics")

    prec = precision_score(true_labels, preds, zero_division=0)
    rec  = recall_score(true_labels, preds, zero_division=0)
    f1   = f1_score(true_labels, preds, zero_division=0)
    acc  = accuracy_score(true_labels, preds)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{prec:.3f}")
    m2.metric("Recall",    f"{rec:.3f}")
    m3.metric("F1 Score",  f"{f1:.3f}")
    m4.metric("Accuracy",  f"{acc:.3f}")

    # ── Classification Report ──
    st.markdown("#### 📋 Classification Report")
    report_dict = classification_report(
        true_labels, preds,
        target_names=["Normal", "Attack"],
        zero_division=0,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

    # ── Confusion Matrix ──
    st.markdown("#### 🟦 Confusion Matrix")
    cm = confusion_matrix(true_labels, preds)
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Normal", "Attack"],
        y=["Normal", "Attack"],
        text_auto=True,
        color_continuous_scale="Blues"
    )
    fig_cm.update_layout(height=350)
    st.plotly_chart(fig_cm, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — VISUALIZATIONS
# ══════════════════════════════════════════════
with tab2:

    # ── PCA Projection ──
    st.markdown("#### 🔵 PCA Projection (2D)")

    pca    = PCA(n_components=2, random_state=42)
    X_pca  = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_.sum() * 100

    df_pca = pd.DataFrame({
        "PC1":     X_pca[:, 0],
        "PC2":     X_pca[:, 1],
        "Label":   np.where(data["anomaly"] == 1, "Anomaly", "Normal"),
        "True":    np.where(true_labels == 1, "Attack", "Normal"),
        "Score":   scores
    })

    fig_pca = px.scatter(
        df_pca, x="PC1", y="PC2",
        color="Label",
        color_discrete_map={"Normal": "#3b82f6", "Anomaly": "#ef4444"},
        opacity=0.5,
        title=f"PCA Projection — {var_explained:.1f}% variance explained",
        hover_data={"Score": ":.4f", "True": True}
    )
    fig_pca.update_traces(marker=dict(size=3))
    fig_pca.update_layout(height=500)
    st.plotly_chart(fig_pca, use_container_width=True)

    # ── Score Distribution ──
    st.markdown("#### 📊 Anomaly Score Distribution")

    df_scores = pd.DataFrame({
        "Score": scores,
        "Class": np.where(true_labels == 1, "Attack (True)", "Normal (True)")
    })

    fig_dist = px.histogram(
        df_scores,
        x="Score",
        color="Class",
        nbins=60,
        barmode="overlay",
        opacity=0.65,
        color_discrete_map={
            "Attack (True)": "#ef4444",
            "Normal (True)": "#3b82f6"
        },
        title="Score Distribution by True Label"
    )
    fig_dist.update_layout(height=400)
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── Attack Type Breakdown (if detections exist) ──
    if "label" in data.columns and preds.sum() > 0:
        st.markdown("#### 🔍 Detected Attack Types")

        detected = data[(data["anomaly"] == 1) & (y != "normal")]
        if len(detected) > 0:
            top_attacks = detected["label"].value_counts().head(10).reset_index()
            top_attacks.columns = ["Attack Type", "Count"]

            fig_atk = px.bar(
                top_attacks,
                x="Attack Type", y="Count",
                color="Count",
                color_continuous_scale="Reds",
                title="Top 10 Detected Attack Types"
            )
            fig_atk.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_atk, use_container_width=True)
        else:
            st.info("No true attacks detected by the current model/threshold.")

# ══════════════════════════════════════════════
# TAB 3 — DATA EXPLORER
# ══════════════════════════════════════════════
with tab3:
    st.markdown("#### 📄 Sample Results")

    display_df = data.copy()
    display_df["true_label"] = y.values
    display_df["prediction"] = np.where(data["anomaly"] == 1, "Anomaly", "Normal")

    col_filter = st.selectbox(
        "Filter by Prediction",
        ["All", "Anomaly", "Normal"]
    )

    if col_filter != "All":
        display_df = display_df[display_df["prediction"] == col_filter]

    st.dataframe(display_df.head(100), use_container_width=True)
    st.caption(f"Showing up to 100 rows of {len(display_df):,} filtered results.")

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Full Results as CSV",
        data=csv,
        file_name=f"anomaly_results_{model_choice.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

# ══════════════════════════════════════════════
# TAB 4 — MODEL COMPARISON
# ══════════════════════════════════════════════
with tab4:
    st.markdown("#### ⚔️ Model Comparison")

    # Try loading from saved JSON files first
    json_if   = os.path.join(RES_DIR, "if_metrics.json")
    json_db   = os.path.join(RES_DIR, "dbscan_metrics.json")
    json_ae   = os.path.join(RES_DIR, "ae_metrics.json")

    use_saved = all(os.path.exists(p) for p in [json_if, json_db, json_ae])

    if use_saved:
        with open(json_if)  as f: if_m  = json.load(f)
        with open(json_db)  as f: db_m  = json.load(f)
        with open(json_ae)  as f: ae_m  = json.load(f)

        comparison = pd.DataFrame([
            {"Model": "Isolation Forest", **if_m},
            {"Model": "DBSCAN",           **db_m},
            {"Model": "Autoencoder",      **ae_m}
        ])
        st.success("✅ Loaded metrics from saved results files.")

    else:
        st.warning("⚠️ Saved metrics not found. Computing live (this may take a while)...")

        with st.spinner("Running all three models for comparison..."):
            # Isolation Forest
            if_model  = get_isolation_forest(X_scaled)
            if_raw    = detect_anomalies(if_model, X_scaled)
            if_pred   = (if_raw == -1).astype(int)

            # DBSCAN
            db_labels = run_dbscan(X_scaled, eps, min_samples)
            db_pred   = (db_labels == -1).astype(int)

            # Autoencoder
            ae_model  = get_autoencoder(X_scaled.shape[1], X_scaled, y)
            ae_recon  = ae_model.predict(X_scaled, verbose=0)
            ae_mse    = np.mean((X_scaled - ae_recon) ** 2, axis=1)
            ae_pred   = (ae_mse > np.percentile(ae_mse, ae_percentile)).astype(int)

        def metrics_row(name, pred):
            return {
                "Model":     name,
                "Precision": round(precision_score(true_labels, pred, zero_division=0), 2),
                "Recall":    round(recall_score(true_labels, pred, zero_division=0), 2),
                "F1-Score":  round(f1_score(true_labels, pred, zero_division=0), 2),
                "Accuracy":  round(accuracy_score(true_labels, pred), 2),
            }

        comparison = pd.DataFrame([
            metrics_row("Isolation Forest", if_pred),
            metrics_row("DBSCAN",           db_pred),
            metrics_row("Autoencoder",      ae_pred),
        ])

    # ── Table ──
    st.dataframe(
        comparison.style.highlight_max(
            subset=["Precision", "Recall", "F1-Score", "Accuracy"],
            color="#bbf7d0"
        ),
        use_container_width=True
    )

    # ── Bar Chart ──
    metrics_long = comparison.melt(
        id_vars="Model",
        value_vars=["Precision", "Recall", "F1-Score", "Accuracy"],
        var_name="Metric",
        value_name="Score"
    )

    fig_cmp = px.bar(
        metrics_long,
        x="Metric", y="Score",
        color="Model",
        barmode="group",
        color_discrete_sequence=["#3b82f6", "#f59e0b", "#10b981"],
        title="Model Performance Comparison",
        range_y=[0, 1.05]
    )
    fig_cmp.update_layout(height=450)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Radar Chart ──
    st.markdown("#### 🕸️ Radar Chart")
    metrics_cols = ["Precision", "Recall", "F1-Score", "Accuracy"]
    colors       = ["#3b82f6", "#f59e0b", "#10b981"]

    fig_radar = go.Figure()
    for i, row in comparison.iterrows():
        vals = [row[m] for m in metrics_cols] + [row[metrics_cols[0]]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=metrics_cols + [metrics_cols[0]],
            fill="toself",
            name=row["Model"],
            line_color=colors[i],
            opacity=0.6
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=450
    )
    st.plotly_chart(fig_radar, use_container_width=True)