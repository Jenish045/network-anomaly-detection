import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, f1_score
import plotly.express as px

from src.preprocessing import load_data, encode_features, split_features
from src.isolation_forest_model import train_isolation_forest, detect_anomalies
from src.autoencoder_model import build_autoencoder

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "KDDTest+.txt")
# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")

st.title("🚀 Network Anomaly Detection System")

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.header("⚙️ Controls")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Isolation Forest", "Autoencoder"]
)

threshold_percentile = st.sidebar.slider(
    "Anomaly Threshold (%)",
    80, 99, 95
)

uploaded_file = st.sidebar.file_uploader("Upload Dataset (optional)", type=["csv", "txt"])

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_and_process(file):
    if file:
        data = pd.read_csv(file)
    else:
        data = load_data(DATA_PATH)

    data = encode_features(data)
    X, y = split_features(data)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return data, X_scaled, y

data, X_scaled, y = load_and_process(uploaded_file)

# ------------------------------
# MODELS (CACHED)
# ------------------------------
@st.cache_resource
def get_if(X):
    return train_isolation_forest(X)

@st.cache_resource
def get_ae(input_dim, X, y):
    model = build_autoencoder(input_dim)
    normal = X[y == "normal"]
    model.fit(normal, normal, epochs=10, batch_size=256, verbose=0)
    return model

# ------------------------------
# TABS
# ------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "📈 Visualizations",
    "📄 Data",
    "⚔️ Model Comparison"
])

# ==============================
# TAB 1: DASHBOARD
# ==============================
with tab1:

    if model_choice == "Isolation Forest":
        model = get_if(X_scaled)
        pred = detect_anomalies(model, X_scaled)

        data["anomaly"] = (pred == -1).astype(int)
        data["score"] = -model.decision_function(X_scaled)

    else:
        model = get_ae(X_scaled.shape[1], X_scaled, y)

        recon = model.predict(X_scaled)
        mse = np.mean((X_scaled - recon)**2, axis=1)

        threshold = np.percentile(mse, threshold_percentile)

        data["anomaly"] = (mse > threshold).astype(int)
        data["score"] = mse

    # Metrics
    st.subheader("📊 Detection Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Samples", len(data))
    col2.metric("Anomalies", int(data["anomaly"].sum()))
    col3.metric("Normal", int((data["anomaly"] == 0).sum()))

    # Classification report
    if "label" in data.columns:
        true = (y != "normal").astype(int)
        pred = data["anomaly"]

        report = classification_report(true, pred, output_dict=True)
        st.subheader("📋 Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

# ==============================
# TAB 2: VISUALIZATIONS
# ==============================
with tab2:

    st.subheader("📈 PCA Projection")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_plot = pd.DataFrame({
        "PC1": X_pca[:,0],
        "PC2": X_pca[:,1],
        "Anomaly": data["anomaly"]
    })

    fig = px.scatter(
        df_plot,
        x="PC1",
        y="PC2",
        color=df_plot["Anomaly"].astype(str),
        opacity=0.6,
        title="Anomaly Visualization (PCA)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Score distribution
    st.subheader("📊 Anomaly Score Distribution")

    fig2 = px.histogram(
        data,
        x="score",
        color=data["anomaly"].astype(str),
        nbins=50
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Attack insights
    if "label" in data.columns:
        st.subheader("🚨 Attack Type Insights")

        attacks = data[data["anomaly"] == 1]

        attack_counts = attacks["label"].value_counts().head(10).reset_index()
        attack_counts.columns = ["Attack Type", "Count"]

        fig3 = px.bar(attack_counts, x="Attack Type", y="Count")

        st.plotly_chart(fig3, use_container_width=True)

        # False positives
        false_pos = data[
            (data["anomaly"] == 1) & (data["label"] == "normal")
        ]

        st.metric("⚠️ False Positives", len(false_pos))

# ==============================
# TAB 3: DATA
# ==============================
with tab3:

    st.subheader("📄 Dataset Preview")

    st.dataframe(data.head(50))

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Results",
        csv,
        "anomaly_results.csv",
        "text/csv"
    )

# ==============================
# TAB 4: MODEL COMPARISON
# ==============================
with tab4:

    st.subheader("⚔️ Model Comparison")

    # Isolation Forest
    if_model = get_if(X_scaled)
    if_pred = detect_anomalies(if_model, X_scaled)
    if_anomaly = (if_pred == -1).astype(int)

    # Autoencoder
    ae_model = get_ae(X_scaled.shape[1], X_scaled, y)
    recon = ae_model.predict(X_scaled)
    mse = np.mean((X_scaled - recon)**2, axis=1)
    ae_threshold = np.percentile(mse, threshold_percentile)
    ae_anomaly = (mse > ae_threshold).astype(int)

    true = (y != "normal").astype(int)

    comparison = pd.DataFrame({
        "Model": ["Isolation Forest", "Autoencoder"],
        "Accuracy": [
            accuracy_score(true, if_anomaly),
            accuracy_score(true, ae_anomaly)
        ],
        "F1 Score": [
            f1_score(true, if_anomaly),
            f1_score(true, ae_anomaly)
        ]
    })

    st.dataframe(comparison)

    fig = px.bar(comparison, x="Model", y=["Accuracy", "F1 Score"], barmode="group")

    st.plotly_chart(fig, use_container_width=True)