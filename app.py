import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.cluster import DBSCAN
import plotly.express as px

from src.preprocessing import load_data, encode_features, split_features
from src.isolation_forest_model import train_isolation_forest, detect_anomalies
from src.autoencoder_model import build_autoencoder

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "KDDTest+.txt")

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
st.title("🚀 Network Anomaly Detection System")

# SIDEBAR
st.sidebar.header("⚙️ Controls")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Isolation Forest", "DBSCAN", "Autoencoder"]
)

threshold_percentile = st.sidebar.slider(
    "Anomaly Threshold (%) (for AE)",
    50, 99, 95
)

eps = st.sidebar.slider("DBSCAN eps", 0.5, 5.0, 3.0)
min_samples = st.sidebar.slider("DBSCAN min_samples", 5, 50, 10)

uploaded_file = st.sidebar.file_uploader("Upload Dataset (optional)", type=["csv", "txt"])

# LOAD DATA
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

# MODELS
@st.cache_resource
def get_if(X):
    return train_isolation_forest(X)

@st.cache_resource
def get_ae(input_dim, X, y):
    model = build_autoencoder(input_dim)
    normal = X[y == "normal"]
    model.fit(normal, normal, epochs=10, batch_size=256, verbose=0)
    return model

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "📈 Visualizations",
    "📄 Data",
    "⚔️ Model Comparison"
])

# TAB 1: DASHBOARD
with tab1:

    if model_choice == "Isolation Forest":
        model = get_if(X_scaled)
        pred = detect_anomalies(model, X_scaled)

        data["anomaly"] = (pred == -1).astype(int)
        data["score"] = -model.decision_function(X_scaled)

    elif model_choice == "DBSCAN":
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)

        data["anomaly"] = (labels == -1).astype(int)
        data["score"] = labels

    else:
        model = get_ae(X_scaled.shape[1], X_scaled, y)

        recon = model.predict(X_scaled)
        mse = np.mean((X_scaled - recon)**2, axis=1)

        threshold = np.percentile(mse, threshold_percentile)

        data["anomaly"] = (mse > threshold).astype(int)
        data["score"] = mse

    # ---------------- Metrics ----------------
    st.subheader("📊 Detection Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Samples", len(data))
    col2.metric("Anomalies", int(data["anomaly"].sum()))
    col3.metric("Normal", int((data["anomaly"] == 0).sum()))

    if "label" in data.columns:
        true = (y != "normal").astype(int)
        pred = data["anomaly"]

        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        f1 = f1_score(true, pred)

        col4, col5, col6 = st.columns(3)
        col4.metric("Precision", round(precision, 3))
        col5.metric("Recall", round(recall, 3))
        col6.metric("F1 Score", round(f1, 3))

        st.subheader("📋 Classification Report")
        report = classification_report(true, pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

# TAB 2: VISUALIZATIONS
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
        opacity=0.6
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Score Distribution")

    fig2 = px.histogram(
        data,
        x="score",
        color=data["anomaly"].astype(str),
        nbins=50
    )

    st.plotly_chart(fig2, use_container_width=True)

# TAB 3: DATA
with tab3:

    st.dataframe(data.head(50))

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Results",
        csv,
        "anomaly_results.csv",
        "text/csv"
    )

# TAB 4: MODEL COMPARISON
with tab4:

    st.subheader("⚔️ Model Comparison")

    true = (y != "normal").astype(int)

    # Isolation Forest
    if_model = get_if(X_scaled)
    if_pred = detect_anomalies(if_model, X_scaled)
    if_anomaly = (if_pred == -1).astype(int)

    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db_labels = db.fit_predict(X_scaled)
    db_anomaly = (db_labels == -1).astype(int)

    # Autoencoder
    ae_model = get_ae(X_scaled.shape[1], X_scaled, y)
    recon = ae_model.predict(X_scaled)
    mse = np.mean((X_scaled - recon)**2, axis=1)
    ae_threshold = np.percentile(mse, threshold_percentile)
    ae_anomaly = (mse > ae_threshold).astype(int)

    comparison = pd.DataFrame({
        "Model": ["Isolation Forest", "DBSCAN", "Autoencoder"],
        "F1 Score": [
            f1_score(true, if_anomaly),
            f1_score(true, db_anomaly),
            f1_score(true, ae_anomaly)
        ],
        "Precision": [
            precision_score(true, if_anomaly),
            precision_score(true, db_anomaly),
            precision_score(true, ae_anomaly)
        ],
        "Recall": [
            recall_score(true, if_anomaly),
            recall_score(true, db_anomaly),
            recall_score(true, ae_anomaly)
        ]
    })

    st.dataframe(comparison)

    fig = px.bar(
        comparison,
        x="Model",
        y=["F1 Score", "Precision", "Recall"],
        barmode="group"
    )

    st.plotly_chart(fig, use_container_width=True)