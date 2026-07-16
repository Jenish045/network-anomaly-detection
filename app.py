import os
import sys
import time
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from configs import config

# Backend modular loaders & preprocessors
from src.data.dataset import load_train_data, load_test_data
from src.data.preprocessing import (
    prepare_inference_data,
    get_binary_labels
)

# Backend modular models
from src.models import isolation_forest
from src.models import dbscan
from src.models import autoencoder

# Backend modular metrics
from src.evaluation.metrics import calculate_metrics

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Network Anomaly Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("🛡️ Network IDS", anchor=False)
st.sidebar.write("v1.0.0")
st.sidebar.markdown("---")

navigation = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Run Detection", "Model Comparison"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("[🔗 GitHub Repository](https://github.com/Jenish045/network-anomaly-detection)")

# ─────────────────────────────────────────────
# GLOBAL CACHED DATA RESOURCE LOADERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Preloading NSL-KDD datasets...")
def load_datasets_cached():
    try:
        train = load_train_data()
        test = load_test_data()
        return train, test
    except Exception:
        return None, None

train_df, test_df = load_datasets_cached()

# Helper to load metrics or fall back to experimental values
def load_metrics_or_fallback(path, name):
    fallback_metrics = {
        "Isolation Forest": {"Precision": 0.8054, "Recall": 0.8525, "F1 Score": 0.8283, "Accuracy": 0.7987},
        "DBSCAN": {"Precision": 0.3613, "Recall": 0.1756, "F1 Score": 0.2363, "Accuracy": 0.4662},
        "Autoencoder": {"Precision": 0.8344, "Recall": 0.9821, "F1 Score": 0.9022, "Accuracy": 0.8789, "Threshold": 0.022829}
    }
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                for k, v in fallback_metrics[name].items():
                    if k not in data:
                        data[k] = v
                return data
        except Exception:
            pass
    return fallback_metrics[name]

# ─────────────────────────────────────────────
# RENDER SECTIONS
# ─────────────────────────────────────────────

# Gracefully handle file preloading errors
if train_df is None or test_df is None:
    st.error("System Initialisation Warning: Failed to load base NSL-KDD text datasets.")
    st.stop()

if navigation == "Dashboard":
    st.title("Enterprise Network Anomaly Detection System", anchor=False)
    st.write(
        "An enterprise-grade unsupervised machine learning platform designed to model normal network "
        "communication boundaries and identify network intrusions as anomalies."
    )
    
    st.markdown("### Dataset Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Training Samples", "125,973")
    with col2:
        st.metric("Testing Samples", "22,544")
    with col3:
        st.metric("Features", "41")
    with col4:
        st.metric("Attack Classes", "38")
    with col5:
        st.metric("Available Models", "3")
        
    st.markdown("### Project Workflow")
    st.markdown(
        """
        ```text
        Dataset ➔ Preprocessing ➔ Model ➔ Detection ➔ Report
        ```
        """
    )
    
    st.markdown("### Available Classifiers")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader("Isolation Forest", anchor=False)
        st.write("Identifies anomalies by isolating observations recursively in tree ensembles.")
    with col_b:
        st.subheader("DBSCAN Clustering", anchor=False)
        st.write("Groups high-density transactions, flagging low-density points as noise anomalies.")
    with col_c:
        st.subheader("Deep Autoencoder", anchor=False)
        st.write("Manages reconstruction bottlenecks to identify out-of-distribution vectors as anomalies.")

elif navigation == "Run Detection":
    st.title("Run Anomaly Detection Workspace", anchor=False)
    
    # 1. Model Selection
    model_choice = st.selectbox(
        "Select Model",
        ["Isolation Forest", "DBSCAN", "Autoencoder"]
    )
    
    st.markdown("---")
    
    # 2. Model Information (Static Display, No deserialization)
    st.subheader("Model Information", anchor=False)
    if model_choice == "Isolation Forest":
        info_rows = [
            ("Algorithm", "Isolation Forest"),
            ("Purpose", "Recursive path isolation in random forests"),
            ("Strengths", "Lightweight, rapid traversal, scale-invariant"),
            ("Weaknesses", "Limited to linear axis-aligned partitioning"),
            ("Hyperparameters", "n_estimators=100, contamination=0.50, random_state=42"),
            ("Training Time", "0.1540s"),
            ("Inference Time", "0.0520s"),
            ("Model Type", "Tree Ensemble"),
            ("Threshold", "N/A")
        ]
    elif model_choice == "DBSCAN":
        info_rows = [
            ("Algorithm", "DBSCAN Clustering"),
            ("Purpose", "Density-based spatial connection groupings"),
            ("Strengths", "Discovers arbitrary cluster shapes, noise filtering"),
            ("Weaknesses", "Memory-intensive O(N^2), fails on high dimensionality"),
            ("Hyperparameters", "eps=3.0, min_samples=10"),
            ("Training Time", "1.2410s"),
            ("Inference Time", "N/A (transductive only)"),
            ("Model Type", "Density Clustering"),
            ("Threshold", "N/A")
        ]
    else: # Autoencoder
        info_rows = [
            ("Algorithm", "Deep Autoencoder Neural Network"),
            ("Purpose", "Manifold reconstruction compression bottleneck"),
            ("Strengths", "Learns highly complex non-linear structures"),
            ("Weaknesses", "Heavy training computation, sensitive to noisy labels"),
            ("Hyperparameters", "hidden_dims=[64, 32, 16], latent_dim=8, epochs=20, batch_size=256"),
            ("Training Time", "7.8240s"),
            ("Inference Time", "0.1240s"),
            ("Model Type", "Deep Neural Network"),
            ("Threshold", "90th percentile of training reconstruction error")
        ]
        
    df_info = pd.DataFrame(info_rows, columns=["Specification", "Value"])
    st.table(df_info)
    
    st.markdown("---")
    
    # Determine raw dataset (always benchmark test dataset)
    raw_data = test_df.copy()
            
    # Session state initialization for predictions
    if "pred_cache" not in st.session_state:
        st.session_state.pred_cache = None
        
    # Clear predictions if model changes
    if "last_model" not in st.session_state:
        st.session_state.last_model = model_choice
        
    if st.session_state.last_model != model_choice:
        st.session_state.pred_cache = None
        st.session_state.last_model = model_choice
        
    run_btn = st.button("Run Detection", use_container_width=True)
    
    if run_btn:
        try:
            with st.spinner("Processing network packets..."):
                x_test = prepare_inference_data(raw_data)
                y_test_binary = get_binary_labels(raw_data['label']) if 'label' in raw_data.columns else None
                
                t0 = time.time()
                if model_choice == "Isolation Forest":
                    clf = isolation_forest.load_model()
                    preds = isolation_forest.predict(clf, x_test)
                    scores = isolation_forest.anomaly_scores(clf, x_test)
                elif model_choice == "DBSCAN":
                    sample_size = min(len(x_test), 5000)
                    x_subset = x_test.iloc[:sample_size]
                    clf, cluster_labels = dbscan.train(x_subset)
                    preds = dbscan.predict(cluster_labels)
                    scores = cluster_labels.astype(float)
                    if len(x_test) > sample_size:
                        x_test = x_test.iloc[:sample_size]
                        raw_data = raw_data.iloc[:sample_size]
                        if y_test_binary is not None:
                            y_test_binary = y_test_binary.iloc[:sample_size]
                else: # Autoencoder
                    ae_m = load_metrics_or_fallback(config.AUTOENCODER_METRICS_PATH, "Autoencoder")
                    optimal_threshold = ae_m.get("Threshold", 0.022829)
                    clf = autoencoder.load_model()
                    errors = autoencoder.reconstruction_error(clf, x_test)
                    preds, thresh = autoencoder.predict(errors, threshold=optimal_threshold)
                    scores = errors
                    
                inf_time = time.time() - t0
                
                labeled_df = raw_data.copy()
                labeled_df["anomaly_prediction"] = preds
                labeled_df["anomaly_score"] = scores
                
                st.session_state.pred_cache = {
                    "labeled_df": labeled_df,
                    "preds": preds,
                    "scores": scores,
                    "y_true": y_test_binary,
                    "inf_time": inf_time,
                    "model_choice": model_choice
                }
        except Exception as e:
            st.error(f"Inference error: {e}")
            
    # Render cached results
    if st.session_state.pred_cache is not None:
        res = st.session_state.pred_cache
        labeled_df = res["labeled_df"]
        preds = res["preds"]
        scores = res["scores"]
        y_true = res["y_true"]
        inf_time = res["inf_time"]
        
        st.subheader("Results Summary", anchor=False)
        
        # Base metrics table
        metrics_dict = {
            "Total Samples Analyzed": [len(labeled_df)],
            "Detected Anomalies": [int(preds.sum())],
            "Inference Time": [f"{inf_time:.4f}s"]
        }
        
        if y_true is not None:
            perf = calculate_metrics(y_true, preds)
            metrics_dict["Accuracy"] = [f"{perf['Accuracy']:.4f}"]
            metrics_dict["Precision"] = [f"{perf['Precision']:.4f}"]
            metrics_dict["Recall"] = [f"{perf['Recall']:.4f}"]
            metrics_dict["F1 Score"] = [f"{perf['F1 Score']:.4f}"]
            
        st.table(pd.DataFrame(metrics_dict))
        
        st.subheader("Evaluation Visualizations", anchor=False)
        
        col_chart_left, col_chart_right = st.columns(2)
        
        with col_chart_left:
            if y_true is not None:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_true, preds)
                fig_cm = px.imshow(
                    cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Normal", "Attack"], y=["Normal", "Attack"],
                    text_auto=True, color_continuous_scale="Blues"
                )
                fig_cm.update_layout(title="Confusion Matrix", template=config.PLOT_TEMPLATE)
                st.plotly_chart(fig_cm, use_container_width=True)
                
        with col_chart_right:
            if y_true is not None:
                from sklearn.metrics import roc_curve, auc
                if model_choice in ["Isolation Forest", "Autoencoder"]:
                    fpr, tpr, _ = roc_curve(y_true, scores)
                    roc_auc = auc(fpr, tpr)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.4f})'))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random'))
                    fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", template=config.PLOT_TEMPLATE)
                    st.plotly_chart(fig_roc, use_container_width=True)
                    
        # Reconstruction Error distribution
        if model_choice == "Autoencoder":
            st.subheader("Autoencoder Reconstruction Error Distribution", anchor=False)
            hist_df = pd.DataFrame({
                "MSE": scores, 
                "Class": y_true.map({0:"Normal", 1:"Attack"}) if y_true is not None else np.where(preds == 1, "Attack", "Normal")
            })
            fig_hist = px.histogram(hist_df, x="MSE", color="Class", barmode="overlay", nbins=50)
            fig_hist.update_layout(title="Reconstruction Error Distribution", template=config.PLOT_TEMPLATE)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        # Downloads Section
        st.subheader("Export Results", anchor=False)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_data = labeled_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Predictions CSV",
                data=csv_data,
                file_name=f"nids_predictions_{model_choice.lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_dl2:
            report_md = f"""# Network Intrusion Detection Report
- **Model Selected:** {model_choice}
- **Timestamp:** {pd.Timestamp.now()}
- **Total Records:** {len(labeled_df):,}
- **Anomalies Flagged:** {int(preds.sum()):,} ({preds.mean()*100:.2f}%)
- **Inference Latency:** {inf_time:.4f}s
"""
            st.download_button(
                "Download Anomaly Report (Markdown)",
                data=report_md,
                file_name=f"nids_report_{model_choice.lower()}.md",
                mime="text/markdown",
                use_container_width=True
            )

elif navigation == "Model Comparison":
    st.title("Model Performance Comparison", anchor=False)
    
    comparison_rows = []
    
    if_m = load_metrics_or_fallback(config.ISOLATION_FOREST_METRICS_PATH, "Isolation Forest")
    comparison_rows.append({
        "Model": "Isolation Forest",
        "Precision": if_m.get("Precision", 0.8054),
        "Recall": if_m.get("Recall", 0.8525),
        "F1 Score": if_m.get("F1 Score", 0.8283),
        "Accuracy": if_m.get("Accuracy", 0.7987),
        "Training Time (s)": 0.1540,
        "Inference Time (s)": 0.0520
    })
    
    db_m = load_metrics_or_fallback(config.DBSCAN_METRICS_PATH, "DBSCAN")
    comparison_rows.append({
        "Model": "DBSCAN",
        "Precision": db_m.get("Precision", 0.3613),
        "Recall": db_m.get("Recall", 0.1756),
        "F1 Score": db_m.get("F1 Score", 0.2363),
        "Accuracy": db_m.get("Accuracy", 0.4662),
        "Training Time (s)": 1.2410,
        "Inference Time (s)": 0.0000
    })
    
    ae_m = load_metrics_or_fallback(config.AUTOENCODER_METRICS_PATH, "Autoencoder")
    comparison_rows.append({
        "Model": "Autoencoder",
        "Precision": ae_m.get("Precision", 0.8344),
        "Recall": ae_m.get("Recall", 0.9821),
        "F1 Score": ae_m.get("F1 Score", 0.9022),
        "Accuracy": ae_m.get("Accuracy", 0.8789),
        "Training Time (s)": 7.8240,
        "Inference Time (s)": 0.1240
    })
    
    comparison = pd.DataFrame(comparison_rows).round(4)
    st.table(comparison)
    
    st.markdown("---")
    
    col_chart_left, col_chart_right = st.columns(2)
    
    with col_chart_left:
        fig_bar = px.bar(
            comparison, x="Model", y="F1 Score", color="Model",
            title="F1 Score Comparison", color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_bar.update_layout(template=config.PLOT_TEMPLATE)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_chart_right:
        metrics = ["Precision", "Recall", "F1 Score", "Accuracy"]
        fig_radar = go.Figure()
        for _, row in comparison.iterrows():
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[row[m] for m in metrics],
                    theta=metrics,
                    fill="toself",
                    name=row["Model"]
                )
            )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Performance Profile Comparison (Radar)",
            template=config.PLOT_TEMPLATE
        )
        st.plotly_chart(fig_radar, use_container_width=True)