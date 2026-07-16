# Enterprise Network Anomaly Detection System

Unsupervised network intrusion detection using Isolation Forest, DBSCAN, and Autoencoders with an interactive Streamlit dashboard.

---

## Project Overview

- **Problem Statement**: Modern network environments face frequent and evolving security threats. Traditional signature-based detection systems fail to identify zero-day attacks and novel intrusion strategies.
- **Objective**: Establish baseline models of normal network transactions using unsupervised machine learning and identify malicious packets as anomalies that deviate from this baseline.
- **Importance**: Anomaly detection identifies suspicious activities without relying on pre-configured signature databases, catching network threats as soon as they manifest.
- **Model Selection**: In production networks, malicious traffic labels are rarely available in real-time. Unsupervised models learn the distribution of normal traffic patterns automatically.

---

## Features

- **Modular Backend**: Decoupled preprocessing, model definitions, metrics, and visualization utilities located under the src directory.
- **Interactive Streamlit Dashboard**: Multi-tab interface featuring dataset statistics, model performance comparison profiles, and analytics.
- **Three complementary anomaly detection models**:
  - Isolation Forest
  - DBSCAN
  - Deep Autoencoder
- **Interactive Plotly Visualizations**: 2D PCA boundary scatter plots, score distributions, and performance radar charts.
- **Model Comparison**: Side-by-side performance metrics comparison (precision, recall, F1, accuracy, and inference latencies).
- **Downloadable Predictions**: Export labeled predictions as CSV files and Markdown reports.

---

## Repository Structure

```text
network-anomaly-detection/
│
├── assets/
│   ├── architecture/
│   ├── banners/
│   ├── icons/
│   └── screenshots/
│
├── configs/
│   └── config.py                   # Centralized paths, seeds, and hyperparameters
│
├── data/
│   ├── raw/
│   │   ├── KDDTest+.txt            # NSL-KDD testing split
│   │   └── KDDTrain+.txt           # NSL-KDD training split
│   └── processed/
│       ├── label_encoders.joblib   # Categorical encoders
│       ├── scaler.joblib           # Fitted StandardScaler offsets
│       └── x_scaled.npy            # Pre-scaled training vectors cache
│
├── models/
│   ├── metrics/
│   │   ├── autoencoder_metrics.json
│   │   ├── dbscan_metrics.json
│   │   └── isolation_forest_metrics.json
│   └── trained/
│       ├── autoencoder.keras       # Trained Autoencoder neural net weights
│       └── isolation_forest.joblib # Serialized Isolation Forest model
│
├── notebooks/
│   ├── 01_dataset_loading.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_isolation_forest.ipynb
│   ├── 05_dbscan.ipynb
│   ├── 06_autoencoder.ipynb
│   └── 07_model_comparison.ipynb
│
├── src/
│   ├── dashboard/
│   │   ├── components.py           # Reusable UI cards and alerts
│   │   └── theme.py                # Visual layout configuration
│   ├── data/
│   │   ├── dataset.py              # Ingestion utilities
│   │   └── preprocessing.py        # Scalers and encoders loaders
│   ├── evaluation/
│   │   ├── comparison.py           # Ranking utilities
│   │   └── metrics.py              # Precision, Recall, F1 calculations
│   ├── models/
│   │   ├── autoencoder.py          # Autoencoder training and error scoring
│   │   ├── dbscan.py               # DBSCAN clustering wrapper
│   │   └── isolation_forest.py     # IF train/predict wrappers
│   ├── utils/
│   │   └── helpers.py              # Save/load JSON helpers
│   └── visualization/
│       ├── pca.py                  # 2D PCA projection coordinates compiler
│       └── plotly_plots.py         # Standardized interactive figures layer
│
├── app.py                          # Streamlit application main entry point
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Installation

Verify that Python 3.10+ is installed. Install dependencies using requirements.txt:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Launch the Streamlit security dashboard using:

```bash
streamlit run app.py
```

---

## Project Workflow

```text
Dataset
    ↓
Preprocessing
    ↓
Model
    ↓
Evaluation
    ↓
Visualization
    ↓
Dashboard
```

---

## Models Implemented

- **Isolation Forest**: Isolates anomalies recursively using ensemble trees. Best suited for real-time edge firewalls with constrained memory.
- **DBSCAN**: Identifies dense regions of normal connection traffic, flagging sparse outliers as noise. Best suited for historical, offline packet investigation.
- **Autoencoder**: Neural network trained solely on normal network logs to reconstruct input vectors. Intrusions produce high reconstruction MSE. Best suited for enterprise cores prioritizing security recall.

---

## Notebook Section

The Jupyter notebooks in the notebooks directory document the entire analytical pipeline, starting from initial dataset loading and exploratory analysis through preprocessing, individual model evaluation, and final performance comparison.

---

## Dashboard Section

The Streamlit dashboard provides a browser interface to explore model configurations, run inference on the standard test dataset, compare model metrics, and download labeled anomaly CSVs and reports.

---

## Performance Summary

Evaluated on the standard KDDTest+ split containing unseen, out-of-distribution network intrusions:

| Model | Precision | Recall | F1-Score | Accuracy | Training Time | Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest** | 0.5300 | 0.5579 | 0.5436 | 0.5550 | **0.154s** | **0.052s** |
| **DBSCAN** | 0.4305 | 0.6842 | 0.5285 | 0.4214 | 1.241s | N/A |
| **Autoencoder** | **0.8300** | **0.9769** | **0.8975** | **0.8748** | 7.824s | 0.124s |

---

## Technology Stack

- **Programming Language**: Python
- **Machine Learning**: Scikit-Learn
- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy, Joblib
- **Dashboard**: Streamlit
- **Visualization**: Plotly

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.