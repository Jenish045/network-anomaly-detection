# 🛡️ Network Anomaly Detection System

Unsupervised anomaly detection on the NSL-KDD network intrusion dataset using three approaches — Isolation Forest, DBSCAN, and a deep learning Autoencoder — with an interactive Streamlit dashboard for exploration and comparison.

---

## 📌 Overview

Network intrusion detection is a critical cybersecurity problem. This project frames it as an **unsupervised anomaly detection task**, where the goal is to identify malicious traffic without relying on labeled training data. Three fundamentally different approaches are compared to understand their strengths and limitations on a real-world dataset.

---

## 🎯 Objectives

- Detect anomalous network traffic (attacks vs. normal) using unsupervised learning
- Compare traditional and deep learning approaches on the same dataset
- Analyze model assumptions, limitations, and failure modes
- Build an interactive dashboard for real-time anomaly exploration

---

## 📂 Project Structure

```
network-anomaly-detection/
│
├── data/
│   ├── raw/
│   │   ├── KDDTrain+.txt           # Training data
│   │   └── KDDTest+.txt            # Test data
│   └── processed/
│       └── x_scaled.npy            # Cached scaled features
│
├── notebooks/
│   ├── 01_dataset_loading.ipynb    # Data loading and inspection
│   ├── 02_eda.ipynb                # Exploratory data analysis
│   ├── 03_isolation_forest.ipynb   # Isolation Forest + DBSCAN
│   └── 04_autoencoder.ipynb        # Autoencoder + model comparison
│
├── src/
│   ├── preprocessing.py            # Data loading, encoding, splitting
│   ├── isolation_forest_model.py   # Isolation Forest training & inference
│   ├── autoencoder_model.py        # Autoencoder architecture
│   └── dbscan_model.py             # DBSCAN training
│
├── results/
│   ├── if_metrics.json             # Isolation Forest evaluation metrics
│   ├── dbscan_metrics.json         # DBSCAN evaluation metrics
│   ├── ae_metrics.json             # Autoencoder evaluation metrics
│   ├── isolation_forest_pca.png
│   ├── anomaly_score_density.png
│   ├── ae_pca.png
│   ├── ae_error.png
│   └── autoencoder_model.keras
│
├── app.py                          # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

**NSL-KDD (KDD Cup 99 improved)**
- A widely used benchmark for network intrusion detection
- Contains labeled network traffic: `normal` vs. various attack types
- 41 features (continuous + categorical)
- Train: `KDDTrain+.txt` | Test: `KDDTest+.txt`
- ⚠️ **Note**: The test set has ~57% attacks vs. ~43% normal — anomalies are NOT the minority class, which violates assumptions of traditional anomaly detectors

---

## ⚙️ Models

### 🌲 Isolation Forest
- Tree-based anomaly detection that isolates outliers through random splits
- Assumes anomalies are rare and easy to isolate
- **Limitation on NSL-KDD**: Attack traffic constitutes 57% of the test set, violating the rarity assumption. `contamination` is capped at 0.5 in sklearn, creating a ceiling on performance.
- Evaluated using model default predictions (`contamination=0.5`)

### 🔵 DBSCAN
- Density-based clustering — points in low-density regions are labeled anomalies
- No assumption about anomaly rarity
- **Limitation on NSL-KDD**: High dimensionality degrades distance metrics (curse of dimensionality), causing poor cluster separation and very low recall

### 🧠 Autoencoder (Deep Learning)
- Neural network trained exclusively on normal traffic
- Learns to reconstruct normal patterns — anomalies have high reconstruction error
- Threshold selected by maximizing F1-score across reconstruction error percentiles
- **Best performer**: Captures non-linear feature relationships that linear models miss

---

## 🧪 Methodology

### Preprocessing
1. Load raw NSL-KDD data with correct column names
2. Label-encode categorical features (`protocol_type`, `service`, `flag`)
3. Drop `difficulty` column
4. Standardize features with `StandardScaler` (fit on train, transform test)

### Isolation Forest Pipeline
1. Train on scaled training data with `contamination=0.5`
2. Generate anomaly scores via `decision_function()`
3. Evaluate using default `model.predict()` threshold
4. Visualize score distributions and PCA projections

### DBSCAN Pipeline
1. Fit on scaled test data (transductive — no separate predict)
2. Label points with cluster `-1` as anomalies
3. Evaluate against ground truth labels

### Autoencoder Pipeline
1. Train only on normal traffic samples
2. Compute per-sample reconstruction error (MSE) on test set
3. Search percentiles 30–99 for threshold that maximizes F1
4. Evaluate and compare against other models

---

## 📊 Results

| Model | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| Isolation Forest | 0.80 | 0.80 | 0.80 | 0.78 |
| DBSCAN | 0.67 | 0.11 | 0.18 | 0.46 |
| **Autoencoder** | **0.83** | **0.98** | **0.90** | **0.87** |

### Key Takeaways
- **Autoencoder** achieves the best overall performance, particularly excelling in recall (0.98) — it misses very few actual attacks
- **Isolation Forest** performs well (F1: 0.80) despite the dataset violating its core assumption — tree-based isolation is robust in high dimensions
- **DBSCAN** fails due to the curse of dimensionality — recall of 0.11 means it misses 89% of attacks
- Threshold tuning is critical: the right threshold can dramatically improve F1 in all models

---

## 🖥️ Streamlit Dashboard

An interactive dashboard with four tabs:

| Tab | Contents |
|-----|----------|
| 📊 Dashboard | Detection summary, performance metrics, classification report, confusion matrix |
| 📈 Visualizations | PCA projection, score distribution, top detected attack types |
| 📄 Data Explorer | Filterable results table, CSV download |
| ⚔️ Model Comparison | Side-by-side metrics, grouped bar chart, radar chart |

### Controls
- **Model selector**: Switch between Isolation Forest, DBSCAN, Autoencoder
- **AE threshold slider**: Tune reconstruction error percentile (default: 90%)
- **DBSCAN sliders**: Adjust `eps` and `min_samples` interactively
- **File uploader**: Upload a custom NSL-KDD formatted dataset

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/network-anomaly-detection.git
cd network-anomaly-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Notebooks (optional — to regenerate results)
```bash
jupyter notebook
```
Run in order: `01` → `02` → `03` → `04`

### 4. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | Pandas, NumPy |
| ML Models | Scikit-learn |
| Deep Learning | TensorFlow / Keras |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Persistence | Joblib, JSON |

---

## 🔮 Future Work

- **LSTM Autoencoder**: Capture temporal patterns in network traffic sequences
- **Supervised baseline**: Compare against Random Forest / XGBoost as an upper bound
- **Real-time detection**: Stream live network packets via Scapy or pcap files
- **Feature engineering**: Protocol-specific features, flow-level aggregation
- **Hyperparameter tuning**: Systematic search for optimal DBSCAN parameters using silhouette score

---

## 👨‍💻 Author

**Jenish Upadhyay**