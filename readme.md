# 🚀 Network Anomaly Detection System (ML + Deep Learning + Streamlit)

## 📌 Overview
This project detects network anomalies using multiple unsupervised learning techniques on the KDD dataset.  
It compares traditional methods with deep learning and provides an interactive dashboard using Streamlit.

---

## 🎯 Objectives
- Detect anomalous network traffic (attacks vs normal)
- Compare different unsupervised learning approaches
- Analyze model performance and limitations
- Build an interactive visualization dashboard

---

## 📂 Dataset
- **KDD Cup Dataset**
- Contains labeled network traffic (normal vs attack)
- High-dimensional and complex → challenging for anomaly detection

---

## ⚙️ Models Implemented

### 🔹 Isolation Forest
- Tree-based anomaly detection
- Assumes anomalies are rare
- Struggles with overlapping data distributions

---

### 🔹 DBSCAN
- Density-based clustering
- Detects anomalies as low-density points
- Highly sensitive to parameters (eps, min_samples)
- Performs poorly in high-dimensional data

---

### 🔹 Autoencoder (Deep Learning)
- Neural network-based reconstruction model
- Trained only on normal data
- Uses reconstruction error for anomaly detection
- Captures complex feature relationships

---

## 🧪 Methodology

### Data Preprocessing
- Encoding categorical features
- Feature scaling using StandardScaler
- Train-test split (KDDTrain+ / KDDTest+)

---

### Isolation Forest
- Generates anomaly scores
- Threshold tuning using percentiles
- Limited by overlapping score distributions

---

### DBSCAN
- Clusters dense regions
- Labels noise points as anomalies
- Performance depends heavily on parameter tuning

---

### Autoencoder
- Trained only on normal data
- Reconstruction error used for anomaly detection
- Threshold optimized using F1-score

---

## 📊 Model Performance Comparison

| Model | Precision | Recall | F1-Score | Accuracy | Key Behavior |
|------|----------|--------|----------|----------|-------------|
| Isolation Forest (Tuned) | 0.44 | 0.54 | 0.49 | 0.35 | Weak separation due to overlap |
| DBSCAN | 0.67 | 0.11 | 0.18 | 0.46 | Misses most attacks |
| Autoencoder (Optimized) | **0.95** | **0.83** | **0.89** | **0.88** | Best overall performance |

---

## 📈 Results & Insights

### Isolation Forest
- Assumes anomalies are rare → violated in KDD dataset
- Overlapping distributions reduce effectiveness

### DBSCAN
- Struggles with high-dimensional data
- Very low recall (0.11) → fails to detect most attacks

### Autoencoder
- Learns complex patterns
- Achieves best balance of precision and recall
- Significantly reduces false positives

---

## 📊 Visualizations
- Anomaly score distribution (Isolation Forest)
- Reconstruction error distribution (Autoencoder)
- PCA-based anomaly visualization
- Attack type analysis

---

## 🧠 Key Learnings
- Model assumptions strongly affect performance
- Traditional methods struggle with complex datasets
- Deep learning models capture non-linear patterns effectively
- Threshold tuning is critical in anomaly detection

---

## 🖥️ Streamlit Dashboard

### Features
- Select model: Isolation Forest / DBSCAN / Autoencoder
- Adjust threshold and DBSCAN parameters
- View real-time anomaly detection
- Interactive PCA visualization
- Compare model performance

---

## 🛠️ Tech Stack
- Python
- Scikit-learn
- TensorFlow / Keras
- Pandas, NumPy
- Matplotlib, Seaborn
- Plotly
- Streamlit

---

## 📁 Project Structure
```
network-anomaly-detection/
│
├── data/
│   └── raw/
│       ├── KDDTrain+.txt
│       └── KDDTest+.txt
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_dbscan.ipynb
│   ├── 03_isolation_forest.ipynb
│   └── 04_autoencoder.ipynb
│
├── src/
│   ├── preprocessing.py          # data loading, encoding, splitting
│   ├── isolation_forest_model.py
│   └── autoencoder_model.py
│
├── results/
│   ├── anomaly_score_density.png
│   ├── ae_error.png
│   ├── ae_pca.png
│   ├── isolation_forest.png
│   └── model_comparison.csv
│
├── app.py              # main dashboard app
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone Repository
```
git clone https://github.com/your-username/network-anomaly-detection.git
cd network-anomaly-detection
```

---

### 2. Install Dependencies
```
pip install -r requirements.txt
```

---

### 3. Run Streamlit App
```
streamlit run streamlit_app.py
```

---

### 4. Run Notebooks
```
jupyter notebook
```

---

## 🏁 Conclusion

The Autoencoder achieved an F1-score of **0.89**, significantly outperforming:
- Isolation Forest (**0.49**)
- DBSCAN (**0.18**)

This demonstrates that deep learning models are more effective for complex, high-dimensional anomaly detection tasks.

---

## 🔮 Future Work
- Hyperparameter tuning for DBSCAN
- LSTM-based anomaly detection
- Real-time deployment
- Feature engineering improvements

---

## 👨‍💻 Author
Jenish Upadhyay