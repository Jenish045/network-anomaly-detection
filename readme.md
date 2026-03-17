# 🚀 Network Anomaly Detection System (NSL-KDD)

A production-ready Machine Learning project for detecting network intrusions using unsupervised learning techniques — built with a complete pipeline from data preprocessing to deployment via Streamlit.

---

## 📌 Project Overview

This project focuses on detecting anomalous network traffic (attacks) using:

- Isolation Forest  
- Autoencoder (Deep Learning)  
- DBSCAN (for comparison)

The system simulates real-world anomaly detection in cybersecurity using the NSL-KDD dataset.

---

## 🎯 Key Features

- End-to-end ML pipeline (data → model → evaluation → deployment)
- Multiple anomaly detection models
- Interactive Streamlit dashboard
- Model comparison (Isolation Forest vs Autoencoder)
- Adjustable anomaly threshold
- PCA-based visualization
- Attack type insights
- False positive analysis
- Downloadable results

---

## 🧠 Models Used

### 1. Isolation Forest
- Tree-based anomaly detection
- Works well in high-dimensional data
- Fast and efficient

### 2. Autoencoder
- Deep learning model trained on normal data
- Detects anomalies via reconstruction error
- Best performance in this project

### 3. DBSCAN
- Density-based clustering
- Used for comparison
- Performs poorly on high-dimensional data

---

## 📊 Model Performance

- Isolation Forest → Accuracy: ~0.79, F1: ~0.79  
- DBSCAN → Accuracy: ~0.46, F1: ~0.18  
- Autoencoder → Accuracy: ~0.86, F1: ~0.86  

---

## 📁 Project Structure

Network-Anomaly-Detection/
│
├── data/
│   └── raw/
│       ├── KDDTrain+.txt
│       └── KDDTest+.txt
│
├── src/
│   ├── preprocessing.py
│   ├── isolation_forest_model.py
│   ├── dbscan_model.py
│   └── autoencoder_model.py
│
├── notebooks/
│   ├── 03_isolation_forest.ipynb
│   └── 04_autoencoder.ipynb
│
├── results/
│
├── app.py
├── requirements.txt
└── README.md

---

## ⚙️ Installation

1. Clone the repository

git clone https://github.com/jenish045/network-anomaly-detection.git  
cd network-anomaly-detection  

2. Install dependencies

pip install -r requirements.txt  

---

## ▶️ Running the App

streamlit run app.py  

Open in browser:  
http://localhost:8501  

---

## 🖥️ Streamlit Dashboard

### Dashboard
- Total samples
- Number of anomalies
- Classification report

### Visualizations
- PCA projection
- Anomaly score distribution
- Attack type insights

### Model Comparison
- Isolation Forest vs Autoencoder
- Accuracy & F1 comparison

### Data
- Preview dataset
- Download results

---

## 🎛️ Interactive Controls

- Model selection
- Threshold slider
- Upload dataset

---

## 📊 Insights

- Isolation Forest performs well but struggles with dense attacks  
- Autoencoder performs best by learning normal patterns  
- DBSCAN fails due to high dimensionality  
- Feature overlap causes false positives  

---

## 🚀 Future Improvements

- Real-time anomaly detection
- Cloud deployment
- Advanced deep learning models
- Feature engineering improvements
- Hybrid model approach

---

## 🛠️ Tech Stack

- Python  
- Scikit-learn  
- TensorFlow / Keras  
- Pandas / NumPy  
- Plotly  
- Streamlit  

---

## 📌 Dataset

NSL-KDD Dataset  
Improved version of KDD Cup 1999 for intrusion detection research  

---

## 🙌 Author

Jenish Upadhyay  

---

## ⭐ Support

If you found this useful, consider giving it a star.