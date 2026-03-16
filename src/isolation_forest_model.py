from sklearn.ensemble import IsolationForest
import numpy as np

def train_isolation_forest(x):
    model = IsolationForest(
        n_estimators=200,
        contamination=0.55,
        random_state=42
    )

    model.fit(x)
    return model

def detect_anomalies(model,x):
    predictions = model.predict(x)
    return predictions 

def anomaly_scores(model,x):
    scores = model.decision_function(x)
    return scores