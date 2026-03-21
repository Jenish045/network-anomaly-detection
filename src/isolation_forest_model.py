from sklearn.ensemble import IsolationForest
import numpy as np

def train_isolation_forest(X_train):
    model = IsolationForest(
        contamination=0.5,  
        random_state=42,
        n_estimators=100
    )
    model.fit(X_train)
    return model

def detect_anomalies(model,x):
    predictions = model.predict(x)
    return predictions 

def anomaly_scores(model,x):
    scores = model.decision_function(x)
    return scores