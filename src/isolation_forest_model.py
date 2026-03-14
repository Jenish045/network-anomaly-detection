from sklearn.ensemble import IsolationForest
import numpy as np

def train_isolation_forest(x):
    model = IsolationForest(
        n_estimators=100,
        contamination=0.4,
        random_state=42
    )

    model.fit(x)
    return model

def detect_anamolies(model,x):
    predictions = model.predict(x)
    return predictions 

def anamoly_scores(model,x):
    scores = model.decision_function(x)
    return scores