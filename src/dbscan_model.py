from sklearn.cluster import DBSCAN
def train_dbscan(x):
    model = DBSCAN(
        eps=0.8,
        min_samples=10
    )

    labels = model.fit_predict(x)

    return labels