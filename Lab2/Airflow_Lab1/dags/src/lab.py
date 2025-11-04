import pickle
import os
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import Counter

def load_wine_data():
    """
    Loads the Wine Recognition dataset from sklearn and serializes it.
    """
    data = load_wine(as_frame=True)
    df = data.frame  # Pandas DataFrame
    serialized = pickle.dumps(df)
    return serialized


def preprocess_wine_data(data):
    """
    Preprocess wine dataset (scales features).
    """
    df = pickle.loads(data)
    X = df.drop("target", axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    serialized = pickle.dumps(X_scaled)
    return serialized


def train_wine_clusters(data, filename):
    """
    Train a KMeans model on wine data and save it.
    """
    X_scaled = pickle.loads(data)

    kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42)
    kmeans.fit(X_scaled)

    # Save model
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(kmeans, f)

    serialized = pickle.dumps(kmeans.labels_)
    return serialized


def evaluate_wine_clusters(filename, labels_data):
    """
    Load model and evaluate cluster distribution.
    """
    labels = pickle.loads(labels_data)
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    model = pickle.load(open(output_path, "rb"))

    counts = Counter(labels)
    print("Wine Cluster distribution:", counts)
    return counts