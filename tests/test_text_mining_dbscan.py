import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from src.algo.text_mining import top_terms_by_cluster


def main():
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "sample_data.csv")
    sample_path = os.path.abspath(sample_path)

    df = pd.read_csv(sample_path, dtype=str)

    # ensure expected text columns exist
    for c in ("tags", "title"):
        if c not in df.columns:
            df[c] = ""

    # keep rows with valid lat,long
    df = df.dropna(subset=["lat", "long"])[:5000]  # limit for speed
    df["lat"] = df["lat"].astype(float)
    df["long"] = df["long"].astype(float)

    # convert to radians for haversine metric
    coords = np.radians(df[["lat", "long"]].values)

    # DBSCAN with haversine metric; eps is in radians
    # eps ~ 0.001 rad ≈ 6.37 km; adjust if needed
    db = DBSCAN(eps=0.001, min_samples=5, metric='haversine')
    labels = db.fit_predict(coords)

    df["cluster"] = labels

    # filter out noise if you want: keep clusters >=0
    # df = df[df['cluster'] >= 0]

    results = top_terms_by_cluster(df, cluster_col="cluster", top_k=10, stopwords=None)

    print("Top terms by cluster (DBSCAN sample):")
    for cl, terms in sorted(results.items()):
        print(f"Cluster {cl}: {', '.join(terms)}")


if __name__ == '__main__':
    main()
