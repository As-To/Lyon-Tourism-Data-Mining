import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.algo.text_mining import top_terms_by_cluster
from src.algo.dbscan_v1 import compute_dbscan_xy_labels


def main():
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "sample_data.csv")
    sample_path = os.path.abspath(sample_path)

    df = pd.read_csv(sample_path, dtype=str)

    # ensure expected text columns exist
    for c in ("tags", "title"):
        if c not in df.columns:
            df[c] = ""

    # Use the project's DBSCAN wrapper which projects coordinates
    # compute_dbscan_xy_labels returns (df_out, X, Y, labels)
    # eps is in meters (Lambert-93 projection) and min_samples as usual
    try:
        df_out, X, Y, labels = compute_dbscan_xy_labels(df, eps=100, min_samples=5)
    except Exception as e:
        print(f"DBSCAN preprocessing failed: {e}")
        return

    results = top_terms_by_cluster(df_out, cluster_col="cluster", top_k=10, stopwords=None)

    print("Top terms by cluster (DBSCAN sample):")
    for cl, terms in sorted(results.items()):
        print(f"Cluster {cl}: {', '.join(terms)}")


if __name__ == '__main__':
    main()
