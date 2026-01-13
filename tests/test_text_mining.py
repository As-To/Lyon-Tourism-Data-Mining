import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.algo.text_mining import top_terms_by_cluster


def main():
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "sample_data.csv")
    sample_path = os.path.abspath(sample_path)

    df = pd.read_csv(sample_path, dtype=str)

    # ensure expected text columns exist
    for c in ("tags", "title"):
        if c not in df.columns:
            df[c] = ""

    # create a simple cluster assignment for testing
    df["cluster"] = (df["user"].fillna("").astype("category").cat.codes % 5)

    results = top_terms_by_cluster(df, cluster_col="cluster", top_k=10, stopwords=None)

    print("Top terms by cluster (sample):")
    for cl, terms in results.items():
        print(f"Cluster {cl}: {', '.join(terms)}")


if __name__ == '__main__':
    main()
