import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from dbscan_v1 import compute_dbscan_xy_labels
from dbscan_v2 import compute_dbscan_with_kmeans_split
from hc import compute_hc_clustering
from K_means import compute_kmeans_clustering


def plot_one(ax, X, Y, labels, title, show_noise=False):
    if show_noise:
        mask = np.ones_like(labels, dtype=bool)
    else:
        mask = labels != -1
    ax.scatter(X[mask], Y[mask], c=labels[mask], s=2, cmap="tab20")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "../../data/cleaned/cleaned_lyon_data.csv")
    df = pd.read_csv(file_path)

    # 1) K-means (réutilise K_means.py)
    df_km, X_km, Y_km, lab_km = compute_kmeans_clustering(df, k=25)

    # 2) HC (réutilise hc.py)
    df_hc, X_hc, Y_hc = compute_hc_clustering(df, max_size=5000)
    lab_hc = df_hc["cluster"].values

    # 3) DBSCAN v1 (réutilise dbscan_v1.py)
    df_db1, X_db1, Y_db1, lab_db1 = compute_dbscan_xy_labels(df, eps=100, min_samples=100)

    # 4) DBSCAN v2 (réutilise dbscan_v2.py)
    df_db2, X_db2, Y_db2, lab_db2 = compute_dbscan_with_kmeans_split(df, eps=100, min_samples=100, max_cluster_size=5000)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    plot_one(axes[0, 0], X_km,  Y_km,  lab_km,  "K-means (k=25)")
    plot_one(axes[0, 1], X_hc,  Y_hc,  lab_hc,  "HC divisional (max_size=5000)")
    plot_one(axes[1, 0], X_db1, Y_db1, lab_db1, "DBSCAN v1 (eps=100, minPts=50)")
    plot_one(axes[1, 1], X_db2, Y_db2, lab_db2, "DBSCAN v2 (split)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
