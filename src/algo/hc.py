import pandas as pd
import os
from pyproj import Transformer
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt


def agglomerative_clustering_algo(df, n_clusters=20, sample_size=5000):
    """
    Clustering Agglomératif (Bottom-Up).
    """
    # 1. Échantillonnage de sécurité
    if len(df) > sample_size:
        print(f" Dataset trop gros pour Agglomerative ({len(df)} pts). Échantillonnage à {sample_size} pts.")
        # On prend un échantillon aléatoire
        df_work = df.sample(n=sample_size, random_state=42).copy()
    else:
        df_work = df.copy()

    # 2. Préparation des points (X, Y doivent déjà être dans le DF)
    points = df_work[['X', 'Y']].values

    # 3. Exécution de l'algorithme Agglomerative (Ward = minimise la variance, comme K-Means)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = model.fit_predict(points)

    df_work['cluster'] = labels
    
    return df_work

def compute_agglomerative_wrapper(df, n_clusters=50, sample_size=5000):
    # Étape 1 : Nettoyage & Projection (copié de compute_hc_clustering)
    df_out = df[['lat', 'long', 'tags', 'title']].dropna(subset=['lat', 'long']).copy()
    
    
    if 'X' not in df_out.columns:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
        X, Y = transformer.transform(df_out['long'].values, df_out['lat'].values)
        df_out['X'] = X
        df_out['Y'] = Y
        
    # Étape 2 : Algo Bottom-Up
    df_result = agglomerative_clustering_algo(df_out, n_clusters=n_clusters, sample_size=sample_size)
    
    return df_result

def divisional_clustering(df, max_size=5000, random_state=42):
    """
    Clustering hiérarchique divisional.
    Sépare récursivement les clusters trop grands avec K-Means (k=2).

    df : DataFrame avec colonnes ['X', 'Y']
    max_size : taille max autorisée par cluster
    retourne : DataFrame avec colonne 'cluster'
    """

    # Initialisation
    df = df.copy()
    df["cluster"] = -1

    # Liste des clusters à traiter : (indices des points)
    clusters_to_split = [df.index.to_numpy()]
    final_clusters = []

    while clusters_to_split:
        idx = clusters_to_split.pop()
        cluster_points = df.loc[idx, ["X", "Y"]].values

        # Condition d'arrêt
        if len(idx) <= max_size:
            final_clusters.append(idx)
            continue

        # Division en 2
        kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(cluster_points)

        idx_0 = idx[labels == 0]
        idx_1 = idx[labels == 1]

        clusters_to_split.extend([idx_0, idx_1])

    # Attribution des labels finaux
    for i, idx in enumerate(final_clusters):
        df.loc[idx, "cluster"] = i

    return df


def compute_hc_clustering(df, max_size=5000, random_state=42):
    """
    Entrée : dataframe contenant au minimum 'lat' et 'long'
    Sortie : (df_out, X, Y)
      - df_out : dataframe nettoyé + colonne 'cluster' (labels Hierarchical Clustering)
      - X, Y   : coordonnées projetées en Lambert-93 (mètres)
    """

    # Étape 1 : Nettoyage
    df_out = df[['lat', 'long', 'tags', 'title']].dropna(subset=['lat', 'long']).copy()

    # Étape 2 : Transformation des coordonnées GEO en coordonnées planes
    # Projection WGS84 (GPS) → Lambert-93 (mètres, France)
    transformer = Transformer.from_crs(
        "EPSG:4326",   # lat / lon
        "EPSG:2154",   # Lambert-93
        always_xy=True
    )

    # Transformation
    X, Y = transformer.transform(
        df_out['long'].values,
        df_out['lat'].values
    )

    # Étape 3 : Clustering spatial avec Hierarchical Clustering (Divisible)
    df_out["X"] = X
    df_out["Y"] = Y
    df_out = divisional_clustering(df_out, max_size=max_size, random_state=random_state)

    return df_out, X, Y



if __name__ == "__main__":
    # TOUT ce code ne s'exécutera QUE si tu lances 'python hc.py' 
    # Il sera IGNORÉ quand Streamlit importera le fichier.
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "../../data/cleaned/cleaned_lyon_data.csv")
    df = pd.read_csv(file_path)

    df_result, X, Y = compute_hc_clustering(df, max_size=5000)

    # Tes affichages de test
    print(f"Nombre de clusters : {len(df_result['cluster'].unique())}")
    plt.scatter(X, Y, c=df_result['cluster'], s=5)
    plt.show()