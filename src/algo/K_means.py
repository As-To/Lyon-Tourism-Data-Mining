import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pandas as pd
from pyproj import Transformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from src.algo.text_mining import top_terms_by_cluster, DEFAULT_STOPWORDS


# On ne garde que les colonnes utiles pour le clustering spatial
def transform_geo(df):
    df = df[['lat', 'long', 'tags', 'title']]
    df = df.dropna()

    print(df.shape)

    # Projection WGS84 (GPS) → Lambert-93 (mètres, France)
    transformer = Transformer.from_crs(
        "EPSG:4326",   # lat / lon
        "EPSG:2154",   # Lambert-93
        always_xy=True
    )

    # Transformation
    X, Y = transformer.transform(
        df['long'].values,
        df['lat'].values
    )

    # Tableau final pour K-Means
    points = np.column_stack((X, Y))
    return points, X, Y, df


def run_kmeans(points, k, df):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(points)
    df['cluster'] = labels
    return df


def compute_kmeans_clustering(df, k=25):
    points, X, Y, df2 = transform_geo(df)
    df2 = run_kmeans(points, k, df2)
    labels = df2["cluster"].values
    return df2, X, Y, labels


def main():
    # Trouve le chemin du dossier actuel du script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construit le chemin vers le fichier CSV
    file_path = os.path.join(current_dir, "../../data/cleaned/cleaned_lyon_data.csv")

    df = pd.read_csv(file_path)
    

    points, X, Y, df = transform_geo(df)
    k_optimal = 200
    df = run_kmeans(points, k_optimal, df)

    # --- ÉTAPE DE FILTRAGE ---
    MIN_SIZE = 75
    
    # 1. Calculer la taille de CHAQUE cluster
    counts = df['cluster'].value_counts()
    
    # 2. Identifier les IDs des clusters qui respectent la condition
    valid_clusters_ids = counts[counts >= MIN_SIZE].index
    
    # 3. Créer le masque pour filtrer le DataFrame et les coordonnées X, Y
    mask = df['cluster'].isin(valid_clusters_ids)
    
    # 4. Appliquer le filtre (très important pour que X, Y et df gardent la même taille)
    df_filtered = df[mask].copy()
    X_filtered = X[mask]
    Y_filtered = Y[mask]

    # --- AFFICHAGE DE LA RÉPARTITION ---
    # On récupère les comptes uniquement pour les clusters valides
    final_counts = df_filtered['cluster'].value_counts().sort_index()
    
    print("-" * 30)
    print(f"Répartition des points sur les {len(final_counts)} clusters valides (>= {MIN_SIZE} pts) :")
    for cluster_id, count in final_counts.items():
        print(f"Cluster {cluster_id} : {count} points")
    print("-" * 30)

    # --- ANALYSE ET VISUALISATION ---
    n_clusters = len(final_counts)
    print(f"Nombre de clusters restants : {n_clusters}")

    # Utiliser le dataframe filtré pour le text mining
    topics = top_terms_by_cluster(df_filtered, cluster_col="cluster", top_k=10,
                                 stopwords=DEFAULT_STOPWORDS, use_lemmas=True)
    print("Mots représentatifs par cluster :")
    for cl, terms in topics.items():
        main_term = terms[0].upper() 
        print(f"Cluster {cl} : {main_term}")
    
    # Utiliser les coordonnées filtrées pour le scatter plot
    xmin, xmax = np.percentile(X, [0, 100])
    ymin, ymax = np.percentile(Y, [0, 100])
    plt.figure(figsize=(7, 7))
    plt.scatter(X_filtered, Y_filtered, c=df_filtered['cluster'], s=5, cmap='tab20')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title("K-Means – zone centrale")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()


    

    


if __name__ == "__main__":
    main()
