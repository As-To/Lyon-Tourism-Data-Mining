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


# === NOUVELLE FONCTION REUTILISABLE (sans effet de bord) ===
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

    '''
    # Plage de k à tester (ex: de 2 à 20 zones)
    K_range = range(2, 20)
    inertias = []

    print("Calcul de l'inertie pour différents k en cours...")

    for k in K_range:
        # On entraine le modèle pour chaque k
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(points)
        # On stocke l'inertie (la somme des distances carrées intra-cluster)
        inertias.append(kmeans.inertia_)

    # Affichage du graphique "Coude"
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Inertie')
    plt.title('Méthode du Coude pour déterminer le nombre optimal de zones')
    plt.xticks(K_range)
    plt.grid(True)
    plt.show()
    '''

    # Avec la méthode du coude, on aurait dit que k=5 est un bon choix, cependant 5 endroits à visiter à Lyon est trop peu.
    # Donc d'apres ce que l'on a vu avec DBSCAN, On va choisir K=25.
    points, X, Y, df = transform_geo(df)

    k_optimal = 25
    df = run_kmeans(points, k_optimal, df)
    df['cluster'].value_counts()

    # AFFICHAGE DES RESULTATS
    n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0)
    print(f"Nombre de clusters : {n_clusters}")

    # VISUALISATION DES CLUSTERS
    xmin, xmax = np.percentile(X, [0, 100])
    ymin, ymax = np.percentile(Y, [0, 100])
    plt.figure(figsize=(7, 7))
    plt.scatter(X, Y, c=df['cluster'], s=5, cmap='tab20')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title("K-Means – zone centrale")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()

    # Analyse des mots les plus fréquents par cluster via text_mining
    print(f"Nombre de clusters : {n_clusters}")

    topics = top_terms_by_cluster(df, cluster_col="cluster", top_k=10,
                                 stopwords=DEFAULT_STOPWORDS, use_lemmas=True)
    print("Mots représentatifs par cluster :")
    for cl, terms in topics.items():
        print(f"Cluster {cl}: {', '.join(terms)}")


if __name__ == "__main__":
    main()
