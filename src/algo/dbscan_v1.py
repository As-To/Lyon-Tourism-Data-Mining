import pandas as pd
from pyproj import Transformer
import numpy as np
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def compute_dbscan_xy_labels(df, eps=100, min_samples=100):
    """
    Entrée : dataframe contenant au minimum 'lat' et 'long'
    Sortie : (df_out, X, Y, labels)
      - df_out : dataframe nettoyé + colonne 'cluster' (labels DBSCAN)
      - X, Y   : coordonnées projetées en Lambert-93 (mètres)
      - labels : array des labels (même ordre que df_out)
    """

    # Ne pas changer le reste : on garde exactement la même logique de nettoyage
    df_out = df[['lat', 'long', 'tags', 'title']].dropna().copy()

    # Projection WGS84 (GPS) → Lambert-93 (mètres, France)
    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:2154",
        always_xy=True
    )

    # Transformation (lon, lat) -> (X, Y)
    X, Y = transformer.transform(
        df_out['long'].values,
        df_out['lat'].values
    )

    points = np.column_stack((X, Y))
    plot_k_distance(points, k=min_samples)

    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    # Ajout au dataframe
    df_out['cluster'] = labels

    return df_out, X, Y, labels

def plot_k_distance(points, k=4):
    """
    Affiche le graphe des distances au k-ième plus proche voisin
    pour aider à choisir le paramètre eps de DBSCAN.
    """
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(points)
    distances, indices = nbrs.kneighbors(points)
    k_distances = np.sort(distances[:, k-1])

    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.title(f"Graphique des distances au {k}ème plus proche voisin")
    plt.xlabel("Points triés par distance")
    plt.ylabel(f"Distance au {k}ème plus proche voisin")
    plt.grid()
    plt.show()


#1ERE ETAPE : CHARGEMENT ET NETTOYAGE DES DONNEES
# df = pd.read_csv("data/cleaned/cleaned_lyon_data.csv")
# Trouve le chemin du dossier actuel du script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construit le chemin vers le fichier CSV
file_path = os.path.join(current_dir, "../../data/cleaned/cleaned_lyon_data.csv")

df = pd.read_csv(file_path)

# Appel de la fonction (calcule X, Y, labels + ajoute cluster)
df, X, Y, labels = compute_dbscan_xy_labels(df, eps=100, min_samples=100)


#4EME ETAPE : ANALYSE DES RESULTATS
df['cluster'].value_counts()

#5EME ETAPE : AFFICHAGE DES RESULTATS
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_ratio = (labels == -1).mean()

print(f"Nombre de clusters : {n_clusters}")
print(f"Proportion de bruit : {noise_ratio:.2%}")

#Nombre de points par cluster
unique, counts = np.unique(labels, return_counts=True)
print("Nombre de points par cluster (label : nombre de points) :")
for label, count in zip(unique, counts):
    cluster_proportion = count / len(labels) * 100
    print(f"{label} : {cluster_proportion:.2f}% ({count} points)")

#5EME ETAPE : VISUALISATION DES CLUSTERS
# Paramètre d'affichage
SHOW_NOISE = False  # True = afficher le bruit, False = le masquer

# Masque des points à afficher
mask = np.ones_like(labels, dtype=bool) if SHOW_NOISE else (labels != -1)

xmin, xmax = np.percentile(X, [5, 95])
ymin, ymax = np.percentile(Y, [5, 80])

plt.figure(figsize=(6, 6))
plt.scatter(X[mask], Y[mask], c=labels[mask], s=5, cmap='tab20')

#plt.xlim(xmin, xmax)
#plt.ylim(ymin, ymax)

plt.title("DBSCAN – zone centrale")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()