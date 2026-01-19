import pandas as pd
from pyproj import Transformer
import numpy as np
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
from src.algo.text_mining import top_terms_by_cluster, DEFAULT_STOPWORDS

def compute_dbscan_with_kmeans_split(df, eps=100, min_samples=80, max_cluster_size=5000):
    """
    Entrée : dataframe contenant au minimum 'lat' et 'long'
    Sortie : (df_out, X, Y, labels)
      - df_out : dataframe nettoyé + colonne 'cluster' (labels DBSCAN + KMeans)
      - X, Y   : coordonnées projetées en Lambert-93 (mètres)
      - labels : array des labels (même ordre que df_out)
    """

    # Ne pas changer le reste : on garde exactement la même logique de nettoyage
    df_out = df[['lat', 'long', 'tags', 'title']].dropna(subset=['lat', 'long']).copy()

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

    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    df['cluster'] = labels

    #K-Means à l'intérieur des clusters trop grands  
    cluster_counts = df['cluster'].value_counts()
    #Retirer les clusters de bruit (-1)
    cluster_counts = cluster_counts[cluster_counts.index != -1]
    large_clusters = cluster_counts[cluster_counts > 5000].index

    for cluster_label in large_clusters:
        cluster_points = points[df['cluster'] == cluster_label]
        kmeans = KMeans(n_clusters=8, random_state=42)
        sub_labels = kmeans.fit_predict(cluster_points)

        # Ajuster les labels pour qu'ils soient uniques
        max_label = df['cluster'].max()
        sub_labels_adjusted = np.where(sub_labels != -1, sub_labels + max_label + 1, -1)

        # Mettre à jour les labels dans le DataFrame principal
        df.loc[df['cluster'] == cluster_label, 'cluster'] = sub_labels_adjusted
        #Insérer le nouveau label dans le tableau des labels
        labels = df['cluster'].values

        df_out['cluster'] = labels

    return df_out, X, Y, labels

# Trouve le chemin du dossier actuel du script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construit le chemin vers le fichier CSV
file_path = os.path.join(current_dir, "../../data/cleaned/cleaned_lyon_data.csv")

df = pd.read_csv(file_path)

# Appel de la fonction (calcule X, Y, labels + ajoute cluster)
df, X, Y, labels = compute_dbscan_with_kmeans_split(df, eps=100, min_samples=80, max_cluster_size=5000) 


#5EME ETAPE : AFFICHAGE DES RESULTATS
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_ratio = (labels == -1).mean()

print(f"Nombre de clusters : {n_clusters}")
print(f"Proportion de bruit : {noise_ratio:.2%}")

#Nombre de points par cluster
unique, counts = np.unique(labels, return_counts=True)
print("Nombre de points par cluster (label : nombre de points) :")
for label, count in zip(unique, counts):
    print(f"{label} : {count}")


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

# Analyse des mots les plus fréquents par cluster via text_mining

print(f"Nombre de clusters : {n_clusters}")
print(f"Proportion de bruit : {noise_ratio:.2%}")

topics = top_terms_by_cluster(df, cluster_col="cluster", top_k=10, stopwords=DEFAULT_STOPWORDS, use_lemmas=True)
print("Mots représentatifs par cluster :")
for cl, terms in topics.items():
    print(f"Cluster {cl}: {', '.join(terms)}")
