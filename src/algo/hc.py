import pandas as pd
from pyproj import Transformer
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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
        kmeans = KMeans(n_clusters=2, random_state=random_state, n_init="auto")
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
    df_out = df[['lat', 'long']].dropna().copy()

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
    df_out = divisional_clustering(df_out[["X", "Y"]], max_size=max_size, random_state=random_state)

    return df_out, X, Y


#1ERE ETAPE : CHARGEMENT ET NETTOYAGE DES DONNEES
df = pd.read_csv("../../data/cleaned/cleaned_lyon_data.csv")

# Appel de la fonction (calcule X, Y, labels + ajoute cluster)
df_result, X, Y = compute_hc_clustering(df, max_size=5000)

#4EME ETAPE : ANALYSE DES RESULTATS
df_result['cluster'].value_counts()

#5EME ETAPE : AFFICHAGE DES RESULTATS
n_clusters = len(set(df_result['cluster'])) - (1 if -1 in df_result['cluster'] else 0)
noise_ratio = (df_result['cluster'] == -1).mean()

print(f"Nombre de clusters : {n_clusters}")
print(f"Proportion de bruit : {noise_ratio:.2%}")

#Nombre de points par cluster
unique, counts = np.unique(df_result['cluster'], return_counts=True)
print("Nombre de points par cluster (label : nombre de points) :")
for label, count in zip(unique, counts):
    print(f"{label} : {count}")


#5EME ETAPE : VISUALISATION DES CLUSTERS
# Paramètre d'affichage
SHOW_NOISE = False  # True = afficher le bruit, False = le masquer

# Masque des points à afficher
mask = np.ones_like(df_result['cluster'], dtype=bool) if SHOW_NOISE else (df_result['cluster'] != -1)


xmin, xmax = np.percentile(X, [5, 95])
ymin, ymax = np.percentile(Y, [5, 80])

plt.figure(figsize=(6, 6))
plt.scatter(X[mask], Y[mask], c=df_result['cluster'][mask], s=5, cmap='tab20')

#plt.xlim(xmin, xmax)
#plt.ylim(ymin, ymax)

plt.title("Hierarchical clustering – zone centrale")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()    