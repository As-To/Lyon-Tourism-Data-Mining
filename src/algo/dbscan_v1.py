import pandas as pd
from pyproj import Transformer
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt



#1ERE ETAPE : CHARGEMENT ET NETTOYAGE DES DONNEES
df = pd.read_csv("../../data/cleaned/cleaned_lyon_data.csv")

# On ne garde que les colonnes utiles pour le clustering spatial
df = df[['lat', 'long', 'tags', 'title']]

# Suppression des lignes avec coordonnées manquantes
df = df.dropna()

print(df.shape)

#2EME ETAPE : TRANSFORMATION DES COORDONNEES GEO EN COORDONNEES PLANES
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
# Tableau final pour DBSCAN
points = np.column_stack((X, Y))

#3EME ETAPE : CLUSTERING SPATIAL AVEC DBSCAN
dbscan = DBSCAN(
    eps=100,        # rayon de 100 mètres
    min_samples=100 # minimum 100 photos pour former une zone
)

labels = dbscan.fit_predict(points)

#4EME ETAPE : ANALYSE DES RESULTATS
df['cluster'] = labels
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