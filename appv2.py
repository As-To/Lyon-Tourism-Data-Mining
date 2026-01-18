import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from src.algo.K_means import run_kmeans,transform_geo
from src.algo.dbscan_v1 import compute_dbscan_xy_labels
from src.algo.dbscan_v2 import compute_dbscan_with_kmeans_split

# 1. Configuration de la page
st.set_page_config(page_title="Projet Lyon - Visualisation", layout="wide")
st.title(" Analyse des zones d'intérêt à Lyon")
 
# 2. Chargement des données (Cache pour éviter de recharger à chaque clic)
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned/cleaned_lyon_data.csv")
    
    # SUPPRIME LES ESPACES INUTILES DANS LES NOMS DE COLONNES
    df.columns = df.columns.str.strip()
   
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Erreur : Le fichier 'sample_data.csv' est introuvable.")
    st.stop()

# 3. Définition de la Zone (Le Rectangle Rouge)
# Coordonnées [Min, Max]
# LAT_MIN, LAT_MAX = 45.65, 45.85
# LON_MIN, LON_MAX = 4.7, 5.0

LAT_MIN, LAT_MAX = 45.73, 45.79
LON_MIN, LON_MAX = 4.81, 4.9

# 4. Interface Latérale (Boutons et Filtres)
st.sidebar.header("Options de Filtrage")

def get_color(cluster_id):
    if cluster_id == -1: return 'black' # Bruit (DBSCAN)

    idx = int(cluster_id)

    colors_list = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                   'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
                   'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 
                   'gray', 'black', 'lightgray']
    return colors_list[idx % len(colors_list)]

# Option A : Choisir quel jeu de données afficher
filter_mode = st.sidebar.radio(
    "Choisissez la vue :",
    ("Données Brutes (Tout)", 
     "Filtrage Zone (Lyon uniquement)", 
     "Filtrage Complet (Zone + Doublons)")
)

# Option B : Afficher le rectangle visuel
show_rectangle = st.sidebar.checkbox("Afficher la zone de Lyon (Rectangle Rouge)", value=True)

data_to_display = pd.DataFrame()
map_center = [45.75, 4.85] # Centre de Lyon
# 6. Création de la carte Folium
m = folium.Map(location=map_center, zoom_start=11)

# Création des onglets
tab1, tab2,tab3 = st.tabs(["Carte 1 : K-Means", "Carte 2 : DBSCAN", "Carte 3 : DBSCAN + K-Means"])

points, X, Y, df = transform_geo(df)
with tab1:
    st.header("Algorithme K-Means")
    
    # Paramètre interactif
    k_value = st.slider("Nombre de zones (k)", min_value=2, max_value=25, value=8, key="k_means_slider")
    
    # Exécution
    res_kmeans = run_kmeans(points, k_value, df.copy())
    
    # Carte
    m_kmeans = folium.Map(location=[45.75, 4.85], zoom_start=12)
    
    # On limite l'affichage pour la fluidité (ex: 2000 points)
    subset = res_kmeans.head(2000)
    
    for idx, row in subset.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=3,
            color=get_color(row['cluster']),
            fill=True,
            fill_opacity=0.7,
            popup=f"Cluster K-Means: {row['cluster']}"
        ).add_to(m_kmeans)
        
    st_folium(m_kmeans, width=700, height=500, key="map_kmeans")

with tab2:
    st.header("Algorithme DBSCAN (Densité)")
    st.info("DBSCAN détecte les formes arbitraires. Les points noirs sont considérés comme du bruit.")
    
    col1, col2 = st.columns(2)
    with col1:
        eps_val = st.slider("Rayon de voisinage (mètres)", 10, 500, 120,key="eps_dbscan")
    with col2:
        min_samples_val = st.slider("Nb min de points", 50, 200, 100,key="min_samples_dbscan")
        
    # Exécution
    res_dbscan = compute_dbscan_xy_labels(df.copy(), eps=eps_val, min_samples=min_samples_val)[0]
    
    # Carte
    m_dbscan = folium.Map(location=[45.75, 4.85], zoom_start=12)
    
    subset_db = res_dbscan.head(2000)
    
    for idx, row in subset_db.iterrows():
        c_color = get_color(row['cluster'])
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=3 if row['cluster'] != -1 else 1, # Petit point si c'est du bruit
            color=c_color,
            fill=True,
            fill_opacity=0.6,
            popup=f"Cluster DBSCAN: {row['cluster']}"
        ).add_to(m_dbscan)
        
    st_folium(m_dbscan, width=700, height=500, key="map_dbscan")
with tab3:
    st.header("Algorithme DBSCAN + K-Means (pour gros clusters)")
    st.info("DBSCAN détecte les formes arbitraires. Les points noirs sont considérés comme du bruit.")
    
    col1, col2 = st.columns(2)
    with col1:
        eps_val = st.slider("Rayon de voisinage (mètres)", 10, 500, 120,key="eps_dbscan_kmeans")
    with col2:
        min_samples_val = st.slider("Nb min de points", 50, 200, 100,key="min_samples_dbscan_kmeans")
        
    # Exécution
    res_dbscan = compute_dbscan_with_kmeans_split(df.copy(), eps=eps_val, min_samples=min_samples_val, max_cluster_size=5000)[0]
    
    # Carte
    m_dbscan = folium.Map(location=[45.75, 4.85], zoom_start=12)
    
    subset_db = res_dbscan.head(2000)
    
    for idx, row in subset_db.iterrows():
        c_color = get_color(row['cluster'])
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=3 if row['cluster'] != -1 else 1, # Petit point si c'est du bruit
            color=c_color,
            fill=True,
            fill_opacity=0.6,
            popup=f"Cluster DBSCAN: {row['cluster']}"
        ).add_to(m_dbscan)
        
    st_folium(m_dbscan, width=700, height=500, key="map_dbscan")



# Ajout du Rectangle Rouge (Zone)
if show_rectangle:
    folium.Rectangle(
        bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
        color="red",
        fill=False,
        weight=3,
        popup="Zone définie pour Lyon"
    ).add_to(m)