import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
from pyproj import Transformer

# Importation de tes algos (Assure-toi que les fichiers sont dans src/algo/)
# Note: J'adapte les imports pour qu'ils utilisent les fonctions que tu m'as montrées
from src.algo.K_means import run_kmeans 
from src.algo.dbscan_v2 import compute_dbscan_with_kmeans_split
from src.algo.hc import divisional_clustering 

# --- CONFIGURATION ---
st.set_page_config(page_title="Lyon Clustering", layout="wide")

# --- 1. CHARGEMENT ET PROJECTION (Optimisé : exécuté 1 seule fois) ---
# on isole le chargement et la préparation des données pour eviter que streamlit le refasse à chaque interaction
# on met le résultat dans un cache
@st.cache_data
def load_and_prep_data():
    
    df = pd.read_csv("data/cleaned/cleaned_lyon_data.csv") 
    
    # Nettoyage de base pour éviter les crashs
    df = df.dropna(subset=['lat', 'long'])
    
    # PROJECTION UNIQUE (WGS84 -> Lambert-93)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    X, Y = transformer.transform(df['long'].values, df['lat'].values)
    
    df['X'] = X
    df['Y'] = Y
    
    # On prépare le tableau numpy pour les algos sklearn
    points = np.column_stack((X, Y))
    
    return df, points

try:
    df_global, points_global = load_and_prep_data()
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# --- 2. FONCTIONS UTILITAIRES ---
def get_color(cluster_id):
    if cluster_id == -1: return 'black'
    # Correction du bug float/int
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'pink']
    return colors[int(cluster_id) % len(colors)]

# --- 3. INTERFACE ---
st.title("📍 Comparaison des Algorithmes de Clustering à Lyon")

tab1, tab2, tab3 = st.tabs(["K-Means", "DBSCAN (Avancé)", "Hiérarchique (Divisif)"])

# === ONGLET 1 : K-MEANS ===
with tab1:
    st.header("K-Means Standard")
    k_val = st.slider("Nombre de clusters (k)", 2, 50, 25, key="km_slider")
    
    # On utilise ta fonction run_kmeans de K_means.py
    # Attention: ta fonction attend 'points', 'k', 'df'
    # On travaille sur une copie pour ne pas casser le df global
    df_km = df_global.copy()
    df_km = run_kmeans(points_global, k_val, df_km)
    
    # Carte
    m = folium.Map(location=[45.75, 4.85], zoom_start=12)
    subset = df_km.head(2000) # Limite pour fluidité
    
    for _, row in subset.iterrows():
        folium.CircleMarker(
            [row['lat'], row['long']], radius=3, color=get_color(row['cluster']),
            fill=True, fill_opacity=0.7, popup=str(row.get('title', ''))
        ).add_to(m)
    st_folium(m, width=700, height=500, key="map_km")

# === ONGLET 2 : DBSCAN v2 ===
# --- Dans l'onglet DBSCAN ---
with tab2:
    st.header("Algorithme DBSCAN")
    
    # On initialise un état dans la mémoire de l'app si il n'existe pas
    if 'db_result' not in st.session_state:
        st.session_state.db_result = None

    col1, col2 = st.columns(2)
    eps = col1.slider("Rayon (mètres)", 50, 500, 100, key="db_eps")
    min_s = col2.slider("Min Points", 10, 200, 100, key="db_min")

    # Le bouton déclenche le calcul ET stocke le résultat
    if st.button("Calculer DBSCAN"):
        with st.spinner("Calcul en cours..."):
            
            df_db, _, _, _ = compute_dbscan_with_kmeans_split(
                df_global.copy(), eps=eps, min_samples=min_s
            )
            st.session_state.db_result = df_db

    # On affiche la carte SEULEMENT si un résultat existe en mémoire
    if st.session_state.db_result is not None:
        m2 = folium.Map(location=[45.75, 4.85], zoom_start=12)
        # Affichage des points stockés
        subset2 = st.session_state.db_result.head(2000)
        for _, row in subset2.iterrows():
            folium.CircleMarker(
                [row['lat'], row['long']], radius=3, 
                color=get_color(row['cluster']), fill=True
            ).add_to(m2)
        
        # L'identifiant 'key' doit être fixe pour éviter la disparition
        st_folium(m2, width=700, height=500, key="map_dbscan_stable")

# === ONGLET 3 : HIERARCHIQUE ===
with tab3:
    st.header("Clustering Hiérarchique Divisif")
    max_size = st.slider("Taille max par zone", 1000, 10000, 5000, step=500, key="hc_slider")
    
    if st.button("Lancer HC", key="run_hc"):
         with st.spinner("Calcul en cours..."):
            
            df_hc = df_global.copy()
            df_db = divisional_clustering(df_hc, max_size=max_size)
            st.session_state.db_result = df_db

    if st.session_state.db_result is not None:
        df_hc = st.session_state.db_result
            
        m3 = folium.Map(location=[45.75, 4.85], zoom_start=12)
        subset3 = df_hc.head(2000)
        for _, row in subset3.iterrows():
            folium.CircleMarker(
                [row['lat'], row['long']], radius=3, color=get_color(row['cluster']),
                fill=True, fill_opacity=0.7
                ).add_to(m3)
        st_folium(m3, width=700, height=500, key="map_hc")