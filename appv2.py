import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
from pyproj import Transformer
from src.algo.text_mining import top_terms_by_cluster, DEFAULT_STOPWORDS

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

def get_cluster_label(cluster_id, topics_dict,nb_words=1):
    """Récupère le mot le plus fréquent ou renvoie 'Zone X' par défaut"""
    if cluster_id in topics_dict and topics_dict[cluster_id]:
        # On prend le premier mot (le plus discriminant) et on le met en majuscules
        label = ""
        for word in topics_dict[cluster_id][:nb_words]:
            label += word + ","
        return label[:-1].capitalize()
    return f"Zone {cluster_id}"

# --- 3. INTERFACE ---
st.title(" Comparaison des Algorithmes de Clustering à Lyon")

tab1, tab2, tab3 = st.tabs(["K-Means", "DBSCAN (Avancé)", "Hiérarchique (Divisif)"])

# === ONGLET 1 : K-MEANS ===
# === ONGLET 1 : K-MEANS ===
with tab1:
    st.header("K-Means Standard")
    
    col1, col2 = st.columns(2)
    # Slider pour le nombre de clusters (K)
    k_val = col1.slider("Nombre de zones (k)", 50, 300, 200, key="km_slider")
    
    # Slider pour le nombre de mots dans l'étiquette
    nb_desc = col2.slider("Mots par étiquette", 1, 5, 3, key="nb_words_km")
    
    # 1. Calcul K-Means
    df_km = df_global.copy()
    df_km = run_kmeans(points_global, k_val, df_km)

    # 2. Text Mining
    # On récupère toujours 10 mots pour avoir de la marge
    with st.spinner("Analyse des descriptions..."):
        topics_km = top_terms_by_cluster(
            df_km, 
            cluster_col="cluster", 
            top_k=10,  # On en prend 10, le slider décidera combien en afficher
            stopwords=DEFAULT_STOPWORDS, 
            use_lemmas=True
        )
    
    # 3. Carte
    m = folium.Map(location=[45.75, 4.85], zoom_start=12)
    subset = df_km.head(2000) 
    
    for _, row in subset.iterrows():
        # Utilisation dynamique du slider 'nb_desc'
        cluster_label = get_cluster_label(row['cluster'], topics_km, nb_words=nb_desc)
        
        folium.CircleMarker(
            [row['lat'], row['long']], radius=3, color=get_color(row['cluster']),
            fill=True, fill_opacity=0.7, 
            popup=f"<b>{cluster_label}</b><br>(Cluster {int(row['cluster'])})"
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
        
        
        st_folium(m2, width=700, height=500, key="map_dbscan_stable")

# === ONGLET 3 : HIERARCHIQUE ===
with tab3:
    st.header("Clustering Hiérarchique Divisif")
    max_size = st.slider("Taille max par zone", 1000, 10000, 5000, step=500, key="hc_slider")
    
    if 'hc_result' not in st.session_state:
        st.session_state.hc_result = None

    if st.button("Lancer HC", key="run_hc"):
         with st.spinner("Calcul en cours..."):
            df_hc = df_global.copy()
            
            st.session_state.hc_result = divisional_clustering(df_hc, max_size=max_size)

    
    if st.session_state.hc_result is not None:
        df_display = st.session_state.hc_result
            
        m3 = folium.Map(location=[45.75, 4.85], zoom_start=12)
        subset3 = df_hc.head(2000)
        for _, row in subset3.iterrows():
            folium.CircleMarker(
                [row['lat'], row['long']], radius=3, color=get_color(row['cluster']),
                fill=True, fill_opacity=0.7
                ).add_to(m3)
        
        st_folium(m3, width=700, height=500, key="map_hc")