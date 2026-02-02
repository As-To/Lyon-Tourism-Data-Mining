import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import numpy as np
from pyproj import Transformer
from mistralai import Mistral

# --- IMPORTS DE TES ALGOS ---
# Assure-toi que les fichiers sont bien dans src/algo/
from src.algo.text_mining import top_terms_by_cluster, apriori_by_cluster, DEFAULT_STOPWORDS
from src.algo.K_means import run_kmeans 
from src.algo.dbscan_v2 import compute_dbscan_with_kmeans_split
from src.algo.hc import divisional_clustering, agglomerative_clustering_algo

# --- CONFIGURATION ---
st.set_page_config(page_title="Lyon Clustering", layout="wide")

# --- 1. CHARGEMENT ET PROJECTION (Cache global) ---
@st.cache_data
def load_and_prep_data():
    # Chargement
    df = pd.read_csv("data/cleaned/cleaned_lyon_data.csv") 
    
    # Nettoyage
    df = df.dropna(subset=['lat', 'long'])
    
    # Projection (WGS84 -> Lambert-93)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    X, Y = transformer.transform(df['long'].values, df['lat'].values)
    
    df['X'] = X
    df['Y'] = Y
    
    # Tableau numpy pour Scikit-Learn
    points = np.column_stack((X, Y))
    
    return df, points

try:
    df_global, points_global = load_and_prep_data()
except Exception as e:
    st.error(f"Erreur de chargement des données : {e}")
    st.stop()

# --- 2. WRAPPERS DE CALCUL AVEC CACHE (Pour la rapidité) ---
# C'est ici que l'optimisation se fait : on ne recalcule pas si les paramètres ne changent pas.

@st.cache_data
def cached_kmeans_execution(points, k, df_input):
    """Exécute K-Means et le Text Mining, mis en cache."""
    # Clustering
    df_res = run_kmeans(points, k, df_input.copy())
    
    # Text Mining (Pré-calculé pour être rapide à l'affichage)
    topics = top_terms_by_cluster(
        df_res, cluster_col="cluster", top_k=10, 
        stopwords=DEFAULT_STOPWORDS, use_lemmas=True
    )
    return df_res, topics

@st.cache_data
def cached_dbscan_execution(df_input, eps, min_s):
    """Exécute DBSCAN et le Text Mining, mis en cache."""
    # Clustering (Note: compute_dbscan_with_kmeans_split gère déjà la projection interne, on passe le DF)
    df_res, _, _, _ = compute_dbscan_with_kmeans_split(
        df_input.copy(), eps=eps, min_samples=min_s
    )
    
    # Text Mining
    topics = top_terms_by_cluster(
        df_res, cluster_col="cluster", top_k=10, 
        stopwords=DEFAULT_STOPWORDS, use_lemmas=True
    )
    return df_res, topics

@st.cache_data
def cached_hc_divisional(df_input, max_size):
    """Exécute HC Divisif et Text Mining."""
    df_res = divisional_clustering(df_input.copy(), max_size=max_size)
    topics = top_terms_by_cluster(
        df_res, cluster_col="cluster", top_k=10, 
        stopwords=DEFAULT_STOPWORDS, use_lemmas=True
    )
    return df_res, topics

@st.cache_data
def cached_hc_agglomerative(df_input, n_clusters, sample_size):
    """Exécute HC Agglomératif et Text Mining."""
    df_res = agglomerative_clustering_algo(
        df_input.copy(), n_clusters=n_clusters, sample_size=sample_size
    )
    topics = top_terms_by_cluster(
        df_res, cluster_col="cluster", top_k=10, 
        stopwords=DEFAULT_STOPWORDS, use_lemmas=True
    )
    return df_res, topics

# --- 3. FONCTIONS UTILITAIRES ---

def get_color(cluster_id):
    if cluster_id == -1: return 'black'
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'pink']
    return colors[int(cluster_id) % len(colors)]

def get_cluster_label(cluster_id, topics_dict, nb_words=1):
    """Récupère les mots-clés pour l'étiquette de la carte."""
    if cluster_id in topics_dict and topics_dict[cluster_id]:
        # On prend les 'nb_words' premiers mots
        words = topics_dict[cluster_id][:nb_words]
        label = ", ".join(words)
        return label.capitalize()
    return f"Zone {cluster_id}"

def ask_llm_description(cluster_id, terms, rules):
    api_key = st.secrets.get("MISTRAL_API_KEY")
    if not api_key: 
        return "⚠️ Clé Mistral manquante dans .streamlit/secrets.toml"
    
    # Initialisation du client
    client = Mistral(api_key=api_key)
    
    # Construction du prompt (identique à vos versions précédentes)
    prompt = f"""
    Tu es un expert touristique de la ville de Lyon. 
    J'ai un cluster de photos géolocalisées. Voici les indices :
    
    1. MOTS-CLÉS (TF-IDF) : {', '.join(terms[:10])}
    
    2. RÈGLES D'ASSOCIATION (Apriori) :
    {chr(10).join(rules[:5]) if rules else "Aucune règle forte détectée."}
    
    Tâche :
    Devine de quel lieu il s'agit, donne un titre et une description courte.
    """

    try:
        
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Erreur API Mistral : {e}"

def show_cluster_analysis(df_clustered,algo_type):
    """Affiche le panneau d'analyse détaillée en bas de page."""
    st.markdown("---")
    st.subheader("Analyse détaillée d'une Zone (IA & Text Mining)")
    
    # Liste des clusters (sans le bruit -1)
    if df_clustered is None or 'cluster' not in df_clustered.columns:
        st.info("Lancez d'abord un calcul pour voir l'analyse.")
        return

    cluster_ids = sorted([c for c in df_clustered['cluster'].unique() if c != -1])
    
    if not cluster_ids:
        st.warning("Aucun cluster valide trouvé.")
        return

    selected_cluster = st.selectbox("Choisir un cluster à analyser :", cluster_ids, key=f"sel_{algo_type}_{len(df_clustered)}")
    
    if selected_cluster is not None:
        col1, col2 = st.columns(2)
        
        # --- PARTIE 1 : TEXT MINING ---
        with col1:
            st.markdown("### 📊 Données Textuelles")
            
            # Recalcul ou récupération rapide pour l'affichage détaillé
            # (On ne met pas ça en cache global car c'est interactif et rapide sur un seul cluster)
            topics = top_terms_by_cluster(df_clustered, cluster_col="cluster", top_k=15, stopwords=DEFAULT_STOPWORDS)
            current_terms = topics.get(selected_cluster, [])
            
            st.caption("Mots les plus fréquents (TF-IDF)")
            st.info(", ".join(current_terms[:10]))
            
            st.caption("Associations fréquentes (Apriori)")
            itemsets, rules_dict = apriori_by_cluster(
                df_clustered[df_clustered['cluster'] == selected_cluster], 
                cluster_col="cluster", min_support=0.05, min_confidence=0.5
            )
            
            current_rules_display = []
            if selected_cluster in rules_dict and rules_dict[selected_cluster]:
                for r in rules_dict[selected_cluster]:
                    rule_str = f"{r['antecedent']} ➡ {r['consequent']} (Conf: {r['confidence']:.2f})"
                    st.write(f"- {rule_str}")
                    current_rules_display.append(rule_str)
            else:
                st.write("Pas de règles fortes trouvées.")

        # --- PARTIE 2 : LLM ---
        with col2:
            st.markdown("### 🤖 Description par l'IA")
            if st.button(f"Demander à l'IA de décrire la Zone {selected_cluster}"):
                with st.spinner("L'IA analyse les données..."):
                    description = ask_llm_description(selected_cluster, current_terms, current_rules_display)
                    st.success("Analyse terminée !")
                    st.markdown(description)

# --- 4. INTERFACE PRINCIPALE ---
st.title("📍 Comparaison des Algorithmes de Clustering à Lyon")

tab1, tab2, tab3 = st.tabs(["K-Means", "DBSCAN (Avancé)", "Hiérarchique"])

# === ONGLET 1 : K-MEANS ===
with tab1:
    st.header("K-Means Standard")
    
    c1, c2 = st.columns(2)
    k_val = c1.slider("Nombre de zones (k)", 10, 300, 50, key="km_k")
    nb_desc = c2.slider("Mots par étiquette", 1, 5, 2, key="km_words")
    
    # APPEL CACHÉ (Rapide !)
    with st.spinner("Calcul K-Means en cours..."):
        df_km, topics_km = cached_kmeans_execution(points_global, k_val, df_global)
    
    # CARTE OPTIMISÉE (MarkerCluster)
    m = folium.Map(location=[45.75, 4.85], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(m)
    
    # On limite l'affichage à 2000 points pour éviter le lag navigateur, 
    # mais le clustering est fait sur tout le dataset.
    subset = df_km.sample(n=min(2000, len(df_km)), random_state=42)
    
    for _, row in subset.iterrows():
        label = get_cluster_label(row['cluster'], topics_km, nb_words=nb_desc)
        folium.CircleMarker(
            [row['lat'], row['long']], radius=5, color=get_color(row['cluster']),
            fill=True, fill_opacity=0.7,
            popup=f"<b>{label}</b><br>Cluster {int(row['cluster'])}"
        ).add_to(marker_cluster) # Ajout au cluster, pas à la carte directement
        
    st_folium(m, width=700, height=500, key="map_km")
    
    # Analyse détaillée
    show_cluster_analysis(df_km,"K_Means")

# === ONGLET 2 : DBSCAN ===
with tab2:
    st.header("DBSCAN + K-Means Split")
    
    if 'db_data' not in st.session_state:
        st.session_state.db_data = None
        st.session_state.db_topics = None

    c1, c2 = st.columns(2)
    eps = c1.slider("Rayon (mètres)", 20, 300, 100, key="db_eps")
    min_s = c2.slider("Min Points", 5, 100, 50, key="db_min")
    
    if st.button("Lancer DBSCAN", key="run_db"):
        with st.spinner("Calcul DBSCAN..."):
            # Appel caché
            df_res, topics_res = cached_dbscan_execution(df_global, eps, min_s)
            st.session_state.db_data = df_res
            st.session_state.db_topics = topics_res

    if st.session_state.db_data is not None:
        nb_desc_db = st.slider("Mots par étiquette", 1, 5, 2, key="db_words_slider")
        
        m2 = folium.Map(location=[45.75, 4.85], zoom_start=12)
        marker_cluster2 = MarkerCluster().add_to(m2)
        
        subset2 = st.session_state.db_data.sample(n=min(2000, len(st.session_state.db_data)), random_state=42)
        
        for _, row in subset2.iterrows():
            label = get_cluster_label(row['cluster'], st.session_state.db_topics, nb_words=nb_desc_db)
            folium.CircleMarker(
                [row['lat'], row['long']], radius=5, color=get_color(row['cluster']),
                fill=True, fill_opacity=0.7,
                popup=f"<b>{label}</b><br>Cluster {int(row['cluster'])}"
            ).add_to(marker_cluster2)
            
        st_folium(m2, width=700, height=500, key="map_db_stable")
        show_cluster_analysis(st.session_state.db_data,"DBSCAN")

# === ONGLET 3 : HIÉRARCHIQUE ===
with tab3:
    st.header("Clustering Hiérarchique")
    
    if 'hc_data' not in st.session_state:
        st.session_state.hc_data = None
        st.session_state.hc_topics = None

    method = st.radio("Méthode :", ("Top-Down (Divisif)", "Bottom-Up (Agglomératif)"))

    if method == "Top-Down (Divisif)":
        max_size = st.slider("Taille max par zone", 1000, 10000, 5000, step=500)
        if st.button("Lancer Top-Down"):
            with st.spinner("Calcul Divisif..."):
                df_res, topics_res = cached_hc_divisional(df_global, max_size)
                st.session_state.hc_data = df_res
                st.session_state.hc_topics = topics_res
                
    else: # Agglomératif
        n_cl = st.slider("Clusters finaux", 10, 100, 20)
        if st.button("Lancer Bottom-Up"):
            with st.spinner("Calcul Agglomératif (sur échantillon)..."):
                df_res, topics_res = cached_hc_agglomerative(df_global, n_cl, 5000)
                st.session_state.hc_data = df_res
                st.session_state.hc_topics = topics_res

    if st.session_state.hc_data is not None:
        nb_desc_hc = st.slider("Mots par étiquette", 1, 5, 2, key="hc_words_slider")
        
        m3 = folium.Map(location=[45.75, 4.85], zoom_start=12)
        marker_cluster3 = MarkerCluster().add_to(m3)
        
        subset3 = st.session_state.hc_data.sample(n=min(2000, len(st.session_state.hc_data)), random_state=42)
        
        for _, row in subset3.iterrows():
            label = get_cluster_label(row['cluster'], st.session_state.hc_topics, nb_words=nb_desc_hc)
            folium.CircleMarker(
                [row['lat'], row['long']], radius=5, color=get_color(row['cluster']),
                fill=True, fill_opacity=0.7,
                popup=f"<b>{label}</b>"
            ).add_to(marker_cluster3)
            
        st_folium(m3, width=700, height=500, key="map_hc_stable")
        show_cluster_analysis(st.session_state.hc_data,"HC")