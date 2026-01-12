import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from src.cleaning.clean_geo import filter_lyon
from src.cleaning.clean_duplicates import filter_same_picture

# 1. Configuration de la page
st.set_page_config(page_title="Projet Lyon - Visualisation", layout="wide")
st.title(" Analyse des zones d'intérêt à Lyon")
 
# 2. Chargement des données (Cache pour éviter de recharger à chaque clic)
@st.cache_data
def load_data():
    df = pd.read_csv("data/sample/sample_data.csv")
    
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

# Option A : Choisir quel jeu de données afficher
filter_mode = st.sidebar.radio(
    "Choisissez la vue :",
    ("Données Brutes (Tout)", 
     "Filtrage Zone (Lyon uniquement)", 
     "Filtrage Complet (Zone + Doublons)")
)

# Option B : Afficher le rectangle visuel
show_rectangle = st.sidebar.checkbox("Afficher la zone de Lyon (Rectangle Rouge)", value=True)

# 5. Logique de traitement des données selon le bouton choisi
data_to_display = pd.DataFrame()
map_center = [45.75, 4.85] # Centre de Lyon

if filter_mode == "Données Brutes (Tout)":
    data_to_display = df
    st.info(f"Affichage de tous les points bruts : {len(df)} photos.")

elif filter_mode == "Filtrage Zone (Lyon uniquement)":
    data_to_display = filter_lyon(df)
    st.success(f"Points dans la zone de Lyon : {len(data_to_display)} photos.")

elif filter_mode == "Filtrage Complet (Zone + Doublons)":
    # 1. Filtre Zone
    in_zone = filter_lyon(df)
    
    # 2. Retire les doublons
    data_to_display = filter_same_picture(in_zone)
        
    st.success(f"Données finales nettoyées : {len(data_to_display)} photos.")

# 6. Création de la carte Folium
m = folium.Map(location=map_center, zoom_start=11)

# Ajout du Rectangle Rouge (Zone)
if show_rectangle:
    folium.Rectangle(
        bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
        color="red",
        fill=False,
        weight=3,
        popup="Zone définie pour Lyon"
    ).add_to(m)

# Ajout des points (Utilisation de MarkerCluster pour la performance)
if not data_to_display.empty:
    subset = data_to_display
    
    mc = MarkerCluster()
    for idx, row in subset.iterrows():
        folium.Marker(
            location=[row['lat'], row['long']],
            popup=f"User: {row['user']}",
        ).add_to(mc)
    mc.add_to(m)

# 7. Affichage dans Streamlit
st_folium(m, width=800, height=600)