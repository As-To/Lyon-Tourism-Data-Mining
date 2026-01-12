import pandas as pd

# FILTRE n°1 : filtrer les données pour ne garder que celles de la zone de Lyon

# pour un point donné (lat, lon), retourne True si il est dans Lyon
def is_within_lyon(lat: float, lon: float) -> bool:
    """Return True if a point is inside the Lyon bounding box."""
    lyon_lat_min = 45.60
    lyon_lat_max = 45.90
    lyon_lon_min = 4.80
    lyon_lon_max = 4.95
    return lyon_lat_min <= lat <= lyon_lat_max and lyon_lon_min <= lon <= lyon_lon_max


# filtre toutes les lignes du DataFrame pour ne garder que celles situées à Lyon
def filter_lyon(data: pd.DataFrame) -> pd.DataFrame:
    mask = data["lat"].between(45.73, 45.79) & data["long"].between(4.81, 4.9)
    return data.loc[mask].copy() # permet d'éviter de modifier l'original, en renvoyant une copie
    # mais on devra modifier directement data_lyon par la suite quand on voudra nettoyer les données pour de bon


# LAT_MIN, LAT_MAX = 45.73, 45.79
# LON_MIN, LON_MAX = 4.81, 4.9

# A ne pas supprimer avant que je vérifie (Asmae) : 
# a priori : 45.7 et 45.8, 4.8 et 4.9 pour être plus restrictif
# Pour la latitude : a priori c'est ça mon référentiel 45.665668 à 45.79
# Pour la longitude : a priori c'est ça mon référentiel 4.8 à 4.9