import pandas as pd


data = pd.read_csv("sample_data.csv")

# pour un point donné (lat, lon), retourne True si il est dans Lyon
def is_within_lyon(lat: float, lon: float) -> bool:
    """Return True if a point is inside the Lyon bounding box."""
    lyon_lat_min = 45.70
    lyon_lat_max = 45.80
    lyon_lon_min = 4.80
    lyon_lon_max = 4.95
    return lyon_lat_min <= lat <= lyon_lat_max and lyon_lon_min <= lon <= lyon_lon_max



# filtre toutes les lignes du DataFrame pour ne garder que celles situées à Lyon
def filter_lyon(data: pd.DataFrame) -> pd.DataFrame:
    mask = data[" lat"].between(45.6, 45.9) & data[" long"].between(4.7, 5.0)
    return data.loc[mask].copy()


if __name__ == "__main__":

    before = len(data)
    data_lyon = filter_lyon(data)
    after = len(data_lyon)

    print(f"Lignes avant filtre Lyon : {before}")
    print(f"Lignes apres filtre Lyon : {after}")
    print(data_lyon.head())
