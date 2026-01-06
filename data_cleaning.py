#2) retirer les lignes avec des dates incohérentes (année dans le futur > 2025)
#3) retirer les lignes de bug (les lignes avec des colonnes en plus) 

import pandas as pd


def remove_inconsistent_dates(df):
    # Filter out rows where the year is greater than 2025
    df = df[df[' date_taken_year'] <= 2025]
    return df

def save_cleaned_data(df, filename='cleaned_data.xlsx'):
    df.to_excel(filename, index=False)

def remove_buggy_rows(df):
    # Remove rows where the columns after the 16th are not NaN
    # Keep only the 16 first columns
    df = df[df.iloc[:, 16:].isna().all(axis=1)]
    df = df.iloc[:, :16]
    return df


data = pd.read_excel('flickr_data2.xlsx')
before = len(data)
data = remove_inconsistent_dates(data)
data = remove_buggy_rows(data)
after = len(data)
print(f"Initial number of rows: {before}")
print(f"Number of rows after cleaning: {after}")
print(f"Removed {before - after} inconsistent or buggy rows.")
print(data.head())
#save_cleaned_data(data)


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
