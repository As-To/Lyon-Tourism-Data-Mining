
import pandas as pd


# data = pd.read_csv("flickr_data2.csv")
data = pd.read_csv("flickr_data2.csv", skipinitialspace=True)
data.columns = data.columns.str.strip()
# data = pd.read_csv("sample_data.csv") à décommenter pour tester avec un échantillon plus petit


# version qui ne modifie pas le dataframe original
def remove_inconsistent_dates(df):
    # Filter out rows where the year is greater than 2025
    return df[df['date_taken_year'] <= 2025].copy()
    
#version originale de Omar qui modifie directement le dataframe original
# def remove_inconsistent_dates(df):
#     # Filter out rows where the year is greater than 2025
#     df = df[df['date_taken_year'] <= 2025]
#     return df

def save_cleaned_data(df, filename='cleaned_data.xlsx'):
    df.to_excel(filename, index=False)


#version qui ne modifie pas le dataframe original
def remove_buggy_rows(df):
    # Remove rows where the columns after the 16th are not NaN
    # Keep only the 16 first columns
    df_cleaned = df[df.iloc[:, 16:].isna().all(axis=1)]
    df_cleaned = df_cleaned.iloc[:, :16]
    return df_cleaned.copy()

#version de Omar qui modifie directement le dataframe original
# def remove_buggy_rows(df):
#     # Remove rows where the columns after the 16th are not NaN
#     # Keep only the 16 first columns
#     df = df[df.iloc[:, 16:].isna().all(axis=1)]
#     df = df.iloc[:, :16]
#     return df

# before = len(data)
# data = remove_inconsistent_dates(data)
# data = remove_buggy_rows(data)
# after = len(data)
# print(f"Initial number of rows: {before}")
# print(f"Number of rows after cleaning: {after}")
# print(f"Removed {before - after} inconsistent or buggy rows.")
# print(data.head())
#save_cleaned_data(data)

# création de la colonne 'date_key' pour regrouper les photos du même jour
data['date_key'] = data['date_taken_year'].astype(str) + '-' + \
                    data['date_taken_month'].astype(str) + '-' + \
                    data['date_taken_day'].astype(str)




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
    mask = data["lat"].between(45.6, 45.9) & data["long"].between(4.7, 5.0) 
    return data.loc[mask].copy() # permet d'éviter de modifier l'original, en renvoyant une copie
    # mais on devra modifier directement data_lyon par la suite quand on voudra nettoyer les données pour de bon



# filtre les photos prises au même endroit par le même utilisateur le même jour
def filter_same_picture(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Avant nettoyage. Lignes restantes : {len(df)}")

    # On crée une colonne 'jour' pour regrouper les photos du même jour
    df['date_key'] = df['date_taken_year'].astype(str) + '-' + \
                    df['date_taken_month'].astype(str) + '-' + \
                    df['date_taken_day'].astype(str)

    # On définit ton "epsilon" (ex: 0.0001 degré correspond à environ 11m)
    epsilon = 0.00001 

    # On trie pour que les photos suspectes d'être des doublons soient côte à côte
    df = df.sort_values(by=['user', 'date_key', 'lat', 'long'])

    # On calcule la différence avec la ligne du dessus
    df['diff_lat'] = df['lat'].diff().abs()
    df['diff_lon'] = df['long'].diff().abs()
    df['same_user'] = df['user'] == df['user'].shift()
    df['same_day'] = df['date_key'] == df['date_key'].shift()

    # On identifie les doublons : même user, même jour ET différence < epsilon
    is_duplicate = (df['same_user']) & \
                (df['same_day']) & \
                (df['diff_lat'] < epsilon) & \
                (df['diff_lon'] < epsilon)

    # On garde les données propres
    df_cleaned = df[~is_duplicate].drop(columns=['diff_lat', 'diff_lon', 'same_user', 'same_day'])

    print(f"Nettoyage terminé. Lignes restantes : {len(df_cleaned)}")
    print(f"Lignes supprimées : {len(df) - len(df_cleaned)}")
    return df_cleaned


if __name__ == "__main__":

# test des fonctions de nettoyage

# 1) filtre Lyon : la zone définie est un rectangle 
    print("1er filtre : Filtrage des données pour la zone de Lyon")
    before = len(data)
    data_lyon = filter_lyon(data)
    after = len(data_lyon)

    print(f"Lignes avant filtre Lyon : {before}")
    print(f"Lignes apres filtre Lyon : {after}")
    print(f"Nombre de lignes supprimées : {before - after}")
    print(data_lyon.head())

   
# 2) filtre les duplicatas de photos : même utilisateur, même jour, même position (rayon) 
    print("2ème filtre : Suppression des photos dupliquées")
    cleaned = filter_same_picture(data)

    # print(data_lyon.sort_values(by=['user', 'date_key', 'lat', 'long']).head())
    print(cleaned.head())

 # 3) supprime les dates incohérentes : supérieur à 2025
    print("3ème filtre : Suppression des dates incohérentes")
    # cleaned = remove_inconsistent_dates(cleaned)
    # print(cleaned.head())

    data_cleaned_inconsistent = remove_inconsistent_dates(data)
    # data = remove_buggy_rows(data)
    after_cleaned_inconsistent = len(data_cleaned_inconsistent)
    print(f"Initial number of rows before cleaning inconsistent: {before}")
    print(f"Number of rows after cleaning inconsistent: {after_cleaned_inconsistent}")
    print(f"Removed {before - after_cleaned_inconsistent} inconsistent rows.")
    print(data_cleaned_inconsistent.head())

#4) supprime les lignes buggées : colonnes après la 16ème non NaN
    # print("4ème filtre : Suppression des lignes buggées")
    

