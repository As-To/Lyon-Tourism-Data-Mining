
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("flickr_data2.csv", skipinitialspace=True)
data.columns = data.columns.str.strip() # enlève les espaces dans les noms de colonnes
# data = pd.read_csv("sample_data.csv") à décommenter pour tester avec un échantillon plus petit


# Phase 1 : analyse exploratoire des données
# affichage des informations générales et de statistiques descriptives pour comprendre la structure des données
print(data.info())

# bornes lat/long
lat_min, lat_max = data["lat"].min(), data["lat"].max()
lon_min, lon_max = data["long"].min(), data["long"].max()

# nombre d’utilisateurs uniques
nb_users = data["user"].nunique()
nb_annees = data["date_taken_year"].nunique()

# bornes des années de prise et d'upload des photos
year_min, year_max = data["date_taken_year"].min(), data["date_taken_year"].max()
year_min_upload, year_max_upload = data["date_upload_year"].min(), data["date_upload_year"].max()

# nombre de photos par année
photos_par_annee = data["date_taken_year"].value_counts().sort_index()
photos_par_upload_annee = data["date_upload_year"].value_counts().sort_index()

# affichage des résultats de l'analyse exploratoire
print(lat_min, lat_max, lon_min, lon_max)
print(f"Utilisateurs uniques : {nb_users}")
print(f"Années uniques : {nb_annees}")
print(f"Années min/max : {year_min} / {year_max}")
print(photos_par_annee)
print(photos_par_upload_annee)

# histogramme du nombre de photos par année
plt.figure(figsize=(10, 4))
photos_par_annee.plot(kind="bar", color="#4a90e2")
plt.xlabel("Année")
plt.ylabel("Nombre de photos")
plt.title("Nombre de photos par année")
plt.tight_layout()
plt.savefig("photos_par_annee.png", dpi=150)
plt.show()  # on remarque que les photos sont majoritairement prises entre 2005 et 2019



# PHASE 2: nettoyage des données  (les fonctions de filtrage sont testées en fin de fichier dans le main)

# FILTRE n°1 : filtrer les données pour ne garder que celles de la zone de Lyon



def remove_inconsistent_dates(df):
    # Filter out rows where the year is greater than 2025
    df = df[df['date_taken_year'] <= 2025]
    return df

def save_cleaned_data(df, filename='cleaned_data.xlsx'):
    df.to_excel(filename, index=False)

def remove_buggy_rows(df):
    # Remove rows where the columns after the 16th are not NaN
    # Keep only the 16 first columns
    df = df[df.iloc[:, 16:].isna().all(axis=1)]
    df = df.iloc[:, :16]
    return df

''' data = pd.read_excel('flickr_data.xlsx')
before = len(data)
data = remove_inconsistent_dates(data)
data = remove_buggy_rows(data)
after = len(data)
print(f"Initial number of rows: {before}")
print(f"Number of rows after cleaning: {after}")
print(f"Removed {before - after} inconsistent or buggy rows.")
print(data.head())
#save_cleaned_data(data) 
'''



data = pd.read_csv("flickr_data2.csv")
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
    return data.loc[mask].copy()

def filter_same_picture(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Avant nettoyage. Lignes restantes : {len(df)}")

    # On crée une colonne 'jour' pour regrouper les photos du même jour
    df['date_key'] = df['date_taken_year'].astype(str) + '-' + \
                    df['date_taken_month'].astype(str) + '-' + \
                    df['date_taken_day'].astype(str)

    # On définit ton "epsilon" (ex: 0.0001 degré correspond à environ 11m)
    epsilon = 0.00001 

    # On trie pour que les photos suspectes d'être des doublons soient côte à côte
    df = df.sort_values(by=['user', 'date_key', 'latitude', 'longitude'])

    # On calcule la différence avec la ligne du dessus
    df['diff_lat'] = df['latitude'].diff().abs()
    df['diff_lon'] = df['longitude'].diff().abs()
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

    before = len(data)
    data_lyon = filter_lyon(data)
    after = len(data_lyon)

    cleaned = filter_same_picture(data)

    print(f"Lignes avant filtre Lyon : {before}")
    print(f"Lignes apres filtre Lyon : {after}")
    print(data_lyon.sort_values(by=['user', 'date_key', 'lat', 'long']).head())
    print(cleaned.head())

 # 3) supprime les dates incohérentes : garde les années entre 2005 et 2025
    print("3ème filtre : Suppression des dates incohérentes")
    data_cleaned_inconsistent = remove_inconsistent_dates(data)
    after_cleaned_inconsistent = len(data_cleaned_inconsistent)
    print(f"Initial number of rows before cleaning inconsistent: {before}")
    print(f"Number of rows after cleaning inconsistent: {after_cleaned_inconsistent}")
    print(f"Removed {before - after_cleaned_inconsistent} inconsistent rows.")
    print(data_cleaned_inconsistent.head())

#4) supprime les lignes buggées : colonnes après la 16ème # à décommenter quand Omar aura corrigé le bug
    # print("4ème filtre : Suppression des lignes buggées")
    # cleaned_buggy = remove_buggy_rows(data)
    # after_cleaned_buggy = len(cleaned_buggy)
    # print(f"Initial number of rows before buggy cleaning: {before}")
    # print(f"Number of rows after buggy cleaning: {after_cleaned_buggy}")
    # print(f"Removed {before - after_cleaned_buggy} buggy rows.")
    # print(cleaned_buggy.head())
    
    
#5) supprime les colonnes de date d'upload ( date_upload_min, date_upload_hour, date_upload_day, date_upload_month, date_upload_year)
    print("5ème filtre : Suppression des colonnes de date d'upload")
    cleaned_final = drop_upload_columns(data)
    print(cleaned_final.head())
    after_cleaned_final = len(cleaned_final)
    print(f"Initial number of rows before final cleaning: {before}")
    print(f"Number of rows after final cleaning: {after_cleaned_final}")
    print(f"Removed {before - after_cleaned_final} rows in final cleaning.")


#6) application de toutes les étapes de nettoyage sur les données de Lyon
    print("Application de toutes les étapes de nettoyage sur les données de Lyon")
    before_lyon = len(data)
    data_lyon_cleaned = filter_same_picture(data_lyon)
    data_lyon_cleaned = remove_inconsistent_dates(data_lyon_cleaned)
    #data_lyon_cleaned = remove_buggy_rows(data_lyon_cleaned) # à décommenter si on veut appliquer quand Omar aura corrigé le bug
    data_lyon_cleaned = drop_upload_columns(data_lyon_cleaned)
    after_lyon = len(data_lyon_cleaned)
    print(f"Initial number of rows in Lyon data: {before_lyon}")
    print(f"Number of rows after cleaning Lyon data: {after_lyon}")
    print(f"Removed {before_lyon - after_lyon} rows in Lyon data cleaning.")
    print(data_lyon_cleaned.head())

    # sauvegarde des données nettoyées : une fois que tous nos filtres seront validés 
    #save_cleaned_data(data_lyon_cleaned, filename='cleaned_lyon_data.xlsx')
