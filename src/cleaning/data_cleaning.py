
import pandas as pd
import matplotlib.pyplot as plt

from clean_geo import filter_lyon
from clean_dates import remove_inconsistent_dates, drop_upload_columns, remove_buggy_rows
from clean_duplicates import filter_same_picture
from utils import save_cleaned_data


data = pd.read_csv("data/raw/flickr_data2.csv", skipinitialspace=True)
data.columns = data.columns.str.strip() # enlève les espaces dans les noms de colonnes
# data = pd.read_csv("sample_data.csv") à décommenter pour tester avec un échantillon plus petit


# PHASE 1 : exploration des données
def exploratory_analysis(df: pd.DataFrame) -> None:

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



def main():
    # data = load_data()

    exploratory_analysis(data)

    # PHASE 2 : test des fonctions de nettoyage

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


if __name__ == '__main__':
    main()


# à décommenter quand la fonction de Omar marchera correctement
# def apply_all_filters(df: pd.DataFrame, apply_buggy: bool = False) -> pd.DataFrame:
#     """Apply all cleaning filters in a stable order and return cleaned DataFrame.

#     Order applied:
#     1. filter_lyon
#     2. filter_same_picture
#     3. remove_inconsistent_dates
#     4. (optional) remove_buggy_rows if apply_buggy is True
#     5. drop_upload_columns
#     """
#     # work on a copy to avoid side-effects
#     cleaned = df.copy()

#     # 1) zone filter
#     cleaned = filter_lyon(cleaned)

#     # 2) duplicate removal
#     cleaned = filter_same_picture(cleaned)

#     # 3) remove inconsistent dates
#     cleaned = remove_inconsistent_dates(cleaned)

#     # 4) optional buggy rows removal
#     if apply_buggy:
#         cleaned = remove_buggy_rows(cleaned)

#     # 5) drop upload date columns
#     cleaned = drop_upload_columns(cleaned)

#     return cleaned
