import pandas as pd

# FILTRE n°5 : supprimer les photos prises au même endroit par le même utilisateur le même jour

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
