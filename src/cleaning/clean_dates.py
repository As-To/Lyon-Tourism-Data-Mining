import pandas as pd
from typing import Tuple


# FILTRE n°2 : supprimer les lignes avec des années incohérentes ( qui ne sont pas entre 2005 et 2025)

# version qui ne modifie pas le dataframe original
def remove_inconsistent_dates(df, year_min: int = 2005, year_max: int = 2025):
    """Keep only rows with date_taken_year between year_min and year_max."""
    return df[df["date_taken_year"].between(year_min, year_max)].copy()
    
#version originale de Omar qui modifie directement le dataframe original
# def remove_inconsistent_dates(df):
#     # Filter out rows where the year is greater than 2025
#     df = df[df['date_taken_year'] <= 2025]
#     return df

# FILTRE n°4 : supprimer les colonnes de date d'upload ( toutes les colonnes de la forme date_upload_...)
def drop_upload_columns(df: pd.DataFrame) -> pd.DataFrame:
	cols_to_drop = [
		"date_upload_minute",
		"date_upload_hour",
		"date_upload_day",
		"date_upload_month",
		"date_upload_year",
	]
	return df.drop(columns=cols_to_drop, errors='ignore').copy()

# FILTRE n°3 : supprimer les lignes buggées (colonnes après la 16ème non NaN)

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