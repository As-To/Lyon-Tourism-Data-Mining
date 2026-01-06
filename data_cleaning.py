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