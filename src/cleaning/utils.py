import pandas as pd

def save_cleaned_data(df: pd.DataFrame, filename: str = 'cleaned_data.xlsx') -> None:
	df.to_excel(filename, index=False)

def save_cleaned_data_csv(df: pd.DataFrame, filename: str = 'cleaned_data.csv') -> None:
	df.to_csv(filename, index=False)