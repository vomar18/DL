import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data.columns

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)
