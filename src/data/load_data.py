### load_data.py

### Imports
import pandas as pd
import os

#Load data into a pandas dataframe
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)