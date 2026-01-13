### prepare_processed_data.py

# Imports
import os, sys
import pandas as pd

# make src importable from this directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

RAW = "data/raw/Telco-Customer-Churn.csv"
OUT = "data/processed/telco_churn_processed.csv"

###
# preprocessing pipeline
###

df = pd.read_csv(RAW) #load raw

df = preprocess_data(df, target_col="Churn") #preprocess

if "Churn" in df.columns and df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1}).astype("Int64") # ensure target is 0/1 only if still object

df_processed = build_features(df, target_col="Churn") # features engineering

os.makedirs(os.path.dirname(OUT), exist_ok=True) # save data
df_processed.to_csv(OUT, index=False)
print(f"✅ Processed dataset saved to {OUT} | Shape: {df_processed.shape}")

### End