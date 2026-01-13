### test_pipeline_phase1.py

# Imports
import os
import pandas as pd

# Make sure Python can find src
import sys
sys.path.append(os.path.abspath("src"))

# Imports from src
from data.load_data import load_data
from data.preprocess import preprocess_data
from features.build_features import build_features

# config
DATA_PATH = "C:\Users\kemin\Documents\Projects\Telco-Customer-Churn\data\raw"
TARGET_COL = "Churn"

def main():
    
    print("Testing Phase 1: Load, Preprocess, Build Features")

    # 1. Load Data
    print("\n[1] Loading data...")
    df = load_data(DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head())

    # 2. Preprocess
    print("\n[2] Preprocessing data...")
    df_clean = preprocess_data(df, target_col=TARGET_COL)
    print(f"Data after preprocessing. Shape: {df_clean.shape}")
    print(df_clean.head())

    # 3. Build Features
    print("\n[3] Building features...")
    df_features = build_features(df_clean, target_col=TARGET_COL)
    print(f"Data after feature engineering. Shape: {df_features.shape}")
    print(df_features.head())

    print("\nPhase 1 pipeline completed successfully!")

if __name__ == "__main__":
    main()