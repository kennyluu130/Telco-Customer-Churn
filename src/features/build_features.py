### build_features.py

# Imports
import pandas as pd

# handles binary encoding for 2-category features
def _map_binary_series(s):

    # Get unique values and remove NaN
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)
    
    # Yes/No mapping
    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")
        
    # Gender mapping
    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1}).astype("Int64")

    # Generic mapping for other 2 category features by alphabetic order
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    return s

# build_features function transform raw data into ML-ready features
def build_features(df, target_col):

    df = df.copy()
    print(f"Starting feature engineering on {df.shape[1]} columns...")

    # Find categorical columns (object dtype) excluding the target variable
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
 
    # Binary features (exactly 2 unique values) get binary encoding
    # Multi-category features (>2 unique values) get one-hot encoding
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2] 

    # Convert 2-category features to 0/1
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))

    # Convert Boolean Columns
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    # One-Hot Encoding for Multi-Category Features
   
    if multi_cols:
        original_shape = df.shape
        
        # Apply one-hot encoding with drop_first=True
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        new_features = df.shape[1] - original_shape[1] + len(multi_cols) #dtypes = int??
        print(f"      Created {new_features} new features from {len(multi_cols)} categorical columns")

    # Convert nullable integers (Int64) to standard integers for XGBoost
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(0).astype(int) # Fill NaN values with int 0

    print(f"Feature engineering complete: {df.shape[1]} final features")
    return df