### inference.py

# Imports
import os
import pandas as pd
import mlflow

# model loading
MODEL_DIR = "/app/model"

try:
    model = mlflow.pyfunc.load_model(MODEL_DIR) #load with pyfunc to ensure compatibility
    print(f"Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"Failed to load model from {MODEL_DIR}: {e}")


# Feature schema loading
try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# Feature transformation constraints

# binary feature mappings 
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1}, 
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

# numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    
    # clean column names
    df.columns = df.columns.str.strip()
    
    # Numeric Type Coercion 
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")  #Convert to numeric, replacing invalid values with NaN
            df[c] = df[c].fillna(0) # Fill NaN with 0
    
    # Binary Feature Encoding
    # Apply deterministic mappings for binary features
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )
    
    # One-Hot Encoding for Remaining Categorical Features
    # Find remaining object/categorical columns
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        # Apply one-hot encoding with drop_first=True
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    # Boolean to Integer Conversion
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # Feature Alignment with Training Schema
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df


# main prediction pipeline
def predict(input_dict: dict) -> str:

    # Convert Input to DataFrame
    df = pd.DataFrame([input_dict])
    
    # Apply Feature Transformations
    df_enc = _serve_transform(df)
    
    # Generate Model Prediction
    try:
        preds = model.predict(df_enc)
        
        # Normalize prediction output to consistent format
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  # Convert numpy array to list
            
        # Extract single prediction value (for single-row input)
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    # Convert to Business-Friendly Output
    if result == 1:
        return "Likely to churn"      # High risk
    else:
        return "Not likely to churn"  # Low risk