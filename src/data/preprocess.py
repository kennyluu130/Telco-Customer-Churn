### preprocess.py

### Imports
import pandas as pd

# preprocessing function
def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:

    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace for headers
    df = df.drop(columns="customerID") # drop ids
    df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1}) #target_col = 0 or 1
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") #convert to float
    df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int) #convert to 0 or 1

    #fill NA with 0 if numeric
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df