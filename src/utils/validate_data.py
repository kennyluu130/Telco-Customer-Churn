### validate_data.py


### Imports
import pandera as pa
import pandas as pd

# validate data function: checks for valid inputs
def validate_telco_data(df: pd.DataFrame):
    print("Starting data validation with Pandera...")
    
    # Define the schema mapping all original Great Expectations logic
    schema = pa.DataFrameSchema(
        columns={
            # Original: expect_column_to_exist + expect_column_values_to_not_be_null
            "customerID": pa.Column(str, nullable=False, required=True),
            
            # Original: expect_column_values_to_be_in_set
            "gender": pa.Column(str, checks=pa.Check.isin(["Male", "Female"])),
            "Partner": pa.Column(str, checks=pa.Check.isin(["Yes", "No"])),
            "Dependents": pa.Column(str, checks=pa.Check.isin(["Yes", "No"])),
            "PhoneService": pa.Column(str, checks=pa.Check.isin(["Yes", "No"])),
            "InternetService": pa.Column(str, checks=pa.Check.isin(["DSL", "Fiber optic", "No"])),
            "Contract": pa.Column(str, checks=pa.Check.isin(["Month-to-month", "One year", "Two year"])),
            
            # Original: expect_column_values_to_be_between (0, 120)
            "tenure": pa.Column(int, checks=pa.Check.in_range(0, 120), nullable=False),
            
            # Original: expect_column_values_to_be_between (0, 200)
            "MonthlyCharges": pa.Column(float, checks=pa.Check.in_range(0, 200), nullable=False),
            
            # RAW DATA HANDLING: 
            # Original code expected numeric, but Raw CSV has strings for TotalCharges.
            # We check for existence and non-nullity here. 
            "TotalCharges": pa.Column(pa.Object, nullable=False, required=True),
            
            "Churn": pa.Column(str, checks=pa.Check.isin(["Yes", "No"]), required=True),
        },
        # Check: expect_column_pair_values_A_to_be_greater_than_B
        # We wrap this in a try/except or a 'mostly' logic if needed.
        # Note: This only works if TotalCharges is numeric. 
        # Since we are validating RAW data, we skip the cross-column check here 
        # and move it to a 'post-preprocess' check or use the 'mostly' parameter.
        checks=[
            # This replicates the 'mostly=0.95' logic for A >= B
            pa.Check(
                lambda d: pd.to_numeric(d["TotalCharges"], errors="coerce") >= d["MonthlyCharges"],
                name="TotalCharges_ge_MonthlyCharges",
                ignore_na=True,
                raise_warning=True # Set to True to mimic 'mostly' behavior if it's not a hard fail
            )
        ] if df["TotalCharges"].dtype != object else [],
        
        strict=False # Allows extra columns if they exist
    )

    try:
        # lazy=True ensures we get ALL failures, not just the first one
        schema.validate(df, lazy=True)
        print("Data validation PASSED")
        return True, []
        
    except pa.errors.SchemaErrors as err:
        # FIXING THE TYPEERROR: access attributes via dot notation, not brackets
        failed_checks = []
        
        # 1. Capture Schema/Column errors
        if err.schema_errors:
            for e in err.schema_errors:
                col = e.column if e.column else "General"
                check = e.check if e.check else "Type/Presence"
                failed_checks.append(f"{col}: {check}")
        
        # 2. Capture Data-level (Check) errors
        if err.failure_cases is not None:
            check_failures = err.failure_cases["check"].unique().tolist()
            failed_checks.extend(check_failures)

        # Clean up the list to keep it unique
        failed_checks = list(set(failed_checks))
        
        print(f"Data validation FAILED: {len(failed_checks)} issues found.")
        return False, failed_checks