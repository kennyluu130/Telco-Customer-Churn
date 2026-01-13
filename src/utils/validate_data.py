### validate_data.py

# Imports
import great_expectations as ge
from typing import Tuple, List

#validate data function using Great Expectations
def validate_telco_data(df):

    print("Starting data validation with Great Expectations...")
    
    # Convert pandas DataFrame to Great Expectations Dataset
    ge_df = ge.dataset.PandasDataset(df)
    
    # Customer identifier must exist (required for business operations)  
    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_be_null("customerID")
    
    # Demographic features
    ge_df.expect_column_to_exist("gender") 
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents")
    
    # Service features
    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")
    
    # Financial features
    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")
    
    # Gender must be one of expected values
    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    
    # Yes/No fields
    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
    
    # Contract types
    ge_df.expect_column_values_to_be_in_set(
        "Contract", 
        ["Month-to-month", "One year", "Two year"]
    )
    
    # Internet service types
    ge_df.expect_column_values_to_be_in_set(
        "InternetService",
        ["DSL", "Fiber optic", "No"]
    )
    

    
    # Financial features need to be non-negative 
    ge_df.expect_column_values_to_be_between("tenure", min_value=0)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)
    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)
    

    # Tenure should be reasonable 
    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    
    # Monthly charges should be within reasonable range
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
    
    # No missing values
    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")

    
    # Total charges should generally be >= Monthly charges
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95  # Allow 5% exceptions for edge cases
    )
    
    # Run Validation Suite
    print("Running complete validation suite...")
    results = ge_df.validate()
    
    # Extract failed expectations for detailed error reporting
    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)
    
    # Print validation summary
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks
    
    if results["success"]:
        print(f"Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"Failed expectations: {failed_expectations}")
    
    return results["success"], failed_expectations