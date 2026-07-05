import pandas as pd

from src.utils.validate_data import validate_telco_data


def _valid_telco_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["customer-1"],
            "gender": ["Female"],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "PhoneService": ["Yes"],
            "InternetService": ["DSL"],
            "Contract": ["Month-to-month"],
            "tenure": [12],
            "MonthlyCharges": [50.0],
            # Match the raw CSV representation to exercise numeric coercion.
            "TotalCharges": ["600.0"],
        }
    )


def test_validate_telco_data_accepts_valid_raw_data():
    is_valid, failed = validate_telco_data(_valid_telco_data())

    assert is_valid is True
    assert failed == []


def test_validate_telco_data_reports_invalid_gender():
    data = _valid_telco_data()
    data.loc[0, "gender"] = "Unknown"

    is_valid, failed = validate_telco_data(data)

    assert is_valid is False
    assert "expect_column_values_to_be_in_set" in failed
