from typing import List, Tuple

import great_expectations as ge
import pandas as pd


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate the raw Telco Customer Churn data with Great Expectations."""
    print("🔍 Starting data validation with Great Expectations...")

    # TotalCharges is stored as text in the source CSV. Validate its numeric
    # meaning without mutating the dataframe that continues through the pipeline.
    validation_df = df.copy()
    if "TotalCharges" in validation_df.columns:
        validation_df["TotalCharges"] = pd.to_numeric(
            validation_df["TotalCharges"], errors="coerce"
        )

    # GX 1.x replaced ge.dataset.PandasDataset with Data Context, Data Source,
    # Data Asset, and Batch Definition objects. An ephemeral context keeps this
    # pipeline check self-contained and avoids writing GX configuration to disk.
    context = ge.get_context(mode="ephemeral")
    context.variables.progress_bars = {
        "globally": False,
        "metric_calculations": False,
    }
    data_source = context.data_sources.add_pandas(name="telco_validation_source")
    data_asset = data_source.add_dataframe_asset(name="telco_validation_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        name="telco_validation_batch"
    )

    print("   📋 Defining schema and required-column checks...")
    expectations = [
        ge.expectations.ExpectColumnToExist(column="customerID"),
        ge.expectations.ExpectColumnValuesToNotBeNull(column="customerID"),
        ge.expectations.ExpectColumnToExist(column="gender"),
        ge.expectations.ExpectColumnToExist(column="Partner"),
        ge.expectations.ExpectColumnToExist(column="Dependents"),
        ge.expectations.ExpectColumnToExist(column="PhoneService"),
        ge.expectations.ExpectColumnToExist(column="InternetService"),
        ge.expectations.ExpectColumnToExist(column="Contract"),
        ge.expectations.ExpectColumnToExist(column="tenure"),
        ge.expectations.ExpectColumnToExist(column="MonthlyCharges"),
        ge.expectations.ExpectColumnToExist(column="TotalCharges"),
    ]

    print("   💼 Defining business-logic checks...")
    expectations.extend(
        [
            ge.expectations.ExpectColumnValuesToBeInSet(
                column="gender", value_set=["Male", "Female"]
            ),
            ge.expectations.ExpectColumnValuesToBeInSet(
                column="Partner", value_set=["Yes", "No"]
            ),
            ge.expectations.ExpectColumnValuesToBeInSet(
                column="Dependents", value_set=["Yes", "No"]
            ),
            ge.expectations.ExpectColumnValuesToBeInSet(
                column="PhoneService", value_set=["Yes", "No"]
            ),
            ge.expectations.ExpectColumnValuesToBeInSet(
                column="Contract",
                value_set=["Month-to-month", "One year", "Two year"],
            ),
            ge.expectations.ExpectColumnValuesToBeInSet(
                column="InternetService", value_set=["DSL", "Fiber optic", "No"]
            ),
        ]
    )

    print("   📊 Defining numeric and statistical checks...")
    expectations.extend(
        [
            ge.expectations.ExpectColumnValuesToBeBetween(
                column="tenure", min_value=0, max_value=120
            ),
            ge.expectations.ExpectColumnValuesToBeBetween(
                column="MonthlyCharges", min_value=0, max_value=200
            ),
            ge.expectations.ExpectColumnValuesToBeBetween(
                column="TotalCharges", min_value=0
            ),
            ge.expectations.ExpectColumnValuesToNotBeNull(column="tenure"),
            ge.expectations.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"),
            ge.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
                column_A="TotalCharges",
                column_B="MonthlyCharges",
                or_equal=True,
                mostly=0.95,
            ),
        ]
    )

    suite = ge.ExpectationSuite(name="telco_data_quality_suite")
    for expectation in expectations:
        suite.add_expectation(expectation)
    suite = context.suites.add(suite)

    validation_definition = ge.ValidationDefinition(
        name="telco_data_quality_validation",
        data=batch_definition,
        suite=suite,
    )
    validation_definition = context.validation_definitions.add(validation_definition)

    print("   ⚙️  Running complete validation suite...")
    results = validation_definition.run(
        batch_parameters={"dataframe": validation_df}
    )

    failed_expectations = [
        result.expectation_config.type
        for result in results.results
        if not result.success
    ]
    total_checks = len(results.results)
    failed_checks = len(failed_expectations)
    passed_checks = total_checks - failed_checks

    if results.success:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return bool(results.success), failed_expectations
