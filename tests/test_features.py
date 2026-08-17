"""
test_features.py — Unit tests for features.py

These tests run with only pytest installed locally.
pandas is imported via `uv run --with pandas pytest` or inside SageMaker.
"""

from __future__ import annotations

import pytest

# Guard: skip entire module if pandas is not installed.
pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from financial_risk_analyst_ml.features import add_gmsc_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(**kwargs) -> pd.DataFrame:
    """Build a minimal GMSC-style single-row DataFrame."""
    defaults = {
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.3,
        "MonthlyIncome": 5000.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 2,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1.0,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# Missingness indicators
# ---------------------------------------------------------------------------

class TestMissingnessIndicators:

    def test_monthly_income_present_gives_zero(self):
        df = _make_row(MonthlyIncome=5000.0)
        result = add_gmsc_features(df)
        assert result["MonthlyIncome_missing"].iloc[0] == 0

    def test_monthly_income_missing_gives_one(self):
        df = _make_row(MonthlyIncome=float("nan"))
        result = add_gmsc_features(df)
        assert result["MonthlyIncome_missing"].iloc[0] == 1

    def test_dependents_present_gives_zero(self):
        df = _make_row(NumberOfDependents=2.0)
        result = add_gmsc_features(df)
        assert result["NumberOfDependents_missing"].iloc[0] == 0

    def test_dependents_missing_gives_one(self):
        df = _make_row(NumberOfDependents=float("nan"))
        result = add_gmsc_features(df)
        assert result["NumberOfDependents_missing"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Total delinquency
# ---------------------------------------------------------------------------

class TestTotalDelinquency:

    def test_no_delinquency(self):
        df = _make_row(
            **{
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfTimes90DaysLate": 0,
            }
        )
        result = add_gmsc_features(df)
        assert result["TotalDelinquencyCount"].iloc[0] == 0
        assert result["HasDelinquency"].iloc[0] == 0
        assert result["SevereDelinquency"].iloc[0] == 0

    def test_minor_delinquency_only(self):
        df = _make_row(
            **{
                "NumberOfTime30-59DaysPastDueNotWorse": 3,
                "NumberOfTime60-89DaysPastDueNotWorse": 1,
                "NumberOfTimes90DaysLate": 0,
            }
        )
        result = add_gmsc_features(df)
        assert result["TotalDelinquencyCount"].iloc[0] == 4
        assert result["HasDelinquency"].iloc[0] == 1
        assert result["SevereDelinquency"].iloc[0] == 0

    def test_severe_delinquency(self):
        df = _make_row(
            **{
                "NumberOfTime30-59DaysPastDueNotWorse": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfTimes90DaysLate": 2,
            }
        )
        result = add_gmsc_features(df)
        assert result["TotalDelinquencyCount"].iloc[0] == 3
        assert result["HasDelinquency"].iloc[0] == 1
        assert result["SevereDelinquency"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Real estate loan ratio
# ---------------------------------------------------------------------------

class TestRealEstateLoanRatio:

    def test_normal_ratio(self):
        df = _make_row(
            NumberRealEstateLoansOrLines=2,
            NumberOfOpenCreditLinesAndLoans=8,
        )
        result = add_gmsc_features(df)
        ratio = result["RealEstateLoanRatio"].iloc[0]
        assert abs(ratio - 0.25) < 1e-9

    def test_zero_denominator_gives_nan(self):
        df = _make_row(
            NumberRealEstateLoansOrLines=2,
            NumberOfOpenCreditLinesAndLoans=0,
        )
        result = add_gmsc_features(df)
        assert pd.isna(result["RealEstateLoanRatio"].iloc[0])

    def test_zero_numerator(self):
        df = _make_row(
            NumberRealEstateLoansOrLines=0,
            NumberOfOpenCreditLinesAndLoans=5,
        )
        result = add_gmsc_features(df)
        assert result["RealEstateLoanRatio"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Original columns are preserved
# ---------------------------------------------------------------------------

class TestOriginalColumnsPreserved:

    def test_raw_columns_still_present(self):
        df = _make_row()
        result = add_gmsc_features(df)
        for col in [
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "DebtRatio",
            "MonthlyIncome",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_does_not_mutate_input(self):
        df = _make_row()
        original_cols = list(df.columns)
        add_gmsc_features(df)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

class TestBatchProcessing:

    def test_multiple_rows(self):
        rows = pd.DataFrame([
            {
                "RevolvingUtilizationOfUnsecuredLines": 0.5,
                "age": 45,
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.3,
                "MonthlyIncome": 5000.0,
                "NumberOfOpenCreditLinesAndLoans": 8,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 2,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 1.0,
            },
            {
                "RevolvingUtilizationOfUnsecuredLines": 0.9,
                "age": 30,
                "NumberOfTime30-59DaysPastDueNotWorse": 1,
                "DebtRatio": 1.2,
                "MonthlyIncome": float("nan"),
                "NumberOfOpenCreditLinesAndLoans": 4,
                "NumberOfTimes90DaysLate": 1,
                "NumberRealEstateLoansOrLines": 0,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": float("nan"),
            },
        ])
        result = add_gmsc_features(rows)
        assert len(result) == 2
        assert result["MonthlyIncome_missing"].iloc[0] == 0
        assert result["MonthlyIncome_missing"].iloc[1] == 1
        assert result["SevereDelinquency"].iloc[1] == 1
