from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from financial_risk_analyst_ml.features import add_gmsc_features


ID_COLUMNS = [
    "Unnamed: 0",
]

TARGET_COLUMN = "SeriousDlqin2yrs"

NUMERIC_FEATURES = [
    # Raw features
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    # Engineered features
    "MonthlyIncome_missing",
    "NumberOfDependents_missing",
    "TotalDelinquencyCount",
    "HasDelinquency",
    "SevereDelinquency",
    "RealEstateLoanRatio",
    "IncomePerCreditLineLoan",
    "PercentageTimePastDue",
]


class GMSCDataCleaner(BaseEstimator, TransformerMixin):
    """
    Dataset-specific cleaning for the Give Me Some Credit dataset.

    This transformer only performs deterministic cleaning.
    Learned operations such as imputation, clipping, and scaling
    are handled by the downstream sklearn pipeline.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Remove identifier.
        X = X.drop(columns=ID_COLUMNS, errors="ignore")

        # Treat impossible age as missing.
        if "age" in X.columns:
            X.loc[X["age"] <= 0, "age"] = np.nan

        # Add engineered features (deterministic, no data leakage).
        X = add_gmsc_features(X)

        # Ensure all expected columns are present.
        missing_columns = [
            column for column in NUMERIC_FEATURES
            if column not in X.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing expected GMSC columns: {missing_columns}"
            )

        return X[NUMERIC_FEATURES]


class QuantileClipper(BaseEstimator, TransformerMixin):
    """
    Clips each feature using quantiles learned from the training data.

    Default:
        lower = 0.5th percentile
        upper = 99.5th percentile
    """

    def __init__(self, lower_quantile=0.005, upper_quantile=0.995):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X_array = np.asarray(X, dtype=float)

        self.lower_bounds_ = np.nanquantile(
            X_array,
            self.lower_quantile,
            axis=0,
        )

        self.upper_bounds_ = np.nanquantile(
            X_array,
            self.upper_quantile,
            axis=0,
        )

        return self

    def transform(self, X):
        X_array = np.asarray(X, dtype=float)

        return np.clip(
            X_array,
            self.lower_bounds_,
            self.upper_bounds_,
        )


def build_preprocessing_pipeline() -> Pipeline:
    """
    Build the complete GMSC preprocessing pipeline.

    The pipeline must be fitted only on training data.
    """

    return Pipeline(
        steps=[
            ("clean", GMSCDataCleaner()),
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "clip",
                QuantileClipper(
                    lower_quantile=0.005,
                    upper_quantile=0.995,
                ),
            ),
            (
                "scaler",
                RobustScaler(),
            ),
        ]
    )


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate GMSC predictors and target.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    return X, y
