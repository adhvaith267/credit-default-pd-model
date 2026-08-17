from __future__ import annotations

import numpy as np
import pandas as pd


def add_gmsc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interpretable credit-risk features for GMSC.

    This function performs deterministic feature construction only.
    It does not learn parameters from the dataset.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # Missingness indicators
    # ---------------------------------------------------------

    if "MonthlyIncome" in df.columns:
        df["MonthlyIncome_missing"] = (
            df["MonthlyIncome"].isna().astype(int)
        )

    if "NumberOfDependents" in df.columns:
        df["NumberOfDependents_missing"] = (
            df["NumberOfDependents"].isna().astype(int)
        )

    # ---------------------------------------------------------
    # Delinquency features
    # ---------------------------------------------------------

    delinquency_columns = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]

    if all(column in df.columns for column in delinquency_columns):

        df["TotalDelinquencyCount"] = (
            df[delinquency_columns].sum(axis=1)
        )

        df["HasDelinquency"] = (
            df["TotalDelinquencyCount"] > 0
        ).astype(int)

        df["SevereDelinquency"] = (
            df["NumberOfTimes90DaysLate"] > 0
        ).astype(int)

    # ---------------------------------------------------------
    # Credit-line utilization features
    # ---------------------------------------------------------

    if {
        "NumberRealEstateLoansOrLines",
        "NumberOfOpenCreditLinesAndLoans",
    }.issubset(df.columns):

        denominator = (
            df["NumberOfOpenCreditLinesAndLoans"]
            .replace(0, np.nan)
        )

        df["RealEstateLoanRatio"] = (
            df["NumberRealEstateLoansOrLines"]
            / denominator
        )

    return df
