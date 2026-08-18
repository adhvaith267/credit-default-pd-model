from __future__ import annotations

import numpy as np
import pandas as pd


def add_gmsc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interpretable credit-risk features to a GMSC DataFrame.

    All transformations are deterministic — no parameters are learned.
    Safe to call before the train/validation/test split.

    Parameters
    ----------
    df:
        Raw GMSC DataFrame. Must contain the standard GMSC column names.

    Returns
    -------
    New DataFrame with original columns plus engineered features.
    The input is not modified.
    """

    df = df.copy()

    # --- Missingness indicators -------------------------------------------
    # Binary flags that capture whether high-missingness fields were imputed.
    # These allow the model to learn a different relationship for borrowers
    # with missing income vs. those with reported income.

    if "MonthlyIncome" in df.columns:
        df["MonthlyIncome_missing"] = df["MonthlyIncome"].isna().astype(int)

    if "NumberOfDependents" in df.columns:
        df["NumberOfDependents_missing"] = df["NumberOfDependents"].isna().astype(int)

    # --- Delinquency aggregates -------------------------------------------

    delinquency_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]

    if all(col in df.columns for col in delinquency_cols):
        df["TotalDelinquencyCount"] = df[delinquency_cols].sum(axis=1)
        df["HasDelinquency"] = (df["TotalDelinquencyCount"] > 0).astype(int)
        df["SevereDelinquency"] = (df["NumberOfTimes90DaysLate"] > 0).astype(int)

    # --- Credit-line ratios -----------------------------------------------
    # Replace zero credit lines with NaN to avoid division-by-zero.
    # Downstream imputation handles the resulting NaNs.

    credit_lines_required = {"NumberRealEstateLoansOrLines", "NumberOfOpenCreditLinesAndLoans"}
    income_required = {"MonthlyIncome", "NumberOfOpenCreditLinesAndLoans"}
    pastdue_required = {"NumberOfTime30-59DaysPastDueNotWorse", "NumberOfOpenCreditLinesAndLoans"}

    if credit_lines_required.issubset(df.columns) or income_required.issubset(df.columns):
        open_lines = df["NumberOfOpenCreditLinesAndLoans"].replace(0, np.nan)

        if credit_lines_required.issubset(df.columns):
            df["RealEstateLoanRatio"] = df["NumberRealEstateLoansOrLines"] / open_lines

        if income_required.issubset(df.columns):
            df["IncomePerCreditLineLoan"] = df["MonthlyIncome"] / open_lines

        if pastdue_required.issubset(df.columns):
            df["PercentageTimePastDue"] = (
                df["NumberOfTime30-59DaysPastDueNotWorse"] / open_lines
            )

    return df
