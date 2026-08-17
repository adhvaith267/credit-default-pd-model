from __future__ import annotations

import numpy as np
import pandas as pd

import shap
from xgboost import XGBClassifier

from financial_risk_analyst_ml.preprocessing import NUMERIC_FEATURES


def compute_shap_values(
    model: XGBClassifier,
    X: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """
    Compute SHAP values for an XGBoost model.

    Uses the fast TreeExplainer, which is exact for tree-based models
    and does not require sampling.

    Parameters
    ----------
    model:
        A fitted XGBClassifier.
    X:
        Feature matrix (n_samples, n_features).

    Returns
    -------
    shap_values: np.ndarray of shape (n_samples, n_features)
        SHAP values for the positive class (probability of default).
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # TreeExplainer on XGBClassifier (binary:logistic) returns a single
    # 2-D array already. Guard against the rare case of a list output.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values


def global_feature_importance(
    model: XGBClassifier,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values as global feature importance.

    This is the standard SHAP-based importance: for each feature, take the
    mean of the absolute SHAP values across all samples. Features that
    consistently push the prediction in either direction score higher.

    Parameters
    ----------
    model:
        A fitted XGBClassifier.
    X:
        Feature matrix used to compute SHAP values.
    feature_names:
        Column names. Defaults to NUMERIC_FEATURES from preprocessing.

    Returns
    -------
    DataFrame with columns ['feature', 'mean_abs_shap'] sorted descending.
    """

    if feature_names is None:
        feature_names = NUMERIC_FEATURES

    shap_values = compute_shap_values(model, X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return importance_df


def explain_single_borrower(
    model: XGBClassifier,
    x: np.ndarray | pd.Series,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Explain a single borrower's PD prediction using SHAP.

    Returns a DataFrame showing which features pushed the prediction
    up (positive SHAP) or down (negative SHAP) relative to the
    model's average prediction.

    Parameters
    ----------
    model:
        A fitted XGBClassifier.
    x:
        A single borrower's feature vector (1-D).
    feature_names:
        Column names. Defaults to NUMERIC_FEATURES from preprocessing.

    Returns
    -------
    DataFrame with columns ['feature', 'value', 'shap_value'] sorted by
    abs(shap_value) descending so the most impactful features appear first.
    """

    if feature_names is None:
        feature_names = NUMERIC_FEATURES

    x_array = np.asarray(x).reshape(1, -1)
    shap_values = compute_shap_values(model, x_array)

    explanation_df = pd.DataFrame(
        {
            "feature": feature_names,
            "value": x_array[0],
            "shap_value": shap_values[0],
        }
    )

    explanation_df["abs_shap"] = explanation_df["shap_value"].abs()
    explanation_df = (
        explanation_df
        .sort_values("abs_shap", ascending=False)
        .drop(columns=["abs_shap"])
        .reset_index(drop=True)
    )

    return explanation_df


def shap_summary_dict(
    model: XGBClassifier,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
    top_n: int = 10,
) -> list[dict]:
    """
    Return the top N most important features as a list of dicts.

    Suitable for serialising to JSON and storing as an evaluation artifact.

    Parameters
    ----------
    model:
        A fitted XGBClassifier.
    X:
        Feature matrix.
    feature_names:
        Column names. Defaults to NUMERIC_FEATURES.
    top_n:
        How many top features to include.

    Returns
    -------
    List of dicts: [{"feature": ..., "mean_abs_shap": ...}, ...]
    """

    importance_df = global_feature_importance(model, X, feature_names)
    top = importance_df.head(top_n)

    return [
        {
            "feature": row["feature"],
            "mean_abs_shap": round(float(row["mean_abs_shap"]), 6),
        }
        for _, row in top.iterrows()
    ]
