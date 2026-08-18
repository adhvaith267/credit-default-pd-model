"""
explain.py — SHAP-based model explainability for tree-based PD models.

Provides global feature importance (mean |SHAP|) and single-borrower
explanations for XGBClassifier and LGBMClassifier.

SHAP is imported lazily inside each function so that the rest of the
package can be imported without shap being installed (e.g. during
lightweight unit tests).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from financial_risk_analyst_ml.preprocessing import NUMERIC_FEATURES

# Type alias for tree-based models supported by SHAP TreeExplainer.
TreeModel = XGBClassifier | LGBMClassifier


def compute_shap_values(
    model: TreeModel,
    X: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """
    Compute SHAP values for a tree-based model.

    Uses the fast TreeExplainer which is exact for tree-based models
    and does not require sampling.

    Parameters
    ----------
    model:
        A fitted XGBClassifier or LGBMClassifier.
    X:
        Feature matrix (n_samples, n_features).

    Returns
    -------
    shap_values: np.ndarray of shape (n_samples, n_features)
        SHAP values for the positive class (probability of default).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # LightGBM TreeExplainer returns a list [neg_class, pos_class].
    # XGBClassifier returns a single 2-D array.
    # Normalise to always return the positive-class array.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values


def global_feature_importance(
    model: TreeModel,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values as global feature importance.

    For each feature, takes the mean of the absolute SHAP values across
    all samples. Features that consistently push the prediction in either
    direction score higher.

    Parameters
    ----------
    model:
        A fitted XGBClassifier or LGBMClassifier.
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

    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def explain_single_borrower(
    model: TreeModel,
    x: np.ndarray | pd.Series,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Explain a single borrower's PD prediction using SHAP.

    Returns a DataFrame showing which features pushed the prediction
    up (positive SHAP) or down (negative SHAP) relative to the model's
    average prediction.

    Parameters
    ----------
    model:
        A fitted XGBClassifier or LGBMClassifier.
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

    return (
        pd.DataFrame({
            "feature": feature_names,
            "value": x_array[0],
            "shap_value": shap_values[0],
        })
        .assign(abs_shap=lambda df: df["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .drop(columns=["abs_shap"])
        .reset_index(drop=True)
    )


def shap_summary_dict(
    model: TreeModel,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
    top_n: int = 10,
) -> list[dict]:
    """
    Return the top N most important features as a JSON-serializable list.

    Suitable for storing as an evaluation artifact in metrics.json.

    Parameters
    ----------
    model:
        A fitted XGBClassifier or LGBMClassifier.
    X:
        Feature matrix.
    feature_names:
        Column names. Defaults to NUMERIC_FEATURES.
    top_n:
        Number of top features to include.

    Returns
    -------
    List of dicts: [{"feature": ..., "mean_abs_shap": ...}, ...]
    """
    importance_df = global_feature_importance(model, X, feature_names)

    return [
        {
            "feature": row["feature"],
            "mean_abs_shap": round(float(row["mean_abs_shap"]), 6),
        }
        for _, row in importance_df.head(top_n).iterrows()
    ]
