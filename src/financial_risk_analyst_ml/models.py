"""
models.py — Default model factory functions for the GMSC PD model.

Each builder returns a freshly instantiated, unfitted classifier with
sensible defaults tuned for the Give Me Some Credit dataset.
Re-exports XGBClassifier and LGBMClassifier for convenience so callers
only need to import from this module.

The ``scale_pos_weight`` parameter in ``build_xgboost_model`` accepts a
pre-computed negative/positive ratio derived from the training labels.
When omitted the GMSC population constant is used as a fallback.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from financial_risk_analyst_ml.config import CONFIG

__all__ = [
    "LGBMClassifier",
    "LogisticRegression",
    "XGBClassifier",
    "build_logistic_model",
    "build_xgboost_model",
    "build_lightgbm_model",
]

# Approximate class imbalance ratio from the full GMSC dataset (93 % / 7 %).
_SCALE_POS_WEIGHT = 139_974 / 10_026


def build_logistic_model() -> LogisticRegression:
    """
    Build the logistic regression PD model.

    ``class_weight='balanced'`` compensates for the relatively rare
    positive class in GMSC (~7 % default rate).
    """
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=CONFIG.random_state,
    )


def build_xgboost_model(
    scale_pos_weight: float | None = None,
) -> XGBClassifier:
    """
    Build the XGBoost PD model.

    Parameters
    ----------
    scale_pos_weight:
        Ratio of negative to positive training samples
        (``n_neg / n_pos``). Pass the value computed from
        ``y_train`` for the most accurate class weighting.
        Defaults to the GMSC population ratio (~13.96).
    """
    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight if scale_pos_weight is not None else _SCALE_POS_WEIGHT,
        random_state=CONFIG.random_state,
        n_jobs=-1,
    )


def build_lightgbm_model() -> LGBMClassifier:
    """
    Build the LightGBM PD model.

    LightGBM uses a leaf-wise growth strategy which tends to be faster
    and more accurate than XGBoost's level-wise strategy on tabular data.
    ``is_unbalance=True`` handles the 93/7 class imbalance natively.
    """
    return LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        num_leaves=63,           # 2^(max_depth) - 1: fully expressive
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,        # bagging every iteration
        colsample_bytree=0.8,
        min_child_samples=20,    # regularises small leaves
        reg_alpha=0.1,           # L1 regularisation
        reg_lambda=1.0,          # L2 regularisation
        is_unbalance=True,       # equivalent to class_weight="balanced"
        objective="binary",
        metric="auc",
        random_state=CONFIG.random_state,
        n_jobs=-1,
        verbose=-1,              # suppress LightGBM stdout
    )
