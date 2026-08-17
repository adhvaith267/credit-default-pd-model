from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def build_logistic_model() -> LogisticRegression:
    """
    Build the logistic regression PD model.

    class_weight='balanced' compensates for the relatively rare
    positive class in GMSC.
    """

    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )


def build_xgboost_model() -> XGBClassifier:
    """
    Build the XGBoost PD model.

    scale_pos_weight reflects the class imbalance in GMSC.
    """

    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=(139974 / 10026),
        random_state=42,
        n_jobs=-1,
    )


def build_lightgbm_model() -> LGBMClassifier:
    """
    Build the LightGBM PD model.

    LightGBM uses a leaf-wise tree growth strategy which tends to achieve
    better accuracy than XGBoost's level-wise strategy on tabular data,
    while being faster to train.

    is_unbalance=True handles the 93/7 class imbalance natively.
    """

    return LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        num_leaves=63,          # 2^(max_depth) - 1: fully expressive
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,       # bagging every iteration
        colsample_bytree=0.8,
        min_child_samples=20,   # min data in a leaf: regularises small leaves
        reg_alpha=0.1,          # L1 regularisation
        reg_lambda=1.0,         # L2 regularisation
        is_unbalance=True,      # equivalent to class_weight="balanced"
        objective="binary",
        metric="auc",
        random_state=42,
        n_jobs=-1,
        verbose=-1,             # suppress LightGBM stdout
    )

