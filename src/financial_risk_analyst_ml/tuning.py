"""
tuning.py — Optuna-based hyperparameter tuning for XGBoost and LightGBM.

Both tuners:
  - Optimise ROC-AUC on the validation set (no test set exposure).
  - Use the already-preprocessed X_train / X_val arrays so preprocessing
    stats are never re-fit during tuning (no leakage).
  - Return the best hyperparameters as a plain dict; the caller builds
    the final model with those params and re-trains on full training data.

Usage
-----
    from financial_risk_analyst_ml.tuning import tune_xgboost, tune_lightgbm

    best_xgb_params = tune_xgboost(X_train, y_train, X_val, y_val, n_trials=50)
    best_lgb_params = tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=50)
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Approximate class imbalance ratio from the full GMSC dataset (93 % / 7 %).
_SCALE_POS_WEIGHT = 139_974 / 10_026

# Suppress Optuna's per-trial output — we log a summary at the end instead.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# XGBoost tuning
# ---------------------------------------------------------------------------

def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    random_state: int = 42,
    scale_pos_weight: float | None = None,
) -> dict:
    """
    Search for the best XGBoost hyperparameters using Optuna.

    Optimises validation ROC-AUC. The search space covers the parameters
    that matter most for XGBoost on tabular credit-risk data.

    Parameters
    ----------
    X_train, y_train:
        Preprocessed training features and labels.
    X_val, y_val:
        Preprocessed validation features and labels (never used to fit
        preprocessing — no leakage).
    n_trials:
        Number of Optuna trials. 50 gives a good balance of coverage
        vs. runtime on a CPU instance (~2–3 minutes).
    random_state:
        Seed for reproducibility.
    scale_pos_weight:
        Ratio of negative to positive training samples (``n_neg / n_pos``).
        Defaults to the GMSC population constant when not provided.

    Returns
    -------
    dict of best hyperparameters (ready to pass to XGBClassifier(**params)).
    """

    _spw = scale_pos_weight if scale_pos_weight is not None else _SCALE_POS_WEIGHT

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": _spw,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": -1,
        }

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_prob)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "XGBoost tuning complete — best ROC-AUC=%.4f after %d trials",
        study.best_value,
        n_trials,
    )
    logger.info("Best XGBoost params: %s", study.best_params)

    best = study.best_params
    # Add back the fixed params the objective doesn't expose to the search space.
    best.update({
        "scale_pos_weight": _spw,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
    })

    return best


# ---------------------------------------------------------------------------
# LightGBM tuning
# ---------------------------------------------------------------------------

def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    random_state: int = 42,
) -> dict:
    """
    Search for the best LightGBM hyperparameters using Optuna.

    Optimises validation ROC-AUC.

    Parameters
    ----------
    X_train, y_train:
        Preprocessed training features and labels.
    X_val, y_val:
        Preprocessed validation features and labels.
    n_trials:
        Number of Optuna trials.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    dict of best hyperparameters (ready to pass to LGBMClassifier(**params)).
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "is_unbalance": True,
            "objective": "binary",
            "metric": "auc",
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }

        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_prob)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "LightGBM tuning complete — best ROC-AUC=%.4f after %d trials",
        study.best_value,
        n_trials,
    )
    logger.info("Best LightGBM params: %s", study.best_params)

    best = study.best_params
    # Add back fixed params.
    best.update({
        "subsample_freq": 1,
        "is_unbalance": True,
        "objective": "binary",
        "metric": "auc",
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    })

    return best
