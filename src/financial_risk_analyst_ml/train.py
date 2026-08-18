"""
train.py — Main training entry point for the GMSC PD model.

This script is the entry point for SageMaker training jobs and can also
be run locally (with the appropriate ML dependencies available).

SageMaker passes hyperparameters as CLI arguments and sets environment
variables (SM_CHANNEL_TRAIN, SM_MODEL_DIR) that control where data is
read from and where artifacts are written.

Local usage:
    uv run --with pandas --with scikit-learn --with xgboost --with lightgbm \
           --with shap --with optuna \
        python -m financial_risk_analyst_ml.train \
        --data-path /tmp/cs-training.csv \
        --model-dir /tmp/model-output \
        --model all \
        --tune \
        --tune-trials 50

SageMaker usage (handled automatically by the SDK):
    The SDK sets SM_CHANNEL_TRAIN and SM_MODEL_DIR.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from financial_risk_analyst_ml.calibration import (
    fit_calibrator,
    calibrate_probabilities,
)
from financial_risk_analyst_ml.config import CONFIG
from financial_risk_analyst_ml.evaluation import (
    evaluate_model,
    compare_models,
    brier_skill_score,
)
from financial_risk_analyst_ml.explain import (
    shap_summary_dict,
)
from financial_risk_analyst_ml.models import (
    build_logistic_model,
    build_xgboost_model,
    build_lightgbm_model,
    XGBClassifier,
    LGBMClassifier,
)
from financial_risk_analyst_ml.preprocessing import (
    build_preprocessing_pipeline,
    split_features_target,
)
from financial_risk_analyst_ml.tuning import (
    tune_xgboost,
    tune_lightgbm,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _str_to_bool(value: str | bool) -> bool:
    """Parse bool CLI values; SageMaker passes hyperparameters as strings."""
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the GMSC probability-of-default model."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
        help="Path to the CSV file or directory containing training data.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Directory where model artifacts will be saved.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["xgboost", "lightgbm", "logistic", "all"],
        help=(
            "Which model(s) to train. 'all' trains logistic + xgboost + "
            "lightgbm and picks the best by validation ROC-AUC."
        ),
    )
    parser.add_argument(
        "--tune",
        nargs="?",
        const=True,
        default=False,
        type=_str_to_bool,
        help=(
            "Run Optuna hyperparameter tuning for XGBoost and LightGBM "
            "before final training. Adds ~5-10 min on a CPU instance."
        ),
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model (default: 50).",
    )
    parser.add_argument(
        "--calibration-method",
        type=str,
        default="isotonic",
        choices=["isotonic", "platt"],
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=CONFIG.random_state,
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_path: str) -> pd.DataFrame:
    path = Path(data_path)

    if not path.exists():
        logger.info("Local data file '%s' not found. Attempting to download from S3...", path)
        try:
            import boto3
            s3 = boto3.client("s3", region_name=CONFIG.region)
            path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(CONFIG.bucket, CONFIG.gmsc_data_key, str(path))
            logger.info("Downloaded s3://%s/%s -> %s", CONFIG.bucket, CONFIG.gmsc_data_key, path.resolve())
        except Exception as err:
            raise FileNotFoundError(
                f"Data file '{data_path}' not found locally and S3 download failed: {err}"
            ) from err

    if path.is_dir():
        csv_files = list(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in: {data_path}")
        if len(csv_files) > 1:
            logger.warning("Multiple CSVs found; using: %s", csv_files[0])
        path = csv_files[0]

    logger.info("Loading data from: %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def make_splits(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = CONFIG.target_column

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target],
        random_state=random_state,
    )

    val_fraction_of_remaining = val_size / (1.0 - test_size)

    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_remaining,
        stratify=train_val[target],
        random_state=random_state,
    )

    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train), len(val), len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Core training logic
# ---------------------------------------------------------------------------

def train_and_evaluate(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and split
    # ------------------------------------------------------------------
    df = load_data(args.data_path)

    train_df, val_df, test_df = make_splits(
        df,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    X_train_raw, y_train = split_features_target(train_df)
    X_val_raw, y_val = split_features_target(val_df)
    X_test_raw, y_test = split_features_target(test_df)

    # ------------------------------------------------------------------
    # 2. Preprocessing — fit only on training data
    # ------------------------------------------------------------------
    logger.info("Fitting preprocessing pipeline on training data...")
    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(X_train_raw)

    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()
    y_test_arr = y_test.to_numpy()

    # Compute class imbalance ratio from training labels.
    # XGBoost uses this as scale_pos_weight = n_negative / n_positive.
    n_pos = int(y_train_arr.sum())
    n_neg = len(y_train_arr) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)  # guard against zero division
    logger.info(
        "Class distribution — train: %d pos, %d neg (scale_pos_weight=%.2f)",
        n_pos, n_neg, scale_pos_weight,
    )

    # ------------------------------------------------------------------
    # 3. Optional Optuna tuning
    # Tuning uses X_train/X_val only — no test data exposure.
    # ------------------------------------------------------------------
    xgb_params: dict | None = None
    lgb_params: dict | None = None

    if args.tune:
        if args.model in ("xgboost", "all"):
            logger.info(
                "Tuning XGBoost with Optuna (%d trials)...", args.tune_trials
            )
            xgb_params = tune_xgboost(
                X_train, y_train_arr,
                X_val, y_val_arr,
                n_trials=args.tune_trials,
                random_state=args.random_state,
                scale_pos_weight=scale_pos_weight,
            )

        if args.model in ("lightgbm", "all"):
            logger.info(
                "Tuning LightGBM with Optuna (%d trials)...", args.tune_trials
            )
            lgb_params = tune_lightgbm(
                X_train, y_train_arr,
                X_val, y_val_arr,
                n_trials=args.tune_trials,
                random_state=args.random_state,
            )

    # ------------------------------------------------------------------
    # 4. Train models
    # ------------------------------------------------------------------
    models_to_train: list[tuple[str, object]] = []

    if args.model in ("logistic", "all"):
        logger.info("Training Logistic Regression...")
        lr_model = build_logistic_model()
        lr_model.fit(X_train, y_train_arr)
        models_to_train.append(("logistic", lr_model))

    if args.model in ("xgboost", "all"):
        logger.info("Training XGBoost%s...", " (tuned)" if xgb_params else "")
        xgb_model = (
            XGBClassifier(**xgb_params)
            if xgb_params
            else build_xgboost_model(scale_pos_weight=scale_pos_weight)
        )
        xgb_model.fit(
            X_train, y_train_arr,
            eval_set=[(X_val, y_val_arr)],
            verbose=False,
        )
        models_to_train.append(("xgboost", xgb_model))

    if args.model in ("lightgbm", "all"):
        logger.info("Training LightGBM%s...", " (tuned)" if lgb_params else "")
        lgb_model = LGBMClassifier(**lgb_params) if lgb_params else build_lightgbm_model()
        lgb_model.fit(
            X_train, y_train_arr,
            eval_set=[(X_val, y_val_arr)],
        )
        models_to_train.append(("lightgbm", lgb_model))

    # ------------------------------------------------------------------
    # 5. Evaluate on validation set — pick best model by ROC-AUC
    # ------------------------------------------------------------------
    logger.info("Evaluating models on validation set...")
    val_results = []

    for name, model in models_to_train:
        val_probs = model.predict_proba(X_val)[:, 1]
        result = evaluate_model(
            y_val_arr, val_probs,
            model_name=name,
            split="validation",
        )
        logger.info(result.summary())
        val_results.append((name, model, val_probs, result))

    # Print comparison table
    comparison = compare_models([t[3] for t in val_results])
    logger.info("Validation comparison:\n%s", comparison.to_string(index=False))

    best_name, best_model, best_val_probs, best_val_result = max(
        val_results,
        key=lambda t: t[3].roc_auc,
    )
    logger.info(
        "Best model: %s (ROC-AUC=%.4f)", best_name, best_val_result.roc_auc
    )

    # ------------------------------------------------------------------
    # 6. Calibrate on validation set
    # ------------------------------------------------------------------
    logger.info(
        "Calibrating '%s' with method='%s'...",
        best_name,
        args.calibration_method,
    )
    calibrator = fit_calibrator(
        best_val_probs,
        y_val_arr,
        method=args.calibration_method,
    )

    # ------------------------------------------------------------------
    # 7. Final evaluation on held-out test set
    # ------------------------------------------------------------------
    logger.info("Evaluating on held-out test set...")
    test_raw_probs = best_model.predict_proba(X_test)[:, 1]
    test_cal_probs = calibrate_probabilities(calibrator, test_raw_probs)

    test_result_raw = evaluate_model(
        y_test_arr, test_raw_probs,
        model_name=f"{best_name}_raw",
        split="test",
    )
    test_result_cal = evaluate_model(
        y_test_arr, test_cal_probs,
        model_name=f"{best_name}_calibrated",
        split="test",
    )

    logger.info(test_result_raw.summary())
    logger.info(test_result_cal.summary())

    bss = brier_skill_score(y_test_arr, test_cal_probs)
    logger.info("Brier Skill Score (calibrated, test): %.4f", bss)

    # ------------------------------------------------------------------
    # 8. SHAP feature importance (tree models only)
    # ------------------------------------------------------------------
    shap_top: list[dict] = []

    if best_name in ("xgboost", "lightgbm"):
        logger.info("Computing SHAP feature importance on test set...")
        shap_top = shap_summary_dict(best_model, X_test, top_n=10)
        for i, entry in enumerate(shap_top, 1):
            logger.info(
                "  #%d  %-45s  mean|SHAP|=%.4f",
                i, entry["feature"], entry["mean_abs_shap"],
            )

    # ------------------------------------------------------------------
    # 9. Save all artifacts
    # ------------------------------------------------------------------
    logger.info("Saving artifacts to: %s", model_dir)

    joblib.dump(best_model, model_dir / "model.joblib")
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")
    joblib.dump(calibrator, model_dir / "calibrator.joblib")

    metrics = {
        "best_model": best_name,
        "tuning_enabled": args.tune,
        "tune_trials": args.tune_trials if args.tune else 0,
        "calibration_method": args.calibration_method,
        "validation": best_val_result.to_dict(),
        "test_raw": test_result_raw.to_dict(),
        "test_calibrated": test_result_cal.to_dict(),
        "brier_skill_score_test": round(bss, 6),
        "shap_top_features": shap_top,
        "all_validation_results": [t[3].to_dict() for t in val_results],
        "xgb_best_params": xgb_params,
        "lgb_best_params": lgb_params,
    }

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("All artifacts saved.")
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args)
