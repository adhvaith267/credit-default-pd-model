"""
inference.py — SageMaker serving script for the GMSC PD model.

SageMaker calls these four functions in sequence for every request:

    model_fn(model_dir)
        Load and return all artifacts from model_dir.

    input_fn(request_body, content_type)
        Deserialise the raw HTTP request body into a pandas DataFrame.

    predict_fn(input_data, model)
        Run preprocessing → model → calibration → return raw PD array.

    output_fn(prediction, accept)
        Serialise the PD array to the requested response format.

Supported content types:
    Request:   application/json
    Response:  application/json

Expected request format
-----------------------
Single borrower:
    {
        "RevolvingUtilizationOfUnsecuredLines": 0.766,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.80,
        "MonthlyIncome": 9120,
        "NumberOfOpenCreditLinesAndLoans": 13,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 6,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2
    }

Batch borrowers:
    [
        { ... borrower 1 ... },
        { ... borrower 2 ... }
    ]

Response format
---------------
Single:
    {"pd": 0.083, "model_version": "gmsc-xgb-v1"}

Batch:
    [
        {"pd": 0.083, "model_version": "gmsc-xgb-v1"},
        {"pd": 0.021, "model_version": "gmsc-xgb-v1"}
    ]
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from financial_risk_analyst_ml.calibration import calibrate_probabilities
except ImportError:
    from calibration import calibrate_probabilities


logger = logging.getLogger(__name__)

# Version tag embedded in every response so the backend can track
# which model version produced each PD.
MODEL_VERSION = os.environ.get("MODEL_VERSION", "gmsc-xgb-v1")

# Expected raw input columns (before feature engineering).
RAW_INPUT_COLUMNS = [
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
]


# ---------------------------------------------------------------------------
# SageMaker inference handlers
# ---------------------------------------------------------------------------

def model_fn(model_dir: str) -> dict:
    """
    Load model artifacts from the SageMaker model directory.

    SageMaker calls this once at container startup.

    Parameters
    ----------
    model_dir:
        Path to the directory containing model artifacts saved by train.py.

    Returns
    -------
    dict with keys:
        'model'        — fitted sklearn-compatible classifier
        'preprocessor' — fitted GMSCPreprocessingPipeline
        'calibrator'   — fitted PlattScaler or IsotonicCalibrator
    """

    model_dir = Path(model_dir)

    logger.info("Loading artifacts from: %s", model_dir)

    model = joblib.load(model_dir / "model.joblib")
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    calibrator = joblib.load(model_dir / "calibrator.joblib")

    logger.info("Artifacts loaded successfully.")

    return {
        "model": model,
        "preprocessor": preprocessor,
        "calibrator": calibrator,
    }


def input_fn(request_body: str, content_type: str) -> pd.DataFrame:
    """
    Deserialise the request body into a pandas DataFrame.

    Accepts either a single borrower dict or a list of borrower dicts.

    Parameters
    ----------
    request_body:
        Raw HTTP request body as a string.
    content_type:
        MIME type of the request (must be 'application/json').

    Returns
    -------
    pd.DataFrame with one row per borrower.
    """

    if content_type != "application/json":
        raise ValueError(
            f"Unsupported content type: '{content_type}'. "
            "Expected 'application/json'."
        )

    data = json.loads(request_body)

    # Normalise to a list so the rest of the code handles both cases.
    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)

    # Validate that all expected columns are present.
    missing = [col for col in RAW_INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Request is missing required columns: {missing}"
        )

    return df


def predict_fn(
    input_data: pd.DataFrame,
    model_artifacts: dict,
) -> np.ndarray:
    """
    Run the full inference pipeline and return calibrated PD probabilities.

    Steps:
        1. Preprocessing (cleaning + feature engineering + imputation +
           clipping + scaling), using training-fitted parameters.
        2. Model forward pass.
        3. Probability calibration.

    Parameters
    ----------
    input_data:
        DataFrame from input_fn.
    model_artifacts:
        Dict from model_fn.

    Returns
    -------
    1-D numpy array of calibrated PD values (one per borrower).
    """

    preprocessor = model_artifacts["preprocessor"]
    model = model_artifacts["model"]
    calibrator = model_artifacts["calibrator"]

    # Preprocessing (uses training-fitted parameters — no leakage at inference).
    X = preprocessor.transform(input_data)

    # Raw model probabilities.
    raw_probs = model.predict_proba(X)[:, 1]

    # Calibrated PD.
    calibrated_probs = calibrate_probabilities(calibrator, raw_probs)

    return calibrated_probs


def output_fn(
    prediction: np.ndarray,
    accept: str,
) -> tuple[str, str]:
    """
    Serialise model output to JSON.

    Parameters
    ----------
    prediction:
        Calibrated PD array from predict_fn.
    accept:
        Requested response MIME type (should be 'application/json').

    Returns
    -------
    Tuple of (response_body: str, content_type: str).
    """

    if accept not in ("application/json", "*/*"):
        raise ValueError(
            f"Unsupported accept type: '{accept}'. "
            "Expected 'application/json'."
        )

    results = [
        {
            "pd": round(float(pd_value), 6),
            "model_version": MODEL_VERSION,
        }
        for pd_value in prediction
    ]

    # Unwrap single-borrower responses for convenience.
    if len(results) == 1:
        body = json.dumps(results[0])
    else:
        body = json.dumps(results)

    return body, "application/json"
