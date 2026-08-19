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

import sys
import types

# Ensure code directories are in sys.path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Support both package and flat directory unpickling for joblib
try:
    import financial_risk_analyst_ml.preprocessing as preprocessing
    import financial_risk_analyst_ml.features as features
    import financial_risk_analyst_ml.calibration as calibration
except ImportError:
    import preprocessing
    import features
    import calibration
    pkg = types.ModuleType("financial_risk_analyst_ml")
    pkg.preprocessing = preprocessing
    pkg.features = features
    pkg.calibration = calibration
    sys.modules["financial_risk_analyst_ml"] = pkg
    sys.modules["financial_risk_analyst_ml.preprocessing"] = preprocessing
    sys.modules["financial_risk_analyst_ml.features"] = features
    sys.modules["financial_risk_analyst_ml.calibration"] = calibration

try:
    from financial_risk_analyst_ml.calibration import calibrate_probabilities
    from financial_risk_analyst_ml.explain import get_risk_drivers
except ImportError:
    from calibration import calibrate_probabilities
    try:
        from explain import get_risk_drivers
    except ImportError:
        get_risk_drivers = None


logger = logging.getLogger(__name__)

# Version tag embedded in every response so the backend can track
# which model version produced each PD.
MODEL_VERSION = os.environ.get("MODEL_VERSION", "gmsc-xgb-v1")
RISK_THRESHOLD = float(os.environ.get("RISK_THRESHOLD", "0.10"))

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
) -> list[dict] | np.ndarray:
    """
    Run the full inference pipeline and return calibrated PD probabilities
    and SHAP risk drivers for high-risk or requested borrowers.

    Steps:
        1. Preprocessing (cleaning + feature engineering + imputation +
           clipping + scaling), using training-fitted parameters.
        2. Model forward pass.
        3. Probability calibration.
        4. SHAP Risk Driver calculation (for declined/requested borrowers).

    Parameters
    ----------
    input_data:
        DataFrame from input_fn.
    model_artifacts:
        Dict from model_fn.

    Returns
    -------
    List of dicts per borrower (or 1D array of PD values).
    """

    preprocessor = model_artifacts["preprocessor"]
    model = model_artifacts["model"]
    calibrator = model_artifacts["calibrator"]

    # Extract optional 'explain' request flag
    explain_requested = "explain" in input_data.columns and bool(input_data["explain"].iloc[0])

    # Preprocessing (uses training-fitted parameters — no leakage at inference).
    X = preprocessor.transform(input_data)

    # Raw model probabilities.
    raw_probs = model.predict_proba(X)[:, 1]

    # Calibrated PD.
    calibrated_probs = calibrate_probabilities(calibrator, raw_probs)

    # Ensure output probabilities are bounded strictly to [0.0, 1.0]
    calibrated_probs = np.clip(calibrated_probs, 0.0, 1.0)

    results = []
    for i in range(len(calibrated_probs)):
        pd_val = float(calibrated_probs[i])
        is_declined = pd_val >= RISK_THRESHOLD
        
        risk_drivers = []
        if (is_declined or explain_requested) and get_risk_drivers is not None:
            try:
                risk_drivers = get_risk_drivers(model, X[i], top_n=3)
            except Exception as e:
                logger.warning("Could not generate SHAP risk drivers: %s", e)
                risk_drivers = []

        results.append({
            "pd": round(pd_val, 6),
            "status": "DECLINED" if is_declined else "APPROVED",
            "model_version": MODEL_VERSION,
            "risk_drivers": risk_drivers,
        })

    return results


def output_fn(
    prediction: list[dict] | np.ndarray,
    accept: str,
) -> tuple[str, str]:
    """
    Serialise model output to JSON.

    Parameters
    ----------
    prediction:
        List of borrower result dicts (or PD array) from predict_fn.
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

    if isinstance(prediction, np.ndarray):
        results = [
            {
                "pd": round(float(pd_value), 6),
                "status": "DECLINED" if float(pd_value) >= RISK_THRESHOLD else "APPROVED",
                "model_version": MODEL_VERSION,
                "risk_drivers": [],
            }
            for pd_value in prediction
        ]
    else:
        results = prediction

    # Unwrap single-borrower responses for convenience.
    if len(results) == 1:
        body = json.dumps(results[0])
    else:
        body = json.dumps(results)

    return body, "application/json"

