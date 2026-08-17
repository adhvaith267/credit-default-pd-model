"""
train_sagemaker.py — Submit a SageMaker training job for the GMSC PD model.

Usage:
    uv run python scripts/train_sagemaker.py [--instance-type ml.m5.large]

What this does:
    1. Resolves the S3 dataset path and IAM role from config.
    2. Submits a SageMaker SKLearn estimator that runs train.py inside
       the managed SageMaker Python 3 runtime.
    3. Waits for the job to complete and prints the model artifact S3 URI.

The training job reads data from:
    s3://financial-risk-analyst-adhvaith-2026/datasets/gmsc/raw/cs-training.csv

And writes model artifacts to:
    s3://financial-risk-analyst-adhvaith-2026/models/gmsc/<job-name>/output/model.tar.gz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BUCKET = "financial-risk-analyst-adhvaith-2026"
DATA_KEY = "datasets/gmsc/raw/cs-training.csv"
MODEL_PREFIX = "models/gmsc"
ROLE_NAME = "FinancialRiskSageMakerExecutionRole"

# The source directory passed to SageMaker — contains train.py (as a module).
SOURCE_DIR = str(Path(__file__).parent.parent / "src")
ENTRY_POINT = "financial_risk_analyst_ml/train.py"

# Python version must match the SageMaker framework version.
PYTHON_VERSION = "py3"
SKLEARN_VERSION = "1.2-1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a SageMaker training job for the GMSC PD model."
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.m5.large",
        help="SageMaker training instance type (default: ml.m5.large).",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
        help="Number of training instances (default: 1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "logistic", "both"],
        help="Which model to train (default: xgboost).",
    )
    parser.add_argument(
        "--calibration-method",
        type=str,
        default="isotonic",
        choices=["isotonic", "platt"],
    )
    parser.add_argument(
        "--region",
        type=str,
        default="ap-south-1",
        help="AWS region (default: ap-south-1).",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        default=True,
        help="Wait for the training job to complete (default: True).",
    )
    parser.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="Submit job and exit immediately without waiting.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve the IAM role ARN.
    iam = boto3.client("iam", region_name=args.region)
    role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
    logger.info("Using IAM role: %s", role_arn)

    # SageMaker session.
    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    # S3 input channel.
    train_input = sagemaker.inputs.TrainingInput(
        s3_data=f"s3://{BUCKET}/{DATA_KEY}",
        content_type="text/csv",
    )

    # Hyperparameters passed to train.py as CLI args.
    hyperparameters = {
        "model": args.model,
        "calibration-method": args.calibration_method,
        "val-size": 0.15,
        "test-size": 0.15,
        "random-state": 42,
    }

    # SageMaker SKLearn estimator.
    # We use SKLearn as the managed container because it includes
    # scikit-learn, numpy, pandas, and joblib out of the box.
    # XGBoost and SHAP are installed via requirements.txt in sagemaker/.
    estimator = SKLearn(
        entry_point=ENTRY_POINT,
        source_dir=SOURCE_DIR,
        role=role_arn,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        framework_version=SKLEARN_VERSION,
        py_version=PYTHON_VERSION,
        hyperparameters=hyperparameters,
        output_path=f"s3://{BUCKET}/{MODEL_PREFIX}",
        base_job_name="gmsc-pd-model",
        sagemaker_session=sm_session,
        dependencies=[
            str(Path(__file__).parent.parent / "sagemaker" / "requirements.txt")
        ],
    )

    logger.info("Submitting training job...")
    estimator.fit(
        inputs={"train": train_input},
        wait=args.wait,
        logs="All" if args.wait else None,
    )

    if args.wait:
        model_uri = estimator.model_data
        logger.info("Training complete.")
        logger.info("Model artifact: %s", model_uri)
        print(f"\nModel artifact S3 URI:\n  {model_uri}")


if __name__ == "__main__":
    main()
