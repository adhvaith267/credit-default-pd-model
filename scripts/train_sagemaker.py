"""
train_sagemaker.py — Submit a SageMaker training job for the GMSC PD model.

Uses the SageMaker v2 SDK with the SKLearn estimator (framework 1.2-1,
Python 3.9). Reads data from S3 and writes model artifacts back to S3.

Usage:
    # Train all models with Optuna tuning on spot instances (defaults):
    uv run python scripts/train_sagemaker.py

    # Train only LightGBM, no tuning:
    uv run python scripts/train_sagemaker.py --model lightgbm --no-tune

    # Custom instance type and trial count:
    uv run python scripts/train_sagemaker.py --instance-type ml.m5.xlarge --tune-trials 25

    # Submit and exit without waiting:
    uv run python scripts/train_sagemaker.py --no-wait

Training data:
    s3://financial-risk-analyst-adhvaith-2026/datasets/gmsc/raw/cs-training.csv

Model artifacts:
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
# Configuration (sourced from the shared Config dataclass)
# ---------------------------------------------------------------------------
# Import at module level so the path is available without instantiating the
# full ML stack (avoids heavy imports when running locally).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from financial_risk_analyst_ml.config import CONFIG  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DIR = str(PROJECT_ROOT / "src")
ENTRY_POINT = "financial_risk_analyst_ml/train.py"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_execution_role_arn(role_name: str, region: str) -> str:
    """Resolve an IAM role name to its full ARN."""
    iam = boto3.client("iam", region_name=region)
    return iam.get_role(RoleName=role_name)["Role"]["Arn"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Submit a SageMaker training job for the GMSC PD model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "xgboost", "lightgbm", "logistic"],
        help="Model(s) to train (default: all).",
    )
    parser.add_argument(
        "--tune",
        dest="tune",
        action="store_true",
        default=True,
        help="Enable Optuna hyperparameter tuning (default: enabled).",
    )
    parser.add_argument(
        "--no-tune",
        dest="tune",
        action="store_false",
        help="Disable hyperparameter tuning.",
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
        choices=["platt", "isotonic"],
        help="Probability calibration method (default: isotonic).",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.m5.xlarge",
        help="Training instance type (default: ml.m5.xlarge).",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
        help="Number of training instances (default: 1).",
    )
    parser.add_argument(
        "--spot",
        dest="spot",
        action="store_true",
        default=True,
        help="Use managed spot instances (default: enabled).",
    )
    parser.add_argument(
        "--no-spot",
        dest="spot",
        action="store_false",
        help="Disable spot instances (use on-demand).",
    )
    parser.add_argument(
        "--wait",
        dest="wait",
        action="store_true",
        default=True,
        help="Block until training completes (default: enabled).",
    )
    parser.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="Submit job and return immediately.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Submit a SageMaker training job."""
    args = parse_args()

    role_arn = get_execution_role_arn(CONFIG.role_name, CONFIG.region)
    logger.info("Using IAM role: %s", role_arn)

    sess = sagemaker.Session(boto_session=boto3.Session(region_name=CONFIG.region))

    hyperparameters = {
        "model": args.model,
        "calibration-method": args.calibration_method,
        "val-size": 0.15,
        "test-size": 0.15,
        "random-state": CONFIG.random_state,
        "tune": "true" if args.tune else "false",
        "tune-trials": args.tune_trials,
    }
    logger.info("Hyperparameters: %s", hyperparameters)

    estimator = SKLearn(
        entry_point=ENTRY_POINT,
        source_dir=SOURCE_DIR,
        role=role_arn,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        framework_version="1.2-1",
        py_version="py3",
        hyperparameters=hyperparameters,
        output_path=f"s3://{CONFIG.bucket}/{CONFIG.model_prefix}",
        code_location=f"s3://{CONFIG.bucket}/{CONFIG.model_prefix}/code",
        sagemaker_session=sess,
        base_job_name="gmsc-pd",
        use_spot_instances=args.spot,
        max_run=3600,
        max_wait=7200 if args.spot else None,
        volume_size=10,
        disable_profiler=True,
        debugger_hook_config=False,
        dependencies=[str(PROJECT_ROOT / "sagemaker" / "requirements.txt")],
    )

    logger.info(
        "Submitting training job (instance=%s, spot=%s, wait=%s)...",
        args.instance_type,
        args.spot,
        args.wait,
    )

    estimator.fit(
        inputs={"train": f"s3://{CONFIG.bucket}/{CONFIG.gmsc_data_key}"},
        wait=args.wait,
        logs="All" if args.wait else False,
    )

    if args.wait:
        logger.info("Training complete!")
        logger.info("Model artifact: %s", estimator.model_data)
    else:
        logger.info("Training job submitted: %s", estimator.latest_training_job.name)
        logger.info(
            "Check status: aws sagemaker describe-training-job "
            "--training-job-name %s --region %s",
            estimator.latest_training_job.name,
            CONFIG.region,
        )


if __name__ == "__main__":
    main()
