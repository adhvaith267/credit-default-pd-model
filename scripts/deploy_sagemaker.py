"""
deploy_sagemaker.py — Deploy a trained GMSC PD model to a SageMaker endpoint.

Usage:
    uv run python scripts/deploy_sagemaker.py \
        --model-artifact s3://financial-risk-analyst-adhvaith-2026/models/gmsc/.../model.tar.gz

What this does:
    1. Creates a SageMaker Model from the artifact and inference.py script.
    2. Creates an endpoint configuration.
    3. Creates or updates the endpoint.
    4. Waits for the endpoint to be InService.

The endpoint name defaults to 'gmsc-pd-endpoint'.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROLE_NAME = "FinancialRiskSageMakerExecutionRole"
ENDPOINT_NAME = "gmsc-pd-endpoint"
SOURCE_DIR = str(Path(__file__).parent.parent / "src")
ENTRY_POINT = "financial_risk_analyst_ml/inference.py"
SKLEARN_VERSION = "1.2-1"
PYTHON_VERSION = "py3"

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
        description="Deploy a trained GMSC PD model to a SageMaker endpoint."
    )
    parser.add_argument(
        "--model-artifact",
        type=str,
        required=True,
        help=(
            "S3 URI of the model artifact (model.tar.gz) produced by "
            "train_sagemaker.py, e.g. "
            "s3://financial-risk-analyst-adhvaith-2026/models/gmsc/.../output/model.tar.gz"
        ),
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default=ENDPOINT_NAME,
        help=f"SageMaker endpoint name (default: {ENDPOINT_NAME}).",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.t2.medium",
        help="Inference instance type (default: ml.t2.medium).",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
        help="Number of inference instances (default: 1).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="ap-south-1",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="gmsc-xgb-v1",
        help="Model version tag embedded in predictions.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve IAM role ARN.
    iam = boto3.client("iam", region_name=args.region)
    role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
    logger.info("Using IAM role: %s", role_arn)

    # SageMaker session.
    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    logger.info("Creating SageMaker model from artifact: %s", args.model_artifact)

    # Create the model object.
    # SageMaker will serve requests using inference.py's four handler functions.
    model = SKLearnModel(
        model_data=args.model_artifact,
        role=role_arn,
        entry_point=ENTRY_POINT,
        source_dir=SOURCE_DIR,
        framework_version=SKLEARN_VERSION,
        py_version=PYTHON_VERSION,
        sagemaker_session=sm_session,
        env={
            "MODEL_VERSION": args.model_version,
        },
        dependencies=[
            str(Path(__file__).parent.parent / "sagemaker" / "requirements.txt")
        ],
    )

    logger.info(
        "Deploying to endpoint '%s' on %s × %d...",
        args.endpoint_name,
        args.instance_type,
        args.instance_count,
    )

    predictor = model.deploy(
        initial_instance_count=args.instance_count,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
        wait=True,
    )

    logger.info("Endpoint is InService: %s", args.endpoint_name)
    print(f"\nEndpoint deployed successfully:")
    print(f"  Name:   {args.endpoint_name}")
    print(f"  Region: {args.region}")
    print(f"  URL:    https://runtime.sagemaker.{args.region}.amazonaws.com/endpoints/{args.endpoint_name}/invocations")


if __name__ == "__main__":
    main()
