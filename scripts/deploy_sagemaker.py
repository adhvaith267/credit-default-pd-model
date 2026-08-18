"""
deploy_sagemaker.py — Deploy the trained GMSC PD model to a SageMaker endpoint.

Uses the SageMaker v2 SDK with the SKLearnModel class.

Usage:
    # Deploy from the latest training job artifact:
    uv run python scripts/deploy_sagemaker.py

    # Deploy a specific model artifact:
    uv run python scripts/deploy_sagemaker.py \\
        --model-artifact s3://financial-risk-analyst-adhvaith-2026/models/gmsc/<job>/output/model.tar.gz

    # Custom instance type and endpoint name:
    uv run python scripts/deploy_sagemaker.py \\
        --instance-type ml.m5.xlarge \\
        --endpoint-name my-custom-endpoint

Endpoint created/updated:
    gmsc-pd-endpoint  (default)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from financial_risk_analyst_ml.config import CONFIG  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DIR = str(PROJECT_ROOT / "src" / "financial_risk_analyst_ml")
ENTRY_POINT = "inference.py"

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


def get_latest_model_artifact(bucket: str, prefix: str, region: str) -> str:
    """
    Return the S3 URI of the most recently uploaded model.tar.gz under prefix.

    Raises RuntimeError if no artifact is found.
    """
    s3 = boto3.client("s3", region_name=region)
    paginator = s3.get_paginator("list_objects_v2")

    all_objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        all_objects.extend(page.get("Contents", []))

    models = [obj for obj in all_objects if obj["Key"].endswith("model.tar.gz")]

    if not models:
        raise RuntimeError(
            f"No model.tar.gz found at s3://{bucket}/{prefix}. "
            "Run train_sagemaker.py first."
        )

    latest = max(models, key=lambda x: x["LastModified"])
    uri = f"s3://{bucket}/{latest['Key']}"
    logger.info("Latest model artifact: %s (modified %s)", uri, latest["LastModified"])
    return uri


def cleanup_existing_endpoint(endpoint_name: str, region: str) -> bool:
    """
    Check if an endpoint already exists. If it exists in a FAILED or OUT_OF_SERVICE state,
    delete it. Also delete any stale EndpointConfig with the same name if update_endpoint is False.
    Returns True if the endpoint exists in IN_SERVICE state (so update_endpoint can be set).
    """
    sm_client = boto3.client("sagemaker", region_name=region)
    is_in_service = False
    try:
        res = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = res["EndpointStatus"]
        logger.info("Found existing endpoint '%s' with status: %s", endpoint_name, status)
        if status in ("Failed", "OutOfService"):
            logger.info("Deleting failed/stopped endpoint '%s'...", endpoint_name)
            sm_client.delete_endpoint(EndpointName=endpoint_name)
            waiter = sm_client.get_waiter("endpoint_deleted")
            waiter.wait(EndpointName=endpoint_name)
            logger.info("Endpoint '%s' deleted successfully.", endpoint_name)
        elif status == "InService":
            is_in_service = True
    except sm_client.exceptions.ClientError as err:
        if "Could not find endpoint" not in str(err):
            logger.warning("Error checking existing endpoint status: %s", err)

    if not is_in_service:
        try:
            sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
            logger.info("Deleted stale endpoint configuration '%s'.", endpoint_name)
        except sm_client.exceptions.ClientError:
            pass

    return is_in_service


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Deploy the GMSC PD model to a SageMaker endpoint."
    )
    parser.add_argument(
        "--model-artifact",
        type=str,
        default=None,
        help="S3 URI of model.tar.gz (default: latest from training jobs).",
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default=CONFIG.endpoint_name,
        help=f"Endpoint name (default: {CONFIG.endpoint_name}).",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.m5.xlarge",
        help="Inference instance type (default: ml.m5.xlarge).",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
        help="Number of inference instances (default: 1).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Deploy the model to a SageMaker real-time endpoint."""
    args = parse_args()

    role_arn = get_execution_role_arn(CONFIG.role_name, CONFIG.region)
    logger.info("Using IAM role: %s", role_arn)

    sess = sagemaker.Session(boto_session=boto3.Session(region_name=CONFIG.region))

    model_data = (
        args.model_artifact
        or get_latest_model_artifact(CONFIG.bucket, CONFIG.model_prefix, CONFIG.region)
    )
    logger.info("Deploying model artifact: %s", model_data)

    model = SKLearnModel(
        model_data=model_data,
        role=role_arn,
        entry_point=ENTRY_POINT,
        source_dir=SOURCE_DIR,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sess,
        dependencies=[str(PROJECT_ROOT / "sagemaker" / "requirements.txt")],
    )

    update_endpoint = cleanup_existing_endpoint(args.endpoint_name, CONFIG.region)

    logger.info(
        "Deploying to endpoint '%s' (instance=%s, count=%d, update=%s)...",
        args.endpoint_name,
        args.instance_type,
        args.instance_count,
        update_endpoint,
    )

    predictor = model.deploy(
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        initial_instance_count=args.instance_count,
        update_endpoint=update_endpoint,
        wait=True,
    )

    logger.info("Deployment complete!")
    logger.info("Endpoint name: %s", predictor.endpoint_name)
    logger.info("Test with: uv run python scripts/invoke_endpoint.py")



if __name__ == "__main__":
    main()
