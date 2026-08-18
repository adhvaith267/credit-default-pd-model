"""
deploy_sagemaker.py — Production-grade idempotent SageMaker deployment script.

Uses boto3 and SageMaker SDK to handle all deployment lifecycles idempotently:
  - Timestamped EndpointConfig names (prevents EndpointConfig collision errors)
  - In-place zero-downtime updates for active (InService) endpoints
  - Automatic status waiting for Creating / Updating endpoints
  - Automatic cleanup of Failed or OutOfService endpoints

Usage:
    uv run python scripts/deploy_sagemaker.py
    uv run python scripts/deploy_sagemaker.py --endpoint-name custom-pd-endpoint
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from financial_risk_analyst_ml.config import CONFIG  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DIR = str(PROJECT_ROOT / "src")
ENTRY_POINT = "financial_risk_analyst_ml/inference.py"

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
    """Return the S3 URI of the most recently uploaded model.tar.gz."""
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


def deploy_or_update_endpoint(
    sm_client,
    model_name: str,
    endpoint_name: str,
    instance_type: str,
    instance_count: int,
) -> None:
    """
    Idempotently deploy or update a SageMaker endpoint.

    Handles all endpoint states:
      - Non-existent: Creates endpoint.
      - InService: Updates endpoint in-place using a unique timestamped EndpointConfig.
      - Creating / Updating: Waits for completion, then updates endpoint in-place.
      - Failed / OutOfService: Deletes failed endpoint and creates a fresh endpoint.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = f"{endpoint_name}-cfg-{timestamp}"

    # 1. Create unique EndpointConfig for this deployment
    logger.info("Creating unique EndpointConfig '%s'...", config_name)
    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": instance_count,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    # 2. Inspect existing endpoint status
    endpoint_status = None
    try:
        res = sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_status = res["EndpointStatus"]
        logger.info("Found existing endpoint '%s' with status: %s", endpoint_name, endpoint_status)
    except sm_client.exceptions.ClientError as err:
        if "Could not find endpoint" in str(err):
            logger.info("Endpoint '%s' does not exist yet. Creating new endpoint...", endpoint_name)
        else:
            raise

    # 3. Handle Creating / Updating states by waiting for InService
    if endpoint_status in ("Creating", "Updating"):
        logger.info("Endpoint '%s' is currently %s. Waiting for InService status...", endpoint_name, endpoint_status)
        waiter = sm_client.get_waiter("endpoint_in_service")
        try:
            waiter.wait(EndpointName=endpoint_name)
            endpoint_status = "InService"
            logger.info("Endpoint '%s' reached InService state.", endpoint_name)
        except Exception as wait_err:
            logger.warning("Endpoint did not reach InService: %s", wait_err)
            res = sm_client.describe_endpoint(EndpointName=endpoint_name)
            endpoint_status = res["EndpointStatus"]

    # 4. Handle Failed / OutOfService states by deleting
    if endpoint_status in ("Failed", "OutOfService"):
        logger.info("Deleting failed/stopped endpoint '%s'...", endpoint_name)
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        waiter = sm_client.get_waiter("endpoint_deleted")
        waiter.wait(EndpointName=endpoint_name)
        logger.info("Endpoint '%s' deleted successfully.", endpoint_name)
        endpoint_status = None

    # 5. Perform Endpoint Creation or Update
    if endpoint_status == "InService":
        logger.info("Updating live endpoint '%s' with new config '%s'...", endpoint_name, config_name)
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
    else:
        logger.info("Creating new endpoint '%s' with config '%s'...", endpoint_name, config_name)
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )

    # 6. Wait for Endpoint to reach InService
    logger.info("Waiting for endpoint '%s' to reach InService state (this takes 3-5 mins)...", endpoint_name)
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)
    logger.info("Endpoint '%s' is now IN_SERVICE!", endpoint_name)


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
    sm_client = boto3.client("sagemaker", region_name=CONFIG.region)

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

    # Prepare and create SageMaker Model resource in AWS
    logger.info("Creating SageMaker model resource in AWS...")
    model._create_sagemaker_model(instance_type=args.instance_type)
    logger.info("SageMaker model created with name: %s", model.name)

    # Idempotent Deployment / Update
    deploy_or_update_endpoint(
        sm_client=sm_client,
        model_name=model.name,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
    )

    logger.info("Deployment complete!")
    logger.info("Endpoint name: %s", args.endpoint_name)
    logger.info("Test with: uv run python scripts/invoke_endpoint.py")


if __name__ == "__main__":
    main()
