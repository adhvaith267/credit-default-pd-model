"""
invoke_endpoint.py — Send test requests to the deployed SageMaker endpoint.

Usage:
    # Single borrower (uses built-in example):
    uv run python scripts/invoke_endpoint.py

    # From a JSON file:
    uv run python scripts/invoke_endpoint.py --input-file /path/to/borrower.json

    # Override endpoint name or region:
    uv run python scripts/invoke_endpoint.py \\
        --endpoint-name gmsc-pd-endpoint \\
        --region ap-south-1

    # Pretty-print the JSON response:
    uv run python scripts/invoke_endpoint.py --pretty

Output example (single borrower):
    {"pd": 0.083, "model_version": "gmsc-xgb-v1"}

Output example (batch):
    [{"pd": 0.083, "model_version": "gmsc-xgb-v1"}, ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from financial_risk_analyst_ml.config import CONFIG  # noqa: E402

# Example borrower for smoke-testing the endpoint without an input file.
EXAMPLE_BORROWER = {
    "RevolvingUtilizationOfUnsecuredLines": 0.766,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 2,
    "DebtRatio": 0.80,
    "MonthlyIncome": 9120.0,
    "NumberOfOpenCreditLinesAndLoans": 13,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 6,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2.0,
}

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
        description="Invoke the deployed GMSC PD endpoint."
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default=CONFIG.endpoint_name,
        help=f"SageMaker endpoint name (default: {CONFIG.endpoint_name}).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=CONFIG.region,
        help=f"AWS region (default: {CONFIG.region}).",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help=(
            "Path to a JSON file containing a borrower dict or list of "
            "borrower dicts. If omitted, uses the built-in example."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=False,
        help="Pretty-print the response JSON.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------

def invoke(
    endpoint_name: str,
    payload: dict | list,
    region: str,
) -> dict | list:
    """
    Invoke the SageMaker endpoint with a borrower payload.

    Parameters
    ----------
    endpoint_name:
        Name of the deployed endpoint.
    payload:
        A single borrower dict or a list of borrower dicts.
    region:
        AWS region where the endpoint is deployed.

    Returns
    -------
    Parsed JSON response from the endpoint.
    """
    client = boto3.client("sagemaker-runtime", region_name=region)

    body = json.dumps(payload)
    logger.info("Invoking endpoint '%s' in region '%s'...", endpoint_name, region)
    logger.debug("Request payload: %s", body)

    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=body.encode("utf-8"),
    )

    response_body = response["Body"].read().decode("utf-8")
    logger.debug("Raw response: %s", response_body)

    return json.loads(response_body)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.input_file:
        path = Path(args.input_file)
        if not path.exists():
            logger.error("Input file not found: %s", args.input_file)
            sys.exit(1)
        with open(path) as fh:
            payload = json.load(fh)
        logger.info("Loaded payload from: %s", args.input_file)
    else:
        payload = EXAMPLE_BORROWER
        logger.info("Using built-in example borrower.")

    result = invoke(
        endpoint_name=args.endpoint_name,
        payload=payload,
        region=args.region,
    )

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))

    if isinstance(result, dict) and "pd" in result:
        pd_value = result["pd"]
        logger.info("PD = %.4f  (%.2f%%)", pd_value, pd_value * 100)


if __name__ == "__main__":
    main()
