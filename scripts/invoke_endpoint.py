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
    {"pd": 0.083, "status": "APPROVED", "model_version": "gmsc-xgb-v1", "risk_drivers": []}

Output example (declined borrower):
    {
      "pd": 0.354,
      "status": "DECLINED",
      "model_version": "gmsc-xgb-v1",
      "risk_drivers": [
        "High credit card & revolving line utilization",
        "Past-due delinquency events (30-59 days late)",
        "High debt-to-income ratio"
      ]
    }
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

# Built-in test suite testing both APPROVED (Low Risk) and DECLINED (High Risk) scenarios.
LOW_RISK_BORROWER = {
    "RevolvingUtilizationOfUnsecuredLines": 0.05,
    "age": 52,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.25,
    "MonthlyIncome": 12500.0,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1.0,
}

HIGH_RISK_BORROWER = {
    "RevolvingUtilizationOfUnsecuredLines": 0.95,
    "age": 28,
    "NumberOfTime30-59DaysPastDueNotWorse": 3,
    "DebtRatio": 0.85,
    "MonthlyIncome": 2100.0,
    "NumberOfOpenCreditLinesAndLoans": 14,
    "NumberOfTimes90DaysLate": 2,
    "NumberRealEstateLoansOrLines": 4,
    "NumberOfTime60-89DaysPastDueNotWorse": 1,
    "NumberOfDependents": 3.0,
}

EXAMPLE_BORROWERS = [LOW_RISK_BORROWER, HIGH_RISK_BORROWER]


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
        payload = EXAMPLE_BORROWERS
        logger.info("Using built-in test suite (Low Risk & High Risk borrowers).")

    result = invoke(
        endpoint_name=args.endpoint_name,
        payload=payload,
        region=args.region,
    )

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))

    results_list = [result] if isinstance(result, dict) else result
    for idx, item in enumerate(results_list, start=1):
        if isinstance(item, dict) and "pd" in item:
            pd_val = item["pd"]
            status = item.get("status", "UNKNOWN")
            drivers = item.get("risk_drivers", [])
            logger.info("Borrower #%d: PD = %.4f (%.2f%%) | Status = %s", idx, pd_val, pd_val * 100, status)
            if drivers:
                logger.info("  Primary Risk Drivers: %s", drivers)


if __name__ == "__main__":
    main()

