"""
download_data.py — Download training dataset from S3 for local training.

Usage:
    uv run python scripts/download_data.py
    # or specify output path:
    uv run python scripts/download_data.py /tmp/cs-training.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from financial_risk_analyst_ml.config import CONFIG  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def download_data(output_path: str = "cs-training.csv") -> None:
    """Download GMSC dataset from S3 to local filesystem."""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading s3://%s/%s -> %s",
        CONFIG.bucket,
        CONFIG.gmsc_data_key,
        dest.resolve(),
    )
    s3 = boto3.client("s3", region_name=CONFIG.region)
    s3.download_file(CONFIG.bucket, CONFIG.gmsc_data_key, str(dest))
    logger.info("Download complete: %s", dest.resolve())


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "cs-training.csv"
    download_data(out_path)
