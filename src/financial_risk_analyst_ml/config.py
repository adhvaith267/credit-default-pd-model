"""
config.py — Central configuration for the GMSC PD model project.

All AWS resource names, S3 paths, and training constants live here so
that scripts never hard-code them independently.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # AWS
    bucket: str = "financial-risk-analyst-adhvaith-2026"
    region: str = "ap-south-1"
    role_name: str = "FinancialRiskSageMakerExecutionRole"

    # S3 paths
    gmsc_data_key: str = "datasets/gmsc/raw/cs-training.csv"
    model_prefix: str = "models/gmsc"

    # SageMaker endpoint
    endpoint_name: str = "gmsc-pd-endpoint"

    # Training
    random_state: int = 42
    target_column: str = "SeriousDlqin2yrs"


CONFIG = Config()
