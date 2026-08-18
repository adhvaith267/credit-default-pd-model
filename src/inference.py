"""
inference.py — Top-level SageMaker entry point that re-exports serving handlers.

SageMaker's SKLearn container imports the entry_point as a flat module name,
so this file must live at the root of source_dir. It delegates all logic to
the real inference module inside the package.
"""

from financial_risk_analyst_ml.inference import (  # noqa: F401
    model_fn,
    input_fn,
    predict_fn,
    output_fn,
)
