"""
train.py — Top-level SageMaker entry point that re-exports the training main.

SageMaker's SKLearn container imports the entry_point as a flat module name,
so this file must live at the root of source_dir. It delegates all logic to
the real training module inside the package.
"""

from financial_risk_analyst_ml.train import main  # noqa: F401

if __name__ == "__main__":
    main()
