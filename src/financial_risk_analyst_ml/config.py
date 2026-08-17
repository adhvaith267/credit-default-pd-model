from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    bucket: str = "financial-risk-analyst-adhvaith-2026"

    gmsc_data_key: str = (
        "datasets/gmsc/raw/cs-training.csv"
    )

    model_prefix: str = "models/gmsc"

    random_state: int = 42

    target_column: str = "SeriousDlqin2yrs"


CONFIG = Config()
