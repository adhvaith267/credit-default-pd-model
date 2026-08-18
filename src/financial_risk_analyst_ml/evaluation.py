from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


@dataclass
class EvaluationResult:
    """
    Container for all evaluation metrics of a PD model.

    Attributes
    ----------
    split:
        Which data split this was evaluated on, e.g. 'validation', 'test'.
    model_name:
        Name of the model, e.g. 'logistic', 'xgboost'.
    roc_auc:
        Area under the ROC curve. Measures discrimination / ranking.
    pr_auc:
        Area under the Precision-Recall curve.
        More informative than ROC-AUC for highly imbalanced data.
    brier_score:
        Mean squared error between predicted probabilities and true labels.
        Lower is better; a naive model predicting the base rate scores
        the fraction of positives squared + fraction of negatives squared.
    n_positive:
        Number of positive (default) examples.
    n_negative:
        Number of negative (non-default) examples.
    calibration_bins:
        Dict with 'mean_predicted_prob' and 'fraction_of_positives' arrays
        for plotting calibration curves.
    """

    split: str
    model_name: str
    roc_auc: float
    pr_auc: float
    brier_score: float
    n_positive: int
    n_negative: int
    calibration_bins: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of all scalar metrics."""
        return {
            "split": self.split,
            "model_name": self.model_name,
            "roc_auc": round(self.roc_auc, 6),
            "pr_auc": round(self.pr_auc, 6),
            "brier_score": round(self.brier_score, 6),
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }

    def summary(self) -> str:
        """Human-readable one-paragraph summary."""
        return (
            f"[{self.model_name} | {self.split}] "
            f"ROC-AUC={self.roc_auc:.4f}  "
            f"PR-AUC={self.pr_auc:.4f}  "
            f"Brier={self.brier_score:.4f}  "
            f"(pos={self.n_positive}, neg={self.n_negative})"
        )


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "model",
    split: str = "validation",
    n_calibration_bins: int = 10,
) -> EvaluationResult:
    """
    Evaluate a binary PD model using discrimination and calibration metrics.

    Parameters
    ----------
    y_true:
        True binary labels (0 = no default, 1 = default).
    y_prob:
        Predicted probabilities for the positive class.
    model_name:
        Label used in output (e.g. 'logistic', 'xgboost').
    split:
        Data split label (e.g. 'validation', 'test').
    n_calibration_bins:
        Number of bins for the calibration curve.

    Returns
    -------
    EvaluationResult
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # Calibration curve: fraction of positives in each probability bin.
    fraction_of_positives, mean_predicted_prob = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_calibration_bins,
        strategy="uniform",
    )

    calibration_bins = {
        "mean_predicted_prob": mean_predicted_prob.tolist(),
        "fraction_of_positives": fraction_of_positives.tolist(),
    }

    n_positive = int(y_true.sum())
    n_negative = int(len(y_true) - n_positive)

    return EvaluationResult(
        split=split,
        model_name=model_name,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=brier,
        n_positive=n_positive,
        n_negative=n_negative,
        calibration_bins=calibration_bins,
    )


def compare_models(
    results: list[EvaluationResult],
) -> pd.DataFrame:
    """
    Build a comparison table from a list of EvaluationResult objects.

    Useful for deciding which model to deploy.
    """

    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)

    # Sort by ROC-AUC descending so the best model appears first.
    df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)

    return df


def brier_skill_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Brier Skill Score (BSS) relative to a naive climatological forecast.

    BSS = 1 - (Brier_model / Brier_naive)

    Where Brier_naive predicts the base rate for every observation.
    BSS = 1 is perfect, BSS = 0 is no skill, BSS < 0 is worse than naive.
    """

    y_true = np.asarray(y_true)
    base_rate = y_true.mean()

    brier_model = brier_score_loss(y_true, y_prob)
    brier_naive = brier_score_loss(
        y_true,
        np.full_like(y_true, fill_value=base_rate, dtype=float),
    )

    if brier_naive == 0:
        return 0.0

    return float(1.0 - brier_model / brier_naive)


def threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute precision, recall, specificity, and F1 at a given threshold.

    In credit risk you will typically choose the threshold based on
    business tolerance for false negatives (missed defaults) vs
    false positives (rejected good borrowers).
    """

    y_true = np.asarray(y_true)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "f1": round(f1, 6),
    }
