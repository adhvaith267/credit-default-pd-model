from __future__ import annotations

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattScaler(BaseEstimator):
    """
    Platt scaling: fits a logistic regression on top of raw model scores.

    This is the "sigmoid" method from sklearn's CalibratedClassifierCV.
    Typically works well when the uncalibrated scores are already
    monotonically related to probability.

    Usage
    -----
    scaler = PlattScaler()
    scaler.fit(raw_scores_val, y_val)
    calibrated_probs = scaler.predict_proba(raw_scores_test)
    """

    def __init__(self):
        self._lr = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
        )

    def fit(
        self,
        scores: np.ndarray,
        y: np.ndarray,
    ) -> PlattScaler:
        """
        Fit a logistic regression to map raw scores → calibrated probability.

        Parameters
        ----------
        scores:
            Raw model output scores or probabilities (1-D array).
        y:
            True binary labels.
        """

        scores = np.asarray(scores).reshape(-1, 1)
        y = np.asarray(y)
        self._lr.fit(scores, y)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """
        Return calibrated probabilities.

        Parameters
        ----------
        scores:
            Raw scores to calibrate (1-D array).

        Returns
        -------
        1-D array of calibrated probabilities for the positive class.
        """

        scores = np.asarray(scores).reshape(-1, 1)
        return self._lr.predict_proba(scores)[:, 1]

    def is_fitted(self) -> bool:
        """Return True if the scaler has been fitted."""
        return hasattr(self._lr, "coef_")


class IsotonicCalibrator(BaseEstimator):
    """
    Isotonic regression calibration.

    More flexible than Platt scaling but requires more data.
    Works well when you have at least ~1,000 validation examples,
    which GMSC with ~22,500 validation rows satisfies easily.

    Monotonicity constraint is appropriate for credit risk:
    higher raw score should always map to higher PD.

    Usage
    -----
    calibrator = IsotonicCalibrator()
    calibrator.fit(raw_scores_val, y_val)
    calibrated_probs = calibrator.predict_proba(raw_scores_test)
    """

    def __init__(self):
        self._iso = IsotonicRegression(
            out_of_bounds="clip",
            increasing=True,
        )

    def fit(
        self,
        scores: np.ndarray,
        y: np.ndarray,
    ) -> IsotonicCalibrator:
        """
        Fit isotonic regression to map raw scores → calibrated probability.

        Parameters
        ----------
        scores:
            Raw model output scores or probabilities (1-D array).
        y:
            True binary labels.
        """

        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y, dtype=float)
        self._iso.fit(scores, y)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """
        Return calibrated probabilities.

        Parameters
        ----------
        scores:
            Raw scores to calibrate (1-D array).

        Returns
        -------
        1-D array of calibrated probabilities for the positive class.
        """

        scores = np.asarray(scores, dtype=float)
        return self._iso.predict(scores)

    def is_fitted(self) -> bool:
        """Return True if the calibrator has been fitted."""
        return hasattr(self._iso, "f_")


def fit_calibrator(
    scores: np.ndarray,
    y: np.ndarray,
    method: str = "isotonic",
) -> PlattScaler | IsotonicCalibrator:
    """
    Fit and return a calibrator.

    This must be called only on held-out validation data,
    never on training data, to avoid leakage.

    Parameters
    ----------
    scores:
        Raw model probabilities from the validation set.
    y:
        True labels from the validation set.
    method:
        'isotonic' or 'platt'. Defaults to 'isotonic' because
        GMSC has enough validation data for isotonic to work well.

    Returns
    -------
    Fitted calibrator (PlattScaler or IsotonicCalibrator).
    """

    if method == "isotonic":
        calibrator = IsotonicCalibrator()
    elif method == "platt":
        calibrator = PlattScaler()
    else:
        raise ValueError(
            f"Unknown calibration method '{method}'. "
            "Choose 'isotonic' or 'platt'."
        )

    calibrator.fit(scores, y)
    return calibrator


def calibrate_probabilities(
    calibrator: PlattScaler | IsotonicCalibrator,
    scores: np.ndarray,
) -> np.ndarray:
    """
    Apply a fitted calibrator to raw scores.

    Parameters
    ----------
    calibrator:
        A fitted PlattScaler or IsotonicCalibrator.
    scores:
        Raw model probabilities to calibrate.

    Returns
    -------
    Calibrated probability array clipped to [0, 1].
    """

    probs = calibrator.predict_proba(scores)
    return np.clip(probs, 0.0, 1.0)
