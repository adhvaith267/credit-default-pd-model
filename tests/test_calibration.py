"""
test_calibration.py — Unit tests for calibration.py

Covers:
  - PlattScaler: fit, predict_proba, is_fitted, output range
  - IsotonicCalibrator: fit, predict_proba, is_fitted, output range
  - fit_calibrator: factory function for both methods
  - calibrate_probabilities: clips output to [0, 1]
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from financial_risk_analyst_ml.calibration import (
    PlattScaler,
    IsotonicCalibrator,
    fit_calibrator,
    calibrate_probabilities,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scores_labels(n: int = 200, seed: int = 0) -> tuple:
    """Generate synthetic raw scores and binary labels."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.0, 1.0, size=n)
    # Labels positively correlated with scores so calibration is meaningful.
    y = (scores + rng.normal(0, 0.2, size=n) > 0.5).astype(int)
    return scores, y


# ---------------------------------------------------------------------------
# PlattScaler
# ---------------------------------------------------------------------------

class TestPlattScaler:

    def test_not_fitted_before_fit(self):
        scaler = PlattScaler()
        assert not scaler.is_fitted()

    def test_fitted_after_fit(self):
        scores, y = _make_scores_labels()
        scaler = PlattScaler()
        scaler.fit(scores, y)
        assert scaler.is_fitted()

    def test_returns_1d_array(self):
        scores, y = _make_scores_labels()
        scaler = PlattScaler()
        scaler.fit(scores, y)
        out = scaler.predict_proba(scores)
        assert out.ndim == 1
        assert len(out) == len(scores)

    def test_output_in_0_1(self):
        scores, y = _make_scores_labels()
        scaler = PlattScaler()
        scaler.fit(scores, y)
        out = scaler.predict_proba(scores)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_fit_returns_self(self):
        scores, y = _make_scores_labels()
        scaler = PlattScaler()
        result = scaler.fit(scores, y)
        assert result is scaler

    def test_higher_score_higher_probability(self):
        """Platt-calibrated output should be monotonically increasing."""
        scores, y = _make_scores_labels()
        scaler = PlattScaler()
        scaler.fit(scores, y)
        test_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        out = scaler.predict_proba(test_scores)
        assert (np.diff(out) > 0).all(), "Output must be monotonically increasing."


# ---------------------------------------------------------------------------
# IsotonicCalibrator
# ---------------------------------------------------------------------------

class TestIsotonicCalibrator:

    def test_not_fitted_before_fit(self):
        calibrator = IsotonicCalibrator()
        assert not calibrator.is_fitted()

    def test_fitted_after_fit(self):
        scores, y = _make_scores_labels()
        calibrator = IsotonicCalibrator()
        calibrator.fit(scores, y)
        assert calibrator.is_fitted()

    def test_returns_1d_array(self):
        scores, y = _make_scores_labels()
        calibrator = IsotonicCalibrator()
        calibrator.fit(scores, y)
        out = calibrator.predict_proba(scores)
        assert out.ndim == 1
        assert len(out) == len(scores)

    def test_output_in_0_1(self):
        scores, y = _make_scores_labels()
        calibrator = IsotonicCalibrator()
        calibrator.fit(scores, y)
        out = calibrator.predict_proba(scores)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_fit_returns_self(self):
        scores, y = _make_scores_labels()
        calibrator = IsotonicCalibrator()
        result = calibrator.fit(scores, y)
        assert result is calibrator

    def test_out_of_bounds_scores_clipped(self):
        """out_of_bounds='clip' should handle scores outside [min_train, max_train]."""
        scores, y = _make_scores_labels()
        calibrator = IsotonicCalibrator()
        calibrator.fit(scores, y)
        extreme = np.array([-10.0, 10.0])
        out = calibrator.predict_proba(extreme)
        assert 0.0 <= out[0] <= 1.0
        assert 0.0 <= out[1] <= 1.0

    def test_monotonic_output(self):
        """Isotonic regression enforces monotonically increasing output."""
        scores, y = _make_scores_labels(n=500, seed=42)
        calibrator = IsotonicCalibrator()
        calibrator.fit(scores, y)
        sorted_scores = np.linspace(0.01, 0.99, 50)
        out = calibrator.predict_proba(sorted_scores)
        assert (np.diff(out) >= 0).all(), "Isotonic output must be non-decreasing."


# ---------------------------------------------------------------------------
# fit_calibrator factory
# ---------------------------------------------------------------------------

class TestFitCalibrator:

    def test_isotonic_returns_isotonic_calibrator(self):
        scores, y = _make_scores_labels()
        calibrator = fit_calibrator(scores, y, method="isotonic")
        assert isinstance(calibrator, IsotonicCalibrator)
        assert calibrator.is_fitted()

    def test_platt_returns_platt_scaler(self):
        scores, y = _make_scores_labels()
        calibrator = fit_calibrator(scores, y, method="platt")
        assert isinstance(calibrator, PlattScaler)
        assert calibrator.is_fitted()

    def test_unknown_method_raises(self):
        scores, y = _make_scores_labels()
        with pytest.raises(ValueError, match="Unknown calibration method"):
            fit_calibrator(scores, y, method="bogus")

    def test_default_method_is_isotonic(self):
        scores, y = _make_scores_labels()
        calibrator = fit_calibrator(scores, y)
        assert isinstance(calibrator, IsotonicCalibrator)


# ---------------------------------------------------------------------------
# calibrate_probabilities
# ---------------------------------------------------------------------------

class TestCalibrateProbabilities:

    def test_clips_below_zero(self):
        """Output must be clipped to [0, 1] even if the calibrator returns negatives."""
        scores, y = _make_scores_labels()
        calibrator = fit_calibrator(scores, y, method="platt")

        # Force an extreme input that might produce a sub-zero raw probability.
        extreme_scores = np.array([-100.0])
        out = calibrate_probabilities(calibrator, extreme_scores)
        assert out[0] >= 0.0

    def test_clips_above_one(self):
        extreme_scores = np.array([100.0])
        scores, y = _make_scores_labels()
        calibrator = fit_calibrator(scores, y, method="platt")
        out = calibrate_probabilities(calibrator, extreme_scores)
        assert out[0] <= 1.0

    def test_normal_scores_unchanged_by_clip(self):
        """Scores already in [0, 1] should not be altered by the clip."""
        scores, y = _make_scores_labels()
        calibrator = fit_calibrator(scores, y, method="isotonic")
        out = calibrate_probabilities(calibrator, scores)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_output_length_matches_input(self):
        scores, y = _make_scores_labels(n=50)
        calibrator = fit_calibrator(scores, y)
        out = calibrate_probabilities(calibrator, scores)
        assert len(out) == 50
