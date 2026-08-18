"""
test_evaluation.py — Unit tests for evaluation.py

Covers:
  - evaluate_model: metric correctness, output structure
  - compare_models: sorting, shape
  - brier_skill_score: perfect / naive / below-naive models
  - threshold_metrics: precision, recall, specificity, F1 at a threshold
  - EvaluationResult: to_dict, summary formatting
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from financial_risk_analyst_ml.evaluation import (
    EvaluationResult,
    evaluate_model,
    compare_models,
    brier_skill_score,
    threshold_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect_predictions(n: int = 100) -> tuple:
    """y_true == y_prob (perfectly calibrated, perfect discrimination)."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=n)
    y_prob = y_true.astype(float)
    return y_true, y_prob


def _random_predictions(n: int = 200, pos_rate: float = 0.1, seed: int = 0) -> tuple:
    """Simulated borrower predictions with realistic class imbalance."""
    rng = np.random.default_rng(seed)
    y_true = (rng.uniform(size=n) < pos_rate).astype(int)
    y_prob = rng.uniform(0.0, 0.3, size=n)
    y_prob[y_true == 1] += 0.2   # defaults score slightly higher on average
    y_prob = np.clip(y_prob, 0.0, 1.0)
    return y_true, y_prob


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:

    def test_returns_evaluation_result(self):
        y_true, y_prob = _random_predictions()
        result = evaluate_model(y_true, y_prob)
        assert isinstance(result, EvaluationResult)

    def test_roc_auc_in_range(self):
        y_true, y_prob = _random_predictions()
        result = evaluate_model(y_true, y_prob)
        assert 0.0 <= result.roc_auc <= 1.0

    def test_pr_auc_in_range(self):
        y_true, y_prob = _random_predictions()
        result = evaluate_model(y_true, y_prob)
        assert 0.0 <= result.pr_auc <= 1.0

    def test_brier_score_in_range(self):
        y_true, y_prob = _random_predictions()
        result = evaluate_model(y_true, y_prob)
        assert 0.0 <= result.brier_score <= 1.0

    def test_n_positive_and_negative_sum_to_total(self):
        y_true, y_prob = _random_predictions(n=200)
        result = evaluate_model(y_true, y_prob)
        assert result.n_positive + result.n_negative == 200

    def test_model_name_is_stored(self):
        y_true, y_prob = _random_predictions()
        result = evaluate_model(y_true, y_prob, model_name="xgboost")
        assert result.model_name == "xgboost"

    def test_split_is_stored(self):
        y_true, y_prob = _random_predictions()
        result = evaluate_model(y_true, y_prob, split="test")
        assert result.split == "test"

    def test_calibration_bins_present(self):
        y_true, y_prob = _random_predictions()
        result = evaluate_model(y_true, y_prob)
        assert "mean_predicted_prob" in result.calibration_bins
        assert "fraction_of_positives" in result.calibration_bins

    def test_perfect_model_roc_auc_is_1(self):
        y_true, y_prob = _perfect_predictions()
        result = evaluate_model(y_true, y_prob)
        assert abs(result.roc_auc - 1.0) < 1e-9

    def test_random_model_roc_auc_near_0_5(self):
        rng = np.random.default_rng(42)
        n = 2000
        y_true = rng.integers(0, 2, size=n)
        y_prob = rng.uniform(0.0, 1.0, size=n)
        result = evaluate_model(y_true, y_prob)
        # Random classifier should be close to 0.5 with enough samples.
        assert abs(result.roc_auc - 0.5) < 0.05


# ---------------------------------------------------------------------------
# EvaluationResult.to_dict / summary
# ---------------------------------------------------------------------------

class TestEvaluationResultMethods:

    def _make_result(self) -> EvaluationResult:
        y_true, y_prob = _random_predictions()
        return evaluate_model(y_true, y_prob, model_name="lgb", split="validation")

    def test_to_dict_has_required_keys(self):
        result = self._make_result()
        d = result.to_dict()
        for key in ("split", "model_name", "roc_auc", "pr_auc", "brier_score",
                    "n_positive", "n_negative"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_values_are_json_serializable(self):
        import json
        result = self._make_result()
        # Should not raise
        json.dumps(result.to_dict())

    def test_summary_is_string(self):
        result = self._make_result()
        assert isinstance(result.summary(), str)

    def test_summary_contains_model_name(self):
        result = self._make_result()
        assert "lgb" in result.summary()

    def test_summary_contains_split(self):
        result = self._make_result()
        assert "validation" in result.summary()


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:

    def test_returns_dataframe(self):
        y_true, y_prob = _random_predictions()
        r1 = evaluate_model(y_true, y_prob, model_name="lgb")
        r2 = evaluate_model(y_true, y_prob, model_name="xgb")
        df = compare_models([r1, r2])
        assert isinstance(df, pd.DataFrame)

    def test_rows_match_number_of_results(self):
        y_true, y_prob = _random_predictions()
        results = [evaluate_model(y_true, y_prob, model_name=f"m{i}") for i in range(3)]
        df = compare_models(results)
        assert len(df) == 3

    def test_sorted_by_roc_auc_descending(self):
        y_true, y_prob = _random_predictions()
        r1 = evaluate_model(y_true, y_prob * 0.5, model_name="weak")
        r2 = evaluate_model(y_true, y_prob,       model_name="strong")
        df = compare_models([r1, r2])
        roc_aucs = df["roc_auc"].tolist()
        assert roc_aucs == sorted(roc_aucs, reverse=True)

    def test_single_result(self):
        y_true, y_prob = _random_predictions()
        result = evaluate_model(y_true, y_prob)
        df = compare_models([result])
        assert len(df) == 1


# ---------------------------------------------------------------------------
# brier_skill_score
# ---------------------------------------------------------------------------

class TestBrierSkillScore:

    def test_perfect_model_bss_is_1(self):
        y_true, y_prob = _perfect_predictions(n=100)
        bss = brier_skill_score(y_true, y_prob)
        assert abs(bss - 1.0) < 1e-9

    def test_naive_model_bss_is_0(self):
        y_true, _ = _random_predictions(n=500)
        base_rate = y_true.mean()
        naive_probs = np.full_like(y_true, fill_value=base_rate, dtype=float)
        bss = brier_skill_score(y_true, naive_probs)
        assert abs(bss) < 1e-9

    def test_good_model_bss_positive(self):
        y_true, y_prob = _random_predictions(n=500)
        bss = brier_skill_score(y_true, y_prob)
        # A model with some signal should beat the naive baseline.
        assert bss > 0.0

    def test_inverted_model_bss_negative(self):
        """A model whose predictions are the opposite of truth is worse than naive."""
        y_true, y_prob = _random_predictions(n=500)
        bss = brier_skill_score(y_true, 1.0 - y_prob)
        assert bss < 0.0

    def test_all_same_labels_returns_zero(self):
        """When there is no positive class the naive Brier score is 0 → BSS = 0."""
        y_true = np.zeros(100, dtype=int)
        y_prob = np.zeros(100, dtype=float)
        bss = brier_skill_score(y_true, y_prob)
        assert bss == 0.0


# ---------------------------------------------------------------------------
# threshold_metrics
# ---------------------------------------------------------------------------

class TestThresholdMetrics:

    def test_returns_required_keys(self):
        y_true, y_prob = _random_predictions()
        metrics = threshold_metrics(y_true, y_prob, threshold=0.5)
        for key in ("threshold", "tp", "fp", "tn", "fn",
                    "precision", "recall", "specificity", "f1"):
            assert key in metrics, f"Missing key: {key}"

    def test_threshold_stored_in_output(self):
        y_true, y_prob = _random_predictions()
        metrics = threshold_metrics(y_true, y_prob, threshold=0.3)
        assert metrics["threshold"] == 0.3

    def test_perfect_classifier_metrics(self):
        """At the right threshold, a perfect classifier has precision=recall=f1=1."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = threshold_metrics(y_true, y_prob, threshold=0.5)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_no_positives_predicted_gives_zero_precision(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.1])
        metrics = threshold_metrics(y_true, y_prob, threshold=0.5)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_all_positives_predicted_gives_recall_1(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.9, 0.9, 0.9, 0.9])
        metrics = threshold_metrics(y_true, y_prob, threshold=0.5)
        assert metrics["recall"] == 1.0

    def test_counts_sum_to_total(self):
        y_true, y_prob = _random_predictions(n=100)
        metrics = threshold_metrics(y_true, y_prob, threshold=0.2)
        total = metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"]
        assert total == 100
