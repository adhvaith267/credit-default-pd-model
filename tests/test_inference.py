"""
test_inference.py — Unit tests for inference.py

These tests exercise the SageMaker handler functions in isolation using
lightweight stubs so they can run without a trained model on disk.
"""

from __future__ import annotations

import json
import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from unittest.mock import MagicMock, patch
from financial_risk_analyst_ml.inference import (
    input_fn,
    predict_fn,
    output_fn,
    RAW_INPUT_COLUMNS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_BORROWER = {
    "RevolvingUtilizationOfUnsecuredLines": 0.766,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 2,
    "DebtRatio": 0.80,
    "MonthlyIncome": 9120.0,
    "NumberOfOpenCreditLinesAndLoans": 13,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 6,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2.0,
}


# ---------------------------------------------------------------------------
# input_fn
# ---------------------------------------------------------------------------

class TestInputFn:

    def test_single_borrower_dict(self):
        body = json.dumps(VALID_BORROWER)
        df = input_fn(body, "application/json")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        for col in RAW_INPUT_COLUMNS:
            assert col in df.columns

    def test_batch_borrowers_list(self):
        body = json.dumps([VALID_BORROWER, VALID_BORROWER])
        df = input_fn(body, "application/json")
        assert len(df) == 2

    def test_unsupported_content_type_raises(self):
        body = json.dumps(VALID_BORROWER)
        with pytest.raises(ValueError, match="Unsupported content type"):
            input_fn(body, "text/csv")

    def test_missing_column_raises(self):
        incomplete = {k: v for k, v in VALID_BORROWER.items() if k != "age"}
        body = json.dumps(incomplete)
        with pytest.raises(ValueError, match="missing required columns"):
            input_fn(body, "application/json")

    def test_values_are_correct(self):
        body = json.dumps(VALID_BORROWER)
        df = input_fn(body, "application/json")
        assert df["age"].iloc[0] == 45
        assert abs(df["RevolvingUtilizationOfUnsecuredLines"].iloc[0] - 0.766) < 1e-9


# ---------------------------------------------------------------------------
# predict_fn
# ---------------------------------------------------------------------------

class TestPredictFn:

    def _make_model_artifacts(self, pd_value: float = 0.083) -> dict:
        """Build a mock model artifacts dict."""
        preprocessor = MagicMock()
        preprocessor.transform.return_value = np.zeros((1, 16))

        model = MagicMock()
        model.predict_proba.return_value = np.array([[1 - pd_value, pd_value]])

        calibrator = MagicMock()
        calibrator.predict_proba.return_value = np.array([pd_value])

        return {
            "model": model,
            "preprocessor": preprocessor,
            "calibrator": calibrator,
        }

    def test_returns_list_of_dicts(self):
        df = pd.DataFrame([VALID_BORROWER])
        artifacts = self._make_model_artifacts(pd_value=0.083)
        result = predict_fn(df, artifacts)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "pd" in result[0]
        assert "status" in result[0]
        assert "risk_drivers" in result[0]

    def test_single_borrower_returns_one_value(self):
        df = pd.DataFrame([VALID_BORROWER])
        artifacts = self._make_model_artifacts(pd_value=0.083)
        result = predict_fn(df, artifacts)
        assert len(result) == 1

    def test_pd_value_is_clipped_to_0_1(self):
        df = pd.DataFrame([VALID_BORROWER])
        artifacts = self._make_model_artifacts(pd_value=0.5)
        # Override calibrator to return out-of-range value.
        artifacts["calibrator"].predict_proba.return_value = np.array([1.5])
        result = predict_fn(df, artifacts)
        assert result[0]["pd"] <= 1.0

    def test_preprocessor_is_called(self):
        df = pd.DataFrame([VALID_BORROWER])
        artifacts = self._make_model_artifacts()
        predict_fn(df, artifacts)
        artifacts["preprocessor"].transform.assert_called_once()

    def test_model_predict_proba_is_called(self):
        df = pd.DataFrame([VALID_BORROWER])
        artifacts = self._make_model_artifacts()
        predict_fn(df, artifacts)
        artifacts["model"].predict_proba.assert_called_once()


# ---------------------------------------------------------------------------
# output_fn
# ---------------------------------------------------------------------------

class TestOutputFn:

    def test_single_result_unwrapped(self):
        preds = [{"pd": 0.083, "status": "APPROVED", "model_version": "gmsc-xgb-v1", "risk_drivers": []}]
        body, content_type = output_fn(preds, "application/json")
        data = json.loads(body)
        # Single borrower: should be a dict, not a list.
        assert isinstance(data, dict)
        assert "pd" in data
        assert "status" in data
        assert "model_version" in data
        assert "risk_drivers" in data

    def test_pd_value_is_correct(self):
        preds = [{"pd": 0.083, "status": "APPROVED", "model_version": "gmsc-xgb-v1", "risk_drivers": []}]
        body, _ = output_fn(preds, "application/json")
        data = json.loads(body)
        assert abs(data["pd"] - 0.083) < 1e-5

    def test_batch_result_is_list(self):
        preds = [
            {"pd": 0.083, "status": "APPROVED", "model_version": "gmsc-xgb-v1", "risk_drivers": []},
            {"pd": 0.354, "status": "DECLINED", "model_version": "gmsc-xgb-v1", "risk_drivers": ["High utilization"]},
        ]
        body, content_type = output_fn(preds, "application/json")
        data = json.loads(body)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_content_type_is_json(self):
        preds = [{"pd": 0.05, "status": "APPROVED", "model_version": "gmsc-xgb-v1", "risk_drivers": []}]
        _, content_type = output_fn(preds, "application/json")
        assert content_type == "application/json"

    def test_unsupported_accept_raises(self):
        preds = [{"pd": 0.05, "status": "APPROVED", "model_version": "gmsc-xgb-v1", "risk_drivers": []}]
        with pytest.raises(ValueError, match="Unsupported accept type"):
            output_fn(preds, "text/csv")

    def test_wildcard_accept_works(self):
        preds = [{"pd": 0.05, "status": "APPROVED", "model_version": "gmsc-xgb-v1", "risk_drivers": []}]
        body, content_type = output_fn(preds, "*/*")
        assert content_type == "application/json"

    def test_pd_is_rounded(self):
        preds = [{"pd": 0.123457, "status": "DECLINED", "model_version": "gmsc-xgb-v1", "risk_drivers": []}]
        body, _ = output_fn(preds, "application/json")
        data = json.loads(body)
        # Should be rounded to 6 decimal places.
        assert len(str(data["pd"]).split(".")[-1]) <= 6

