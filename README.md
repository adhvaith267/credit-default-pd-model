<div align="center">

# credit-default-pd-model

A machine learning model that predicts the Probability of Default (PD) for individual borrowers, trained on the Give Me Some Credit dataset using LightGBM with Optuna hyperparameter tuning and deployed on AWS SageMaker.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange?logo=amazonaws&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blue?logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)
![Optuna](https://img.shields.io/badge/Tuning-Optuna-lightgrey)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

## Overview

This repository implements a **Probability of Default (PD) model** — a binary classifier that estimates the likelihood a borrower will experience serious credit delinquency within two years. It covers the full ML lifecycle: data preprocessing, feature engineering, model training, hyperparameter tuning, calibration, explainability, and real-time inference via AWS SageMaker.

This model is one subsystem of a larger **Financial Analyst AI** platform. The broader platform performs end-to-end credit analysis for individual borrowers. This repository is responsible solely for producing a calibrated PD score. All other credit engine logic lives in the backend service.

```
Financial Analyst AI  (backend service)
  - REST API and credit decision workflows
  - Loss Given Default (LGD)
  - Exposure at Default (EAD)
  - Expected Loss = PD x LGD x EAD
  - Risk band assignment and business rules
        |
        | requests PD for a borrower
        v
credit-default-pd-model  (this repository)
  - Probability of Default inference
  - Preprocessing and feature engineering
  - Hosted on AWS SageMaker real-time endpoint
```

The output of this model — a single calibrated probability between 0 and 1 — feeds directly into the Expected Loss calculation in the backend. Well-calibrated probabilities are therefore a hard requirement, not an optional enhancement.

---

## Dataset

**Give Me Some Credit (GMSC)** — Kaggle, 2011.

| Property | Value |
|---|---|
| Rows | 150,000 |
| Features | 10 raw numeric |
| Target | SeriousDlqin2yrs |
| Positive class rate | 6.68% |
| Missing values | MonthlyIncome (19.8%), NumberOfDependents (2.6%) |

The dataset is stored in S3 and is not committed to this repository.

---

## Model Architecture

Three candidate models are trained and compared on a held-out validation set. The best model by ROC-AUC is selected, calibrated, and deployed. No ensembling is used — a single production model is served.

### Logistic Regression

A regularised logistic regression with balanced class weights. Serves as the interpretable baseline. Coefficients are directly auditable, which matters in regulated credit environments. Competitive ROC-AUC on this dataset despite its simplicity.

### XGBoost

Gradient-boosted decision trees with `scale_pos_weight` for class imbalance. Hyperparameters are tuned via Optuna using the TPE sampler over 50 trials, optimising validation ROC-AUC. Strong non-linear modelling capacity.

### LightGBM

Leaf-wise gradient-boosted trees. Faster to train than XGBoost and marginally better on tabular credit data. Uses `is_unbalance=True` for class imbalance. Hyperparameters are tuned via Optuna in parallel with XGBoost. Currently the best-performing model on GMSC.

### Model Comparison (untuned defaults, validation set)

| Model | ROC-AUC | PR-AUC | Brier Score |
|---|---|---|---|
| Logistic Regression | 0.8678 | 0.3981 | 0.1442 |
| XGBoost | 0.8658 | 0.4060 | 0.1286 |
| LightGBM | 0.8620 | 0.4031 | 0.1200 |

### Model Comparison (Optuna-tuned, 50 trials, validation set)

| Model | ROC-AUC | PR-AUC | Brier Score |
|---|---|---|---|
| LightGBM (tuned) | 0.8731 | 0.4149 | 0.1417 |
| XGBoost (tuned) | 0.8725 | 0.4137 | 0.1396 |
| Logistic Regression | 0.8678 | 0.3981 | 0.1442 |

### Model Tradeoffs

**Logistic Regression** is the most interpretable and easiest to audit. Each feature contributes a signed coefficient that can be explained to a credit committee. The main limitation is that it cannot capture non-linear relationships.

**XGBoost** captures non-linearities and feature interactions. It is the industry standard for credit risk tabular models. The tradeoff is interpretability, addressed here via SHAP.

**LightGBM** achieves similar or better accuracy than XGBoost while training significantly faster, which matters when running Optuna hyperparameter searches. Chosen as the production model.

**Why not an ensemble?** Published benchmarks on GMSC show that stacking or averaging multiple models produces negligible improvement in ROC-AUC over a single well-tuned model. The added deployment, monitoring, and versioning complexity is not justified.

---

## Feature Engineering

All feature engineering is deterministic — no parameters are learned, so it can safely happen before the train/validation/test split.

| Feature | Description |
|---|---|
| `MonthlyIncome_missing` | Indicator: 1 if MonthlyIncome was NaN |
| `NumberOfDependents_missing` | Indicator: 1 if NumberOfDependents was NaN |
| `TotalDelinquencyCount` | Sum of all delinquency count columns |
| `HasDelinquency` | 1 if any delinquency event exists |
| `SevereDelinquency` | 1 if any 90+ day late payment exists |
| `RealEstateLoanRatio` | Real estate loans / total open credit lines |
| `IncomePerCreditLineLoan` | Monthly income / total open credit lines |
| `PercentageTimePastDue` | 30–59 day delinquencies / total credit lines |

---

## Preprocessing Pipeline

The pipeline is fitted exclusively on training data and applied identically to validation and test sets to prevent data leakage.

```
Raw borrower features
        |
        v
GMSCDataCleaner
  - Drop identifier column (Unnamed: 0)
  - Set age <= 0 to NaN
  - Add engineered features
        |
        v
SimpleImputer (median strategy)
        |
        v
QuantileClipper (0.5th - 99.5th percentile)
  - Bounds learned from training data only
        |
        v
RobustScaler
  - Centres on median, scales by IQR
```

---

## Calibration

Raw model probabilities are not guaranteed to be well-calibrated. Calibration is performed using isotonic regression fitted on the validation set. The calibration step reduced the Brier score from 0.1421 to 0.0487 on the test set, achieving a Brier Skill Score (BSS) of 0.2190.

---

## Explainability

SHAP TreeExplainer is used for both XGBoost and LightGBM. Two levels of explanation are available:

- **Global importance**: mean absolute SHAP value across the test set.
- **Per-borrower explanation**: signed SHAP values for a single inference.

Top features by mean absolute SHAP value on the test set (LightGBM, tuned):

| Rank | Feature | Mean Absolute SHAP |
|---|---|---|
| 1 | RevolvingUtilizationOfUnsecuredLines | 0.6511 |
| 2 | TotalDelinquencyCount | 0.2944 |
| 3 | age | 0.2433 |
| 4 | HasDelinquency | 0.2098 |
| 5 | NumberOfOpenCreditLinesAndLoans | 0.1120 |

---

## AWS Architecture

```
GitHub (source code)
        |
        v
SageMaker Training Job  (train_sagemaker.py)
  - Reads dataset from S3
  - Runs train.py inside managed sklearn container
  - Writes model artifacts to S3
        |
        v
S3 (model.tar.gz)
        |
        v
SageMaker Endpoint  (deploy_sagemaker.py)
  - Runs inference.py
  - Accepts JSON borrower features
  - Returns calibrated PD
        |
        v
Backend Service
```

---

## Repository Structure

```
financial-risk-analyst-ml/
|
|-- src/financial_risk_analyst_ml/
|   |-- config.py           S3 paths, AWS region, and configuration constants
|   |-- features.py         Deterministic feature engineering
|   |-- preprocessing.py    Cleaning, imputation, quantile clipping, robust scaling
|   |-- models.py           Model constructors (Logistic Regression, XGBoost, LightGBM)
|   |-- tuning.py           Optuna hyperparameter tuning
|   |-- evaluation.py       ROC-AUC, PR-AUC, Brier score, and threshold evaluation
|   |-- calibration.py      Platt scaling and Isotonic probability calibration
|   |-- explain.py          SHAP feature importance and explanation utilities
|   |-- train.py            Training script entry point (SageMaker + local execution)
|   |-- inference.py        SageMaker real-time inference handler
|
|-- scripts/
|   |-- download_data.py    Download GMSC dataset from S3 for local training
|   |-- train_sagemaker.py  Submit SageMaker training job
|   |-- deploy_sagemaker.py Deploy model artifact to SageMaker real-time endpoint
|   |-- invoke_endpoint.py  Test live SageMaker endpoint with sample payload
|
|-- sagemaker/
|   |-- requirements.txt    Dependencies installed inside SageMaker container
|
|-- tests/
|   |-- test_calibration.py
|   |-- test_evaluation.py
|   |-- test_features.py
|   |-- test_inference.py
|   |-- test_preprocessing.py
|
|-- pyproject.toml
|-- uv.lock
```

---

## Local Development

This project uses [uv](https://github.com/astral-sh/uv) for environment management.

**Run tests:**

```bash
uv run --with pandas --with numpy --with scikit-learn pytest tests/ -v
```

---

## SageMaker Workflow

### Prerequisites

- AWS credentials configured (`aws configure` or IAM Identity Center)
- IAM role `FinancialRiskSageMakerExecutionRole` exists with SageMaker full access and S3 read/write
- Dataset uploaded to S3: `s3://financial-risk-analyst-adhvaith-2026/datasets/gmsc/raw/cs-training.csv`
- Service quota approved: `ml.m5.xlarge for training job usage` ≥ 1 (request via AWS Service Quotas)

### 1. Submit training job

```bash
uv run python scripts/train_sagemaker.py
```

Options:

```bash
# Choose model and tuning settings
uv run python scripts/train_sagemaker.py \
  --model all \
  --tune-trials 50 \
  --instance-type ml.m5.xlarge

# Skip tuning for a quick run
uv run python scripts/train_sagemaker.py --no-tune

# Submit and exit without waiting
uv run python scripts/train_sagemaker.py --no-wait
```

The script uses **managed spot training by default** (~70% cheaper). On completion it prints the model artifact S3 URI.

### 2. Deploy to endpoint

```bash
uv run python scripts/deploy_sagemaker.py \
  --model-artifact s3://financial-risk-analyst-adhvaith-2026/models/gmsc/<job-name>/output/model.tar.gz
```

Options:

```bash
uv run python scripts/deploy_sagemaker.py \
  --model-artifact s3://... \
  --endpoint-name gmsc-pd-endpoint \
  --instance-type ml.m5.xlarge
```

### 3. Test the endpoint

```bash
uv run python scripts/invoke_endpoint.py --pretty
```

**Request format:**

```json
{
  "RevolvingUtilizationOfUnsecuredLines": 0.766,
  "age": 45,
  "NumberOfTime30-59DaysPastDueNotWorse": 2,
  "DebtRatio": 0.80,
  "MonthlyIncome": 9120.0,
  "NumberOfOpenCreditLinesAndLoans": 13,
  "NumberOfTimes90DaysLate": 0,
  "NumberRealEstateLoansOrLines": 6,
  "NumberOfTime60-89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 2.0
}
```

**Response format:**

```json
{
  "pd": 0.083,
  "model_version": "gmsc-lgb-v1"
}
```

---

## Evaluation Metrics

Accuracy is not used as a primary metric. With a 6.68% positive class, a model that always predicts no default achieves 93.32% accuracy and is completely useless.

Primary metrics:

- **ROC-AUC**: measures the model's ability to rank borrowers by risk.
- **PR-AUC**: precision-recall AUC. More informative under severe class imbalance.
- **Brier Score**: mean squared error between predicted probability and true label. Measures calibration quality.
- **Brier Skill Score**: improvement over a naive baseline that always predicts the base rate.

---

## Design Decisions

**Single model, not an ensemble.** Published GMSC benchmarks show that stacking produces no meaningful ROC-AUC improvement over a single tuned model. A single model is simpler to deploy, monitor, version, and explain.

**Calibration over raw probabilities.** The backend uses PD in `Expected Loss = PD x LGD x EAD`. A model with poor calibration will systematically misprice risk. Isotonic calibration is applied as a mandatory step.

**No data in Git.** Datasets and model artifacts live in S3. The repository contains code, configuration, and tests only.

**Spot training by default.** Managed spot training is enabled by default in `train_sagemaker.py`. For a ~30-minute training job the interruption risk is low and the cost saving is ~70%.

**SageMaker v2 SDK.** Both `train_sagemaker.py` and `deploy_sagemaker.py` use the SageMaker v2 SDK (`sagemaker>=2.231,<3.0`) with `SKLearn` estimator and `SKLearnModel` for training and deployment respectively.

---

## Requirements

**Local environment:**

- Python 3.11
- uv
- AWS CLI configured with appropriate credentials

**Python dependencies (managed by uv):**

- `boto3` — AWS SDK
- `sagemaker>=2.231,<3.0` — SageMaker v2 SDK for job submission and deployment

**SageMaker container (see `sagemaker/requirements.txt`):**

- pandas, numpy, scikit-learn
- xgboost, lightgbm
- optuna
- shap, joblib
