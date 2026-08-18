<div align="center">

# credit-default-pd-model

A machine learning pipeline for predicting borrower Probability of Default (PD) on the Give Me Some Credit (GMSC) dataset, featuring Optuna hyperparameter tuning, isotonic probability calibration, SHAP explainability, and real-time AWS SageMaker deployment.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange?logo=amazonaws&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blue?logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)
![Optuna](https://img.shields.io/badge/Tuning-Optuna-lightgrey)

</div>

---

## Overview

This repository implements a **Probability of Default (PD) model** — a binary classifier estimating the likelihood that a credit borrower experiences serious delinquency (90+ days past due) within two years.

It serves as the **PD Subsystem** within a broader **Financial Analyst AI** platform. The backend service consumes this model's calibrated output to calculate **Expected Loss**:

$$\text{Expected Loss} = \text{PD} \times \text{LGD} \times \text{EAD}$$

```
Financial Analyst AI (Backend Service)
  ├── Credit decisioning workflows & REST APIs
  ├── Loss Given Default (LGD) & Exposure at Default (EAD)
  └── Expected Loss = PD x LGD x EAD
        │
        │ Requests calibrated PD for a borrower
        ▼
credit-default-pd-model (This Repository)
  ├── Preprocessing, feature engineering & Optuna tuning
  ├── Calibrated probability estimation (Isotonic Regression)
  └── Hosted on AWS SageMaker Real-Time Endpoint
```

Because the output directly scales monetary risk in Expected Loss, **well-calibrated probabilities are a strict requirement**.

---

## Dataset

**Give Me Some Credit (GMSC)** — Kaggle / Industry Benchmark.

| Property | Value |
|---|---|
| **Total Rows** | 150,000 |
| **Features** | 10 raw financial attributes |
| **Target Variable** | `SeriousDlqin2yrs` (Binary) |
| **Positive Class Rate** | 6.68% (Severe Class Imbalance) |
| **Missing Values** | `MonthlyIncome` (19.8%), `NumberOfDependents` (2.6%) |

---

## Performance Leaderboard & Calibration

Three candidate models were evaluated on a held-out validation set ($N=22,500$) using dynamic class imbalance weighting ($\text{scale\_pos\_weight} \approx 13.96$).

### Validation Leaderboard (Optuna-Tuned, 50 Trials)

| Model | ROC-AUC | PR-AUC | Brier Score | Status |
|---|---|---|---|---|
| **LightGBM (Tuned)** | **0.8731** | **0.4149** | **0.1417** | **Selected for Production** |
| XGBoost (Tuned) | 0.8725 | 0.4137 | 0.1396 | Benchmark |
| Logistic Regression | 0.8678 | 0.3981 | 0.1442 | Baseline |

### Isotonic Probability Calibration (Held-Out Test Set)

Raw gradient-boosting scores are uncalibrated probabilities. Applying Isotonic Regression trained on validation predictions yielded substantial calibration gains on the test set ($N=22,500$):

- **Raw Model Brier Score**: `0.1421`
- **Calibrated Brier Score**: **`0.0487`** (*~65.7% error reduction*)
- **Brier Skill Score (BSS)**: **`0.2190`**

---

## Feature Engineering & Preprocessing

### Deterministic Features
All feature engineering is strictly deterministic with zero learned parameters:
- `MonthlyIncome_missing`, `NumberOfDependents_missing` (Missingness indicators)
- `TotalDelinquencyCount` (Sum of 30-59, 60-89, and 90+ day late counts)
- `HasDelinquency`, `SevereDelinquency` (Binary risk flags)
- `RealEstateLoanRatio`, `IncomePerCreditLineLoan`, `PercentageTimePastDue`

### Leakage-Free Preprocessing Pipeline
```
Raw Borrower Attributes
        │
        ▼
GMSCDataCleaner (Drops IDs, cleans invalid age <= 0, computes features)
        │
        ▼
SimpleImputer (Median imputation fitted on X_train only)
        │
        ▼
QuantileClipper (0.5th – 99.5th percentile bounds fitted on X_train only)
        │
        ▼
RobustScaler (Centres on median, scales by IQR)
```

---

## Explainability (SHAP)

Global feature importance calculated via SHAP `TreeExplainer` on the tuned LightGBM production model:

| Rank | Feature | Mean \|SHAP\| | Impact Description |
|---|---|---|---|
| **1** | `RevolvingUtilizationOfUnsecuredLines` | **0.6511** | Credit card utilization ratio (Primary driver) |
| **2** | `TotalDelinquencyCount` | **0.2944** | Combined historical delinquency events |
| **3** | `age` | **0.2433** | Borrower age |
| **4** | `HasDelinquency` | **0.2098** | Binary indicator of prior late payments |
| **5** | `NumberOfOpenCreditLinesAndLoans` | **0.1120** | Total open credit accounts |

---

## Repository Structure

```
financial-risk-analyst-ml/
│
├── src/financial_risk_analyst_ml/   # Core Package (10 modular files)
│   ├── config.py           # S3 paths, AWS region, and configuration constants
│   ├── features.py         # Deterministic feature engineering
│   ├── preprocessing.py    # Leakage-free preprocessing pipeline
│   ├── models.py           # Model builders (Logistic Regression, XGBoost, LightGBM)
│   ├── tuning.py           # Optuna hyperparameter optimization engine
│   ├── evaluation.py       # ROC-AUC, PR-AUC, Brier score, and BSS metrics
│   ├── calibration.py      # Platt & Isotonic probability calibration
│   ├── explain.py          # SHAP explainability utilities
│   ├── train.py            # Main training execution script (with auto-S3 download)
│   └── inference.py        # SageMaker real-time serving handler
│
├── scripts/                         # Operational CLI Entry Points
│   ├── train_sagemaker.py  # Submit managed spot training job to AWS SageMaker
│   ├── deploy_sagemaker.py # Deploy model artifact to SageMaker real-time endpoint
│   └── invoke_endpoint.py  # Test live SageMaker endpoint with sample payload
│
├── sagemaker/
│   └── requirements.txt    # SageMaker container dependencies
│
├── tests/                           # Unit test suite (98 passing tests)
├── pyproject.toml
└── README.md
```

---

## Getting Started

### Local Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

```bash
# Clone the repository
git clone https://github.com/adhvaith267/credit-default-pd-model.git
cd credit-default-pd-model

# Install dependencies and sync environment
uv sync

# Run test suite
uv run pytest tests/ -v
```

### Local Model Training

Train and evaluate models locally. If `./cs-training.csv` is not present locally, `train.py` will automatically download it from S3:

```bash
# Train all models with 50 Optuna trials
uv run python -m financial_risk_analyst_ml.train --data-path cs-training.csv --model all --tune --tune-trials 50
```

---

## AWS SageMaker Deployment Workflow

### Prerequisites
1. AWS CLI configured (`aws configure` or SSO credentials).
2. IAM Role `FinancialRiskSageMakerExecutionRole` with SageMaker and S3 permissions.
3. Dataset uploaded to `s3://financial-risk-analyst-adhvaith-2026/datasets/gmsc/raw/cs-training.csv`.

### Step 1: Submit SageMaker Spot Training Job
Submits a managed spot training job (`ml.m5.xlarge`) to AWS SageMaker:

```bash
# Full Optuna tuning job (~70% cost savings via spot instances)
uv run python scripts/train_sagemaker.py

# Quick execution without hyperparameter tuning (~2-3 mins)
uv run python scripts/train_sagemaker.py --no-tune
```

### Step 2: Deploy to SageMaker Real-Time Endpoint
Deploys the trained `model.tar.gz` from S3 to a real-time SageMaker endpoint (`gmsc-pd-endpoint`):

```bash
uv run python scripts/deploy_sagemaker.py
```

### Step 3: Invoke & Validate the Live Endpoint
Sends a sample borrower payload to the live endpoint to verify inference latency and output structure:

```bash
uv run python scripts/invoke_endpoint.py --pretty
```

**Sample Request Payload:**
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

**Sample Response Output:**
```json
{
  "pd": 0.083,
  "model_version": "gmsc-lgb-v1"
}
```

---

## Core Engineering Decisions

1. **Single Production Model over Ensembles**: Benchmarks on GMSC show that model stacking yields negligible ROC-AUC gain over a single well-tuned LightGBM model. Serving a single model dramatically reduces endpoint latency, deployment complexity, and monitoring overhead.
2. **Mandatory Calibration**: Because the PD output directly scales Expected Loss ($\text{PD} \times \text{LGD} \times \text{EAD}$), uncalibrated models cause severe mispricing of credit risk. Isotonic calibration is enforced prior to model packaging.
3. **Container Compatibility**: Pinned dependencies and compatibility fixes ensure seamless execution across local Python 3.11 environments and AWS SageMaker Python 3.9 containers.

---
