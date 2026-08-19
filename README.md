<div align="center">

# credit-default-pd-model

A production-grade machine learning pipeline for predicting borrower Probability of Default (PD) on the Give Me Some Credit (GMSC) dataset, featuring Optuna hyperparameter tuning, isotonic probability calibration, SHAP explainability, and real-time AWS SageMaker deployment with zero-downtime idempotent updates.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange?logo=amazonaws&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blue?logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)
![Optuna](https://img.shields.io/badge/Tuning-Optuna-lightgrey)

</div>

---

## Table of Contents
- [Overview & Business Context](#overview--business-context)
- [Dataset Specifications](#dataset-specifications)
- [Performance Leaderboard & Calibration](#performance-leaderboard--calibration)
- [Feature Engineering & Preprocessing](#feature-engineering--preprocessing)
- [Explainability Architecture (SHAP)](#explainability-architecture-shap)
- [SageMaker Infrastructure & Deployment Architecture](#sagemaker-infrastructure--deployment-architecture)
- [Repository Structure](#repository-structure)
- [Getting Started & Local Execution](#getting-started--local-execution)
- [Deployment & Operational Workflows](#deployment--operational-workflows)
- [Core Engineering & Architecture Decisions](#core-engineering--architecture-decisions)

---

## Overview & Business Context

This repository implements a **Probability of Default (PD) model** — a binary credit classification engine estimating the likelihood that a borrower will experience severe delinquency (90+ days past due) within two years.

It serves as the **PD Subsystem** within a broader **Financial Analyst AI** platform. The backend credit decisioning engine consumes this model's calibrated probability output to compute monetary **Expected Loss (EL)**:

$$\text{Expected Loss} = \text{PD} \times \text{LGD} \times \text{EAD}$$

```
Financial Analyst AI (Backend Service)
  ├── Credit decisioning workflows & REST APIs
  ├── Loss Given Default (LGD) & Exposure at Default (EAD)
  └── Expected Loss = PD x LGD x EAD
        │
        │ Requests calibrated PD for a borrower (sub-20ms)
        ▼
credit-default-pd-model (This Repository)
  ├── Preprocessing, feature engineering & Optuna tuning
  ├── Calibrated probability estimation (Isotonic Regression)
  ├── Global SHAP audit & local adverse action explanations
  └── Hosted on AWS SageMaker Real-Time Endpoint
```

Because output probabilities scale financial risk directly in capital allocation and credit pricing, **well-calibrated probabilities and strict regulatory compliance are absolute requirements**.

---

## Dataset Specifications

**Give Me Some Credit (GMSC)** — Kaggle / Industry Benchmark Credit Risk Dataset.

| Property | Value | Notes |
|---|---|---|
| **Total Samples** | 150,000 | 105,000 Train / 22,500 Val / 22,500 Test |
| **Features** | 10 raw borrower attributes | Financial utilization, age, debt ratio, income, delinquencies |
| **Target Variable** | `SeriousDlqin2yrs` | Binary (1 = Serious delinquency, 0 = Otherwise) |
| **Class Imbalance** | 6.68% Positive Rate | Handled via `scale_pos_weight` $\approx 13.96$ |
| **Missingness** | `MonthlyIncome` (19.8%), `NumberOfDependents` (2.6%) | Handled via median imputation with missingness flags |

---

## Performance Leaderboard & Calibration

Three candidate algorithms were optimized and evaluated on a held-out validation set ($N=22,500$) using dynamic class imbalance weighting.

### Validation Leaderboard (Optuna-Tuned, 50 Trials)

| Model | ROC-AUC | PR-AUC | Brier Score | Selection Status |
|---|---|---|---|---|
| **LightGBM (Tuned)** | **0.8731** | **0.4149** | **0.1417** | **Selected for Production** |
| XGBoost (Tuned) | 0.8725 | 0.4137 | 0.1396 | Benchmark |
| Logistic Regression | 0.8678 | 0.3981 | 0.1442 | Baseline |

### Isotonic Probability Calibration (Held-Out Test Set)

Raw gradient-boosting scores produce uncalibrated probabilities. Applying Isotonic Regression trained on validation predictions yielded significant calibration improvements on the test set ($N=22,500$):

- **Raw Model Brier Score**: `0.1421`
- **Calibrated Brier Score**: **`0.0487`** (*~65.7% error reduction*)
- **Brier Skill Score (BSS)**: **`0.2190`**

---

## Feature Engineering & Preprocessing

### Deterministic Features
All feature transformations are strictly deterministic with zero learned parameters:
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

## Explainability Architecture (SHAP)

### Why SHAP is Essential in Credit Risk
SHAP (SHapley Additive exPlanations) grounds model decisions in cooperative game theory. In credit risk, SHAP fulfills three critical requirements:
1. **Regulatory Compliance (Adverse Action Notices)**: Under laws such as the US Equal Credit Opportunity Act (ECOA) and FCRA, lenders are legally mandated to disclose the top specific financial reasons why an applicant was denied credit or given higher rates.
2. **Model Risk Governance (SR 11-7 / Basel III)**: Provides transparent validation that tree models rely on intuitive financial logic rather than spurious artifacts.
3. **Additive Credit Attribution**: Quantifies exact feature-level probability shifts for each borrower.

### Dual-Layer SHAP Architecture
To balance sub-20ms real-time API latency with regulatory compliance, SHAP is integrated across two layers:

```
                    ┌────────────────────────────────────────────────────────┐
                    │                    SHAP Architecture                   │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
┌─────────────────────────────────┐                           ┌──────────────────────────────────┐
│   Layer 1: Offline Compliance   │                           │   Layer 2: Real-Time In-Endpoint │
│       Audit (train.py)          │                           │    Adverse Action (inference.py) │
├─────────────────────────────────┤                           ├──────────────────────────────────┤
│ • Computes global feature rank  │                           │ • Automatically triggered when   │
│ • Saved to metrics.json         │                           │   PD >= RISK_THRESHOLD (0.10)   │
│ • Inspectable by Risk Audit     │                           │   or when explain=true requested │
│   Committee before deployment   │                           │ • Returns human-readable risk    │
│                                 │                           │   drivers in JSON payload        │
└─────────────────────────────────┘                           └──────────────────────────────────┘
```

#### Global Feature Impact (Production LightGBM Model)

| Rank | Feature | Mean \|SHAP\| | Impact Description |
|---|---|---|---|
| **1** | `RevolvingUtilizationOfUnsecuredLines` | **0.6511** | Credit card utilization ratio (Primary driver) |
| **2** | `TotalDelinquencyCount` | **0.2944** | Combined historical delinquency events |
| **3** | `age` | **0.2433** | Borrower age |
| **4** | `HasDelinquency` | **0.2098** | Binary indicator of prior late payments |
| **5** | `NumberOfOpenCreditLinesAndLoans` | **0.1120** | Total open credit accounts |

### Real-Time Endpoint Payload & Latency Strategy
The SageMaker real-time endpoint (`inference.py`) dynamically computes and attaches human-readable SHAP Adverse Action risk drivers directly in the HTTPS response payload whenever an applicant is flagged as **DECLINED** (`PD >= RISK_THRESHOLD`, default `0.10`) or when `"explain": true` is explicitly passed in the request body.

For low-risk **APPROVED** applications (`PD < 0.10` and `explain=false`), SHAP calculations are conditionally bypassed (`risk_drivers: []`), preserving **sub-20ms serving latency** for high-throughput decisioning. This decoupled microservice architecture satisfies Equal Credit Opportunity Act (ECOA) and FCRA Adverse Action notification mandates directly at the inference layer without incurring latency penalties on standard credit approvals.

---

## SageMaker Infrastructure & Deployment Architecture

### Production-Grade Idempotent Deployment (`scripts/deploy_sagemaker.py`)
Deploying real-time machine learning models in automated CI/CD pipelines requires robust state management to handle stale resources, race conditions, and zero-downtime updates.

```
                    ┌────────────────────────────────────────────────────────┐
                    │       Idempotent Deployment Flow (deploy_sagemaker)    │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                  1. Inspect describe_endpoint()
                                                │
        ┌───────────────────────┬───────────────┴───────────────┬────────────────────────┐
        ▼                       ▼                               ▼                        ▼
┌───────────────┐      ┌─────────────────┐             ┌─────────────────┐      ┌────────────────┐
│  Non-Existent │      │    InService    │             │ Creating/Updating│     │Failed/OutOfSvc │
└───────┬───────┘      └────────┬────────┘             └────────┬────────┘      └───────┬────────┘
        │                       │                               │                       │
        │               Creates Timestamped             Waits for InService             Deletes Failed
        │               EndpointConfig                  via AWS Waiters                 Endpoint & Config
        │                       │                               │                       │
        ▼                       ▼                               ▼                       ▼
Call create_endpoint()  Call update_endpoint()         Call update_endpoint()   Call create_endpoint()
(Fresh Endpoint)        (Zero-Downtime Update)         (In-Place Update)        (Fresh Endpoint)
```

#### Key Resilience Mechanics:
1. **Immutable EndpointConfigs**: Every deployment generates a unique, timestamped configuration name (`gmsc-pd-endpoint-cfg-YYYYMMDD-HHMMSS`). This completely prevents `ValidationException: Cannot create already existing endpoint configuration` errors.
2. **In-Service State Resolution**: If an existing endpoint is in `Creating` or `Updating` status, the deployment script uses AWS `boto3` waiters to safely wait until it reaches `InService` before triggering an in-place zero-downtime update.
3. **Automatic Failed Endpoint Cleanup**: Stale or failed endpoints (`Failed` or `OutOfService`) are automatically torn down along with their unused endpoint configurations before initiating a clean deployment.

---

## Repository Structure

```
financial-risk-analyst-ml/
│
├── src/
│   ├── financial_risk_analyst_ml/   # Core Modular Package (10 files)
│   │   ├── config.py           # S3 paths, AWS region, and configuration constants
│   │   ├── features.py         # Deterministic feature engineering
│   │   ├── preprocessing.py    # Leakage-free preprocessing pipeline
│   │   ├── models.py           # Model builders (Logistic Regression, XGBoost, LightGBM)
│   │   ├── tuning.py           # Optuna hyperparameter optimization engine
│   │   ├── evaluation.py       # ROC-AUC, PR-AUC, Brier score, and BSS metrics
│   │   ├── calibration.py      # Platt & Isotonic probability calibration
│   │   ├── explain.py          # SHAP explainability utilities
│   │   ├── train.py            # Main training execution script (auto S3 dataset fetch)
│   │   └── inference.py        # SageMaker real-time serving handler
│   ├── inference.py            # SageMaker SKLearn container entry point wrapper
│   └── train.py                # SageMaker SKLearn container entry point wrapper
│
├── scripts/                         # Operational Entry Points
│   ├── train_sagemaker.py  # Submit managed spot training job to AWS SageMaker
│   ├── deploy_sagemaker.py # Idempotent real-time endpoint deployment script
│   └── invoke_endpoint.py  # Test live SageMaker endpoint with sample payload
│
├── sagemaker/
│   └── requirements.txt    # SageMaker container dependencies (pinned ranges)
│
├── tests/                           # Comprehensive unit test suite (98 passing tests)
│   ├── test_features.py
│   ├── test_preprocessing.py
│   ├── test_calibration.py
│   ├── test_evaluation.py
│   └── test_inference.py
│
├── pyproject.toml                   # Project metadata and dependencies
└── README.md
```

---

## Getting Started & Local Execution

### Environment Setup

This repository uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible virtual environment management.

```bash
# Clone repository
git clone https://github.com/adhvaith267/credit-default-pd-model.git
cd credit-default-pd-model

# Install dependencies and sync virtual environment
uv sync

# Run comprehensive test suite (98 tests)
uv run pytest tests/ -v
```

### Local Model Training

If `./cs-training.csv` is not present locally, `train.py` will automatically download it from the designated S3 bucket:

```bash
# Train all models with 50 Optuna trials
uv run python -m financial_risk_analyst_ml.train --data-path cs-training.csv --model all --tune --tune-trials 50
```

---

## Deployment & Operational Workflows

### Prerequisites
1. AWS CLI configured (`aws configure` or SSO credentials).
2. IAM Role `FinancialRiskSageMakerExecutionRole` with SageMaker and S3 permissions.
3. Dataset uploaded to `s3://financial-risk-analyst-adhvaith-2026/datasets/gmsc/raw/cs-training.csv`.

### Option A: GitHub Actions (Automated Cloud Pipeline)
Trigger cloud training and deployment directly from GitHub UI:
1. Navigate to **Actions** $\rightarrow$ **SageMaker ML Pipeline**.
2. Click **Run workflow**.
3. Select Action (`train_and_deploy`, `train_only`, or `deploy_only`) $\rightarrow$ Click **Run workflow**.

---

### Option B: Command Line Interface (CLI)

#### 1. Submit SageMaker Managed Spot Training Job
Submits a spot training job (`ml.m5.xlarge`) to AWS SageMaker (~70% cost savings):

```bash
# Full Optuna hyperparameter tuning job
uv run python scripts/train_sagemaker.py

# Quick fast training job (no tuning, ~2-3 mins)
uv run python scripts/train_sagemaker.py --no-tune
```

#### 2. Deploy to Real-Time SageMaker Endpoint
Idempotently deploys the latest model artifact from S3 to `gmsc-pd-endpoint`:

```bash
uv run python scripts/deploy_sagemaker.py
```

#### 3. Invoke & Validate Live Endpoint
Sends a dual-borrower test payload (testing both Low Risk / APPROVED and High Risk / DECLINED scenarios) to verify endpoint availability and response formatting:

```bash
uv run python scripts/invoke_endpoint.py --pretty
```

**Sample Batch Response Output:**
```json
[
  {
    "pd": 0.0185,
    "status": "APPROVED",
    "model_version": "gmsc-xgb-v1",
    "risk_drivers": []
  },
  {
    "pd": 0.354128,
    "status": "DECLINED",
    "model_version": "gmsc-xgb-v1",
    "risk_drivers": [
      "High credit card & revolving line utilization",
      "Past-due delinquency events (30-59 days late)",
      "High debt-to-income ratio"
    ]
  }
]
```

---

## Core Engineering & Architecture Decisions

1. **Single Production Model over Complex Ensembles**: Benchmarking shows that model stacking yields negligible ROC-AUC gain over a single well-tuned LightGBM model. Serving a single model dramatically reduces endpoint latency, operational complexity, and monitoring overhead.
2. **Mandatory Isotonic Calibration**: Uncalibrated gradient-boosting scores distort financial risk estimates. Enforcing isotonic calibration reduces Brier Score error by ~65.7%, ensuring accurate monetary Expected Loss calculations.
3. **Automated Microservice Explainability**: High-risk declined applications (`PD >= 0.10`) automatically calculate and attach SHAP-derived financial risk drivers directly in the HTTPS response payload. This eliminates cross-repository package dependencies in backend decisioning services.
4. **Idempotent Infrastructure-as-Code Deployment**: The deployment script uses timestamped `EndpointConfig` resources and AWS waiters to handle zero-downtime updates, stuck creation states, and failed endpoint cleanups seamlessly in CI/CD.

