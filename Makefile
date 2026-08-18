.PHONY: train deploy invoke train-fast train-local test lint clean

# Default target
help:
	@echo "Available commands:"
	@echo "  make train        - Submit a full SageMaker spot training job (Optuna tuned)"
	@echo "  make train-fast   - Submit a fast SageMaker spot training job (No tuning)"
	@echo "  make deploy       - Deploy latest model artifact to SageMaker real-time endpoint"
	@echo "  make invoke       - Send test prediction request to live SageMaker endpoint"
	@echo "  make train-local  - Run training locally on cs-training.csv"
	@echo "  make test         - Run test suite with pytest"
	@echo "  make clean        - Clean Python cache files"

train:
	uv run python scripts/train_sagemaker.py

train-fast:
	uv run python scripts/train_sagemaker.py --no-tune

deploy:
	uv run python scripts/deploy_sagemaker.py

invoke:
	uv run python scripts/invoke_endpoint.py --pretty

train-local:
	uv run python -m financial_risk_analyst_ml.train --data-path cs-training.csv --model all

test:
	uv run pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
