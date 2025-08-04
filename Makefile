.PHONY: help install test lint format train serve clean docker-build docker-run

help:  ## Show this help message
	@echo "Text Summarization MLOps Pipeline"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies with UV
	uv pip install -r requirements.txt
	@echo "✅ Dependencies installed"

install-dev:  ## Install development dependencies
	uv pip install -r requirements.txt
	uv run pre-commit install
	@echo "✅ Development environment set up"

test:  ## Run tests
	uv run pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "✅ Tests completed"

test-unit:  ## Run unit tests only
	uv run pytest tests/unit/ -v
	@echo "✅ Unit tests completed"

test-integration:  ## Run integration tests only
	uv run pytest tests/integration/ -v
	@echo "✅ Integration tests completed"

lint:  ## Run linting
	uv run flake8 src
	uv run black --check src
	uv run isort --check-only src
	@echo "✅ Linting completed"

format:  ## Format code
	uv run black src
	uv run isort src
	@echo "✅ Code formatted"

pre-commit-install:  ## Install pre-commit hooks
	uv run pre-commit install
	@echo "✅ Pre-commit hooks installed"

pre-commit-run:  ## Run pre-commit on all files
	uv run pre-commit run --all-files

# Pipeline stages
ingest:  ## Run data ingestion only
	uv run main.py --stage ingest

transform:  ## Run data transformation only
	uv run main.py --stage transform

train:  ## Run model training only
	uv run main.py --stage train

evaluate:  ## Run model evaluation only
	uv run main.py --stage evaluate

pipeline:  ## Run full pipeline
	uv run main.py --stage all

quick-train:  ## Run quick training pipeline (for development)
	uv run quick_train.py

serve:  ## Start FastAPI server
	uv run uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Docker commands
docker-build:  ## Build Docker image
	docker build -t text-summarizer .

docker-run:  ## Run Docker container
	docker run -p 8000:8000 text-summarizer

# MLflow commands
mlflow-ui:  ## Start MLflow UI
	uv run mlflow ui --host 0.0.0.0 --port 5000

mlflow-experiments:  ## List MLflow experiments
	uv run mlflow experiments list

# Utility commands
clean:  ## Clean artifacts and cache
	rm -rf artifacts/
	rm -rf logs/
	rm -rf mlruns/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "✅ Cleaned up artifacts and cache"

logs:  ## Show recent logs
	@echo "Recent pipeline logs:"
	@echo "===================="
	@if [ -d "logs" ]; then tail -n 50 logs/*.log 2>/dev/null || echo "No log files found"; else echo "No logs directory found"; fi

status:  ## Check pipeline status
	@echo "Pipeline Status Check"
	@echo "===================="
	@echo "Artifacts directory:"
	@ls -la artifacts/ 2>/dev/null || echo "No artifacts directory found"
	@echo ""
	@echo "Model files:"
	@ls -la artifacts/model_trainer/ 2>/dev/null || echo "No model files found"
	@echo ""
	@echo "Recent logs:"
	@tail -n 10 logs/*.log 2>/dev/null || echo "No recent logs found"
