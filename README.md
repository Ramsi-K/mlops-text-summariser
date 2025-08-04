# Conversational Text Summarisation with MLOps Pipeline

> **MLOps Zoomcamp Final Project** - A complete production-ready text summarization pipeline with AWS deployment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD Pipeline](https://github.com/Ramsi-K/mlops-text-summariser/actions/workflows/ci.yml/badge.svg)](https://github.com/Ramsi-K/mlops-text-summariser/actions/workflows/ci.yml)
[![Code Quality](https://img.shields.io/badge/code%20quality-tested-green.svg)](https://github.com/Ramsi-K/mlops-text-summariser/actions/workflows/ci.yml)
[![MLOps Zoomcamp](https://img.shields.io/badge/MLOps-Zoomcamp-orange.svg)](https://github.com/DataTalksClub/mlops-zoomcamp)

## Problem Statement

This project addresses the challenge of automatically summarising conversational text using state-of-the-art transformer models. Traditional summarisation approaches struggle with:

- **Multi-speaker conversations** with context switching
- **Informal language patterns** in chat logs and dialogues
- **Maintaining coherence** across dialogue turns
- **Preserving speaker intent** and key information

**Business Value:**

- 🚀 Reduce manual effort in processing meeting notes by 80%
- 💬 Enable real-time chat summarisation for customer support
- 📄 Automate document processing for legal and healthcare sectors
- ⏱️ Process hours of conversation into concise summaries in seconds

**Dataset:** SAMSum - A collection of 16k messenger-like conversations with human-annotated summaries, specifically designed for conversational summarisation tasks.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Data Ingestion │───▶│ Data Transform   │───▶│  Model Training │───▶│ Model Evaluation │
│                 │    │                  │    │                 │    │                  │
│ • Download data │    │ • Tokenization   │    │ • Fine-tuning   │    │ • ROUGE metrics  │
│ • Extract files │    │ • Preprocessing  │    │ • Validation    │    │ • Model registry │
│ • Validation    │    │ • Data splitting │    │ • Checkpointing │    │ • Performance    │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────┐
                                              │   FastAPI Server │
                                              │                  │
                                              │ • REST endpoints │
                                              │ • Model serving  │
                                              │ • Health checks  │
                                              └──────────────────┘
```

## Features

- **4-Stage Modular Pipeline** - Data ingestion, transformation, training, evaluation
- **MLflow Integration** - Experiment tracking and model registry
- **FastAPI Server** - Production-ready inference API
- **Docker Support** - Containerized deployment
- **Cloud Deployment** - AWS infrastructure with Terraform (IaC)
- **Configuration Management** - YAML-based configs with validation
- **Comprehensive Logging** - Structured logging throughout pipeline
- **Model Monitoring** - Performance tracking and drift detection
- **Reproducible Experiments** - Seed management and environment control

## Quick Start

### Prerequisites

- Python 3.8+
- [UV package manager](https://docs.astral.sh/uv/getting-started/installation/)
- Git

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd mlops-text-summariser
   ```

2. **Set up environment with UV**

   ```bash
   # Install UV if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create virtual environment and install dependencies
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. **Initialize the project**

   ```bash
   # Install pre-commit hooks (optional but recommended)
   pre-commit install

   # Verify installation
   python -c "import transformers; print('✅ Setup complete!')"
   ```

### Running the Pipeline

#### Option 1: Quick Training Pipeline (Recommended)

```bash
# Run complete 4-stage pipeline with quick training (3 steps)
python main.py --quick-train
```

#### Option 2: Full Training Pipeline

```bash
# Run complete 4-stage pipeline (takes 2+ hours)
python main.py
```

#### Option 2: Individual Stages

```bash
# Stage 1: Data Ingestion
python -m src.textSummariser.pipeline.stage_1_data_ingestion_pipeline

# Stage 2: Data Transformation
python -m src.textSummariser.pipeline.stage_2_data_transformation_pipeline

# Stage 3: Model Training
python -m src.textSummariser.pipeline.stage_3_model_trainer_pipeline

# Stage 4: Model Evaluation
python -m src.textSummariser.pipeline.stage_4_model_evaluation
```

#### Option 3: Using Make Commands

```bash
make install    # Install dependencies
make train      # Run training pipeline
make test       # Run tests
make serve      # Start API server
make lint       # Code quality checks
```

### API Server

Start the FastAPI inference server:

```bash
# Development server
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Or using make
make serve
```

**API Endpoints:**

- `POST /predict` - Summarise text
- `POST /train` - Trigger training pipeline
- `GET /health` - Health check
- `GET /metrics` - Model performance metrics

**Example API Usage:**

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your conversation text here..."}'
```

### Docker Deployment

```bash
# Build image
docker build -t text-summariser .

# Run container (uses base model if no trained model available)
docker run -p 8000:8000 text-summariser
```

**Note:** The Docker container will use the base Pegasus model for inference if no trained model artifacts are available. To use a fine-tuned model, run the training pipeline first.

### ☁️ Cloud Deployment (AWS)

Deploy to production using Infrastructure as Code (Terraform):

```bash
# Prerequisites: AWS CLI configured, Terraform installed
# One-command deployment
./infrastructure/deploy.sh
```

**What gets deployed:**

- ECS Fargate cluster with auto-scaling
- Application Load Balancer for high availability
- ECR repository for container images
- VPC with security groups
- CloudWatch monitoring and logging

**Access your deployed application:**

- Application URL: Output from deployment script
- Health check: `{URL}/health`
- API docs: `{URL}/docs`

See [`infrastructure/README.md`](infrastructure/README.md) for detailed deployment instructions.

## Project Structure

```text
mlops-text-summariser/
├── src/textSummariser/           # Main package
│   ├── components/               # Pipeline components
│   ├── config/                   # Configuration management
│   ├── entity/                   # Data classes and entities
│   ├── pipeline/                 # Training pipelines
│   ├── utils/                    # Utility functions
│   └── logging.py               # Logging configuration
├── infrastructure/               # Infrastructure as Code
│   ├── terraform/               # Terraform configurations
│   │   ├── main.tf              # Main infrastructure
│   │   ├── variables.tf         # Input variables
│   │   └── outputs.tf           # Output values
│   ├── deploy.sh                # Automated deployment script
│   ├── destroy.sh               # Infrastructure cleanup script
│   └── README.md                # Deployment documentation
├── config/                       # Configuration files
│   └── config.yaml              # Main configuration
├── notebooks/                    # Jupyter notebooks for exploration
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── conftest.py              # Test configuration
├── artifacts/                    # Generated artifacts (gitignored)
├── logs/                        # Log files (gitignored)
├── mlruns/                      # MLflow tracking (gitignored)
├── app.py                       # FastAPI application
├── main.py                      # Training pipeline entry point
├── params.yaml                  # Model parameters
├── requirements.txt             # Dependencies
├── Dockerfile                   # Container configuration
├── Makefile                     # Automation commands
└── README.md                    # This file
```

## Configuration

The project uses YAML-based configuration management:

- **`config/config.yaml`** - Main configuration (paths, URLs, etc.)
- **`params.yaml`** - Model hyperparameters and training settings

Key configurations:

```yaml
# Model settings
model_name: 'google/pegasus-cnn_dailymail'
max_input_length: 1024
max_target_length: 128

# Training parameters
num_train_epochs: 1
learning_rate: 1e-4
per_device_train_batch_size: 16
```

## MLflow Experiment Tracking

The project integrates MLflow for experiment tracking:

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
```

**Tracked Metrics:**

- ROUGE-1, ROUGE-2, ROUGE-L scores
- Training loss and validation loss
- Model parameters and hyperparameters
- Training duration and resource usage

## Model Performance

**Current Model:** Google Pegasus CNN/DailyMail fine-tuned on SAMSum

| Metric  | Score |
| ------- | ----- |
| ROUGE-1 | 0.42  |
| ROUGE-2 | 0.21  |
| ROUGE-L | 0.33  |

## Development

### Code Quality

```bash
# Format code
black src tests
isort src tests

# Lint code
flake8 src tests

# Type checking
mypy src

# Run all quality checks
make lint
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test category
pytest tests/unit/
pytest tests/integration/

# Using make
make test
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## CI/CD Pipeline

The project includes a comprehensive CI/CD pipeline using GitHub Actions:

### Continuous Integration

**Automated on every push and PR:**

- **Multi-Python Testing**: Tests on Python 3.8, 3.9, 3.10
- **Code Quality Checks**: Black formatting, isort, flake8 linting
- **Unit & Integration Tests**: Full test suite with coverage reporting
- **Security Scanning**: Bandit security analysis
- **Docker Build Testing**: Validates containerization

### Continuous Deployment

**Automated deployment pipeline:**

- **Staging Deployment**: Automatic deployment to staging on main branch
- **Production Deployment**: Manual/release-triggered production deployment
- **Health Checks**: Post-deployment validation
- **Model Registry Updates**: Automatic model versioning

### Local CI/CD Testing

```bash
# Run CI tests locally
make ci-test

# Security scan
make security-scan

# Test Docker build
make docker-test
```

### Pipeline Status

- ✅ **Code Quality**: Automated formatting and linting
- ✅ **Testing**: Unit and integration tests with coverage
- ✅ **Security**: Bandit security scanning
- ✅ **Containerization**: Docker build and test validation
- ✅ **Deployment**: Staging and production deployment simulation

## Deployment

### Local Development

```bash
uvicorn app:app --reload
```

### Production Deployment

```bash
# Using Docker
docker build -t text-summariser .
docker run -p 8000:8000 text-summariser

# Using cloud platforms (AWS/GCP/Azure)
# See deployment/ directory for infrastructure code
```

## Monitoring

The application includes built-in monitoring:

- **Health checks** at `/health` endpoint
- **Metrics collection** for model performance
- **Logging** with structured format
- **Error tracking** and alerting

## Troubleshooting

### Common Issues

**1. CUDA/GPU Issues**

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CPU-only version if needed
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**2. Memory Issues**

- Reduce batch size in `params.yaml`
- Use gradient accumulation
- Enable mixed precision training

**3. Download Issues**

- Check internet connection
- Verify Hugging Face Hub access
- Use offline mode if needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed as part of the [DataTalks.Club MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) - an excellent free course on MLOps engineering practices.

**Special thanks to:**

- **DataTalks.Club** for providing comprehensive MLOps education and community
- [Hugging Face Transformers](https://huggingface.co/transformers/) for the model architecture
- [SAMSum Dataset](https://arxiv.org/abs/1911.12237) creators for the conversational data
- [MLflow](https://mlflow.org/) for experiment tracking and model registry
- [FastAPI](https://fastapi.tiangolo.com/) for the production-ready web framework

---
