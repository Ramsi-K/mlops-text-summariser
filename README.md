# Conversational Text Summarization with MLOps Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Problem Statement

This project addresses the challenge of automatically summarizing conversational text using state-of-the-art transformer models. Traditional summarization approaches struggle with:

- **Multi-speaker conversations** with context switching
- **Informal language patterns** in chat logs and dialogues
- **Maintaining coherence** across dialogue turns
- **Preserving speaker intent** and key information

**Business Value:**

- 🚀 Reduce manual effort in processing meeting notes by 80%
- 💬 Enable real-time chat summarization for customer support
- 📄 Automate document processing for legal and healthcare sectors
- ⏱️ Process hours of conversation into concise summaries in seconds

**Dataset:** SAMSum - A collection of 16k messenger-like conversations with human-annotated summaries, specifically designed for conversational summarization tasks.

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

✅ **4-Stage Modular Pipeline** - Data ingestion, transformation, training, evaluation
✅ **MLflow Integration** - Experiment tracking and model registry
✅ **FastAPI Server** - Production-ready inference API
✅ **Docker Support** - Containerized deployment
✅ **Configuration Management** - YAML-based configs with validation
✅ **Comprehensive Logging** - Structured logging throughout pipeline
✅ **Model Monitoring** - Performance tracking and drift detection
✅ **Reproducible Experiments** - Seed management and environment control

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

#### Option 1: Full Training Pipeline

```bash
# Run complete 4-stage pipeline
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

- `POST /predict` - Summarize text
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
docker build -t text-summarizer .

# Run container
docker run -p 8000:8000 text-summarizer

# Or using make
make docker-build
make docker-run
```

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

## Deployment

### Local Development

```bash
uvicorn app:app --reload
```

### Production Deployment

```bash
# Using Docker
docker build -t text-summarizer .
docker run -p 8000:8000 text-summarizer

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

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the model architecture
- [SAMSum Dataset](https://arxiv.org/abs/1911.12237) creators
- [MLflow](https://mlflow.org/) for experiment tracking
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

---
