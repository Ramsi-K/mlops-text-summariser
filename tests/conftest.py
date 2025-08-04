import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
from src.textSummariser.entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_data_ingestion_config(temp_dir):
    """Sample DataIngestionConfig for testing"""
    return DataIngestionConfig(
        root_dir=temp_dir / "data_ingestion",
        source_URL="https://example.com/data.zip",
        local_data_file=temp_dir / "data.zip",
        unzip_dir=temp_dir / "extracted",
    )


@pytest.fixture
def sample_data_transformation_config(temp_dir):
    """Sample DataTransformationConfig for testing"""
    return DataTransformationConfig(
        root_dir=temp_dir / "data_transformation",
        data_path=temp_dir / "data",
        tokenizer_name="google/pegasus-cnn_dailymail",
    )


@pytest.fixture
def sample_model_trainer_config(temp_dir):
    """Sample ModelTrainerConfig for testing"""
    return ModelTrainerConfig(
        root_dir=temp_dir / "model_trainer",
        data_path=temp_dir / "data",
        model_ckpt="google/pegasus-cnn_dailymail",
        num_train_epochs=1,
        warmup_steps=100,
        per_device_train_batch_size=1,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=1000,
        gradient_accumulation_steps=4,
    )


@pytest.fixture
def sample_model_evaluation_config(temp_dir):
    """Sample ModelEvaluationConfig for testing"""
    return ModelEvaluationConfig(
        root_dir=temp_dir / "model_evaluation",
        data_path=temp_dir / "data",
        model_path=temp_dir / "model",
        tokenizer_path=temp_dir / "tokenizer",
        metric_file_name=temp_dir / "metrics.csv",
    )


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing"""
    return {
        "train": [
            {"dialogue": "Hello, how are you?", "summary": "Greeting"},
            {
                "dialogue": "What's the weather like?",
                "summary": "Weather inquiry",
            },
        ],
        "validation": [{"dialogue": "See you later!", "summary": "Farewell"}],
        "test": [
            {"dialogue": "Thanks for your help!", "summary": "Gratitude"}
        ],
    }
