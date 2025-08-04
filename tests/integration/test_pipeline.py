import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.textSummariser.pipeline.stage_1_data_ingestion_pipeline import (
    DataIngestionTrainingPipeline,
)
from src.textSummariser.pipeline.stage_2_data_transformation_pipeline import (
    DataTransformationTrainingPipeline,
)


class TestPipelineIntegration:
    """Integration tests for pipeline stages"""

    @pytest.fixture
    def temp_artifacts_dir(self):
        """Create temporary artifacts directory"""
        temp_path = tempfile.mkdtemp()
        artifacts_path = Path(temp_path) / "artifacts"
        artifacts_path.mkdir(parents=True, exist_ok=True)
        yield artifacts_path
        shutil.rmtree(temp_path)

    @patch("src.textSummariser.components.data_ingestion.load_dataset")
    @patch("src.textSummariser.config.configuration.ConfigurationManager")
    def test_data_ingestion_pipeline(
        self, mock_config_manager, mock_load_dataset, temp_artifacts_dir
    ):
        """Test data ingestion pipeline integration"""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.save_to_disk = Mock()
        mock_load_dataset.return_value = mock_dataset

        # Mock configuration
        mock_config = Mock()
        mock_config.unzip_dir = temp_artifacts_dir / "data_ingestion"
        mock_config_manager.return_value.get_data_ingestion_config.return_value = (
            mock_config
        )

        # Run pipeline
        pipeline = DataIngestionTrainingPipeline()
        pipeline.initiate_data_ingestion()

        # Verify calls
        mock_load_dataset.assert_called_once()
        mock_dataset.save_to_disk.assert_called_once()

    @patch("src.textSummariser.components.data_transformation.load_from_disk")
    @patch("src.textSummariser.components.data_transformation.AutoTokenizer")
    @patch("src.textSummariser.config.configuration.ConfigurationManager")
    def test_data_transformation_pipeline(
        self,
        mock_config_manager,
        mock_tokenizer,
        mock_load_from_disk,
        temp_artifacts_dir,
    ):
        """Test data transformation pipeline integration"""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.map = Mock(return_value=mock_dataset)
        mock_dataset.save_to_disk = Mock()
        mock_load_from_disk.return_value = mock_dataset

        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock configuration
        mock_config = Mock()
        mock_config.data_path = temp_artifacts_dir / "data"
        mock_config.root_dir = temp_artifacts_dir / "data_transformation"
        mock_config.tokenizer_name = "google/pegasus-cnn_dailymail"
        mock_config_manager.return_value.get_data_transformation_config.return_value = (
            mock_config
        )

        # Run pipeline
        pipeline = DataTransformationTrainingPipeline()
        pipeline.initiate_data_transformation()

        # Verify calls
        mock_load_from_disk.assert_called_once_with(mock_config.data_path)
        mock_dataset.map.assert_called_once()
        mock_dataset.save_to_disk.assert_called_once()

    def test_pipeline_stage_order(self):
        """Test that pipeline stages can be called in correct order"""
        # This is a basic test to ensure stages can be imported and instantiated
        stage1 = DataIngestionTrainingPipeline()
        stage2 = DataTransformationTrainingPipeline()

        assert stage1 is not None
        assert stage2 is not None
        assert hasattr(stage1, "initiate_data_ingestion")
        assert hasattr(stage2, "initiate_data_transformation")

    @patch("src.textSummariser.utils.mlflow_utils.mlflow")
    def test_mlflow_integration_in_pipeline(self, mock_mlflow):
        """Test that MLflow integration works in pipeline context"""
        from src.textSummariser.utils.mlflow_utils import MLflowTracker

        # Mock MLflow responses
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "123"
        mock_mlflow.start_run.return_value.__enter__ = Mock()
        mock_mlflow.start_run.return_value.__exit__ = Mock()

        # Test tracker initialization
        tracker = MLflowTracker("test-pipeline-experiment")

        # Test basic operations
        with tracker.start_run("test-pipeline-run"):
            tracker.log_params({"test_param": "value"})
            tracker.log_metrics({"test_metric": 0.5})

        # Verify MLflow was called
        mock_mlflow.set_experiment.assert_called_once()
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_metrics.assert_called_once()
