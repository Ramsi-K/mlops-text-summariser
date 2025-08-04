from unittest.mock import MagicMock, Mock, patch

import pytest

from src.textSummariser.utils.mlflow_utils import MLflowTracker


class TestMLflowTracker:
    """Test MLflow tracking utilities"""

    @patch("src.textSummariser.utils.mlflow_utils.mlflow")
    def test_init_new_experiment(self, mock_mlflow):
        """Test initializing tracker with new experiment"""
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "123"

        tracker = MLflowTracker("test-experiment")

        mock_mlflow.set_tracking_uri.assert_called_once_with("file:./mlruns")
        mock_mlflow.get_experiment_by_name.assert_called_once_with("test-experiment")
        mock_mlflow.create_experiment.assert_called_once_with("test-experiment")
        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")

    @patch("src.textSummariser.utils.mlflow_utils.mlflow")
    def test_init_existing_experiment(self, mock_mlflow):
        """Test initializing tracker with existing experiment"""
        mock_experiment = Mock()
        mock_experiment.experiment_id = "456"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        tracker = MLflowTracker("existing-experiment")

        mock_mlflow.get_experiment_by_name.assert_called_once_with(
            "existing-experiment"
        )
        mock_mlflow.create_experiment.assert_not_called()
        mock_mlflow.set_experiment.assert_called_once_with("existing-experiment")

    @patch("src.textSummariser.utils.mlflow_utils.mlflow")
    def test_log_params(self, mock_mlflow):
        """Test logging parameters"""
        tracker = MLflowTracker("test-experiment")

        params = {"learning_rate": 0.001, "batch_size": 16}
        tracker.log_params(params)

        mock_mlflow.log_params.assert_called_once_with(params)

    @patch("src.textSummariser.utils.mlflow_utils.mlflow")
    def test_log_metrics(self, mock_mlflow):
        """Test logging metrics"""
        tracker = MLflowTracker("test-experiment")

        metrics = {"accuracy": 0.95, "loss": 0.05}
        tracker.log_metrics(metrics)

        mock_mlflow.log_metrics.assert_called_once_with(metrics)

    @patch("src.textSummariser.utils.mlflow_utils.mlflow")
    def test_log_metrics_with_step(self, mock_mlflow):
        """Test logging metrics with step"""
        tracker = MLflowTracker("test-experiment")

        metrics = {"loss": 0.1}
        tracker.log_metrics(metrics, step=5)

        mock_mlflow.log_metric.assert_called_once_with("loss", 0.1, step=5)

    @patch("src.textSummariser.utils.mlflow_utils.mlflow")
    def test_log_model(self, mock_mlflow):
        """Test logging model"""
        tracker = MLflowTracker("test-experiment")

        mock_model = Mock()
        mock_tokenizer = Mock()

        tracker.log_model(mock_model, mock_tokenizer, "model", "test-model")

        mock_mlflow.transformers.log_model.assert_called_once_with(
            transformers_model={
                "model": mock_model,
                "tokenizer": mock_tokenizer,
            },
            artifact_path="model",
            registered_model_name="test-model",
        )

    @patch("src.textSummariser.utils.mlflow_utils.mlflow")
    def test_log_dataset_info(self, mock_mlflow):
        """Test logging dataset information"""
        tracker = MLflowTracker("test-experiment")

        dataset_info = {
            "name": "samsum",
            "size": 1000,
            "train_samples": 800,
            "val_samples": 100,
            "test_samples": 100,
        }

        tracker.log_dataset_info(dataset_info)

        expected_params = {
            "dataset_name": "samsum",
            "dataset_size": 1000,
            "train_samples": 800,
            "val_samples": 100,
            "test_samples": 100,
        }
        mock_mlflow.log_params.assert_called_once_with(expected_params)

    @patch("src.textSummariser.utils.mlflow_utils.mlflow")
    def test_get_best_model(self, mock_mlflow):
        """Test getting best model from experiment"""
        # Mock experiment
        mock_experiment = Mock()
        mock_experiment.experiment_id = "123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        # Mock runs dataframe
        mock_runs = Mock()
        mock_runs.__len__ = Mock(return_value=1)  # Add len() support
        mock_runs.iloc = [Mock()]
        mock_runs.iloc[0].run_id = "best_run_id"
        mock_runs.iloc[0].__getitem__ = lambda self, key: (
            0.85 if key == "metrics.rouge_l" else None
        )
        mock_mlflow.search_runs.return_value = mock_runs

        best_run = MLflowTracker.get_best_model("test-experiment", "rouge_l")

        assert best_run is not None
        mock_mlflow.get_experiment_by_name.assert_called_once_with("test-experiment")
        mock_mlflow.search_runs.assert_called_once()
