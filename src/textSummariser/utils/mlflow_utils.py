import mlflow
import mlflow.transformers
import os
from pathlib import Path
from src.textSummariser.logging import logger


class MLflowTracker:
    """MLflow experiment tracking utilities"""

    def __init__(
        self, experiment_name="text-summarization", tracking_uri=None
    ):
        self.experiment_name = experiment_name

        # Set tracking URI (local by default)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local mlruns directory
            mlflow.set_tracking_uri("file:./mlruns")

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(
                    f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})"
                )
            else:
                experiment_id = experiment.experiment_id
                logger.info(
                    f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})"
                )

            mlflow.set_experiment(experiment_name)

        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            raise e

    def start_run(self, run_name=None, tags=None):
        """Start a new MLflow run"""
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params):
        """Log parameters to MLflow"""
        try:
            mlflow.log_params(params)
            logger.info(f"Logged {len(params)} parameters to MLflow")
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")

    def log_metrics(self, metrics, step=None):
        """Log metrics to MLflow"""
        try:
            if step is not None:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metrics(metrics)
            logger.info(f"Logged {len(metrics)} metrics to MLflow")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

    def log_model(
        self, model, tokenizer, artifact_path="model", model_name=None
    ):
        """Log model and tokenizer to MLflow"""
        try:
            # Log the transformers model
            mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                artifact_path=artifact_path,
                registered_model_name=model_name,
            )
            logger.info(f"Logged model to MLflow at path: {artifact_path}")

            if model_name:
                logger.info(f"Registered model as: {model_name}")

        except Exception as e:
            logger.error(f"Error logging model: {e}")

    def log_artifact(self, local_path, artifact_path=None):
        """Log artifact to MLflow"""
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Error logging artifact: {e}")

    def log_dataset_info(self, dataset_info):
        """Log dataset information"""
        try:
            mlflow.log_params(
                {
                    "dataset_name": dataset_info.get("name", "unknown"),
                    "dataset_size": dataset_info.get("size", 0),
                    "train_samples": dataset_info.get("train_samples", 0),
                    "val_samples": dataset_info.get("val_samples", 0),
                    "test_samples": dataset_info.get("test_samples", 0),
                }
            )
            logger.info("Logged dataset information to MLflow")
        except Exception as e:
            logger.error(f"Error logging dataset info: {e}")

    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()
        logger.info("Ended MLflow run")

    @staticmethod
    def get_best_model(
        experiment_name, metric_name="rouge_l", ascending=False
    ):
        """Get the best model from an experiment based on a metric"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.error(f"Experiment {experiment_name} not found")
                return None

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[
                    f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"
                ],
            )

            if len(runs) == 0:
                logger.warning(
                    f"No runs found in experiment {experiment_name}"
                )
                return None

            best_run = runs.iloc[0]
            logger.info(
                f"Best model run ID: {best_run.run_id} with {metric_name}: {best_run[f'metrics.{metric_name}']}"
            )

            return best_run

        except Exception as e:
            logger.error(f"Error getting best model: {e}")
            return None
