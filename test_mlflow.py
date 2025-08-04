#!/usr/bin/env python3
"""
Quick test script to verify MLflow integration
"""

from src.textSummariser.utils.mlflow_utils import MLflowTracker
import time


def test_mlflow_integration():
    """Test basic MLflow functionality"""
    print("🧪 Testing MLflow integration...")

    # Initialize tracker
    tracker = MLflowTracker(experiment_name="test-experiment")

    # Start a test run
    with tracker.start_run(run_name="test-run") as run:
        print(f"✅ Started MLflow run: {run.info.run_id}")

        # Log some test parameters
        test_params = {
            "learning_rate": 0.001,
            "batch_size": 16,
            "model_name": "test-model",
        }
        tracker.log_params(test_params)
        print("✅ Logged parameters")

        # Log some test metrics
        test_metrics = {"accuracy": 0.95, "loss": 0.05, "rouge_1": 0.42}
        tracker.log_metrics(test_metrics)
        print("✅ Logged metrics")

        # Simulate some training steps
        for step in range(3):
            tracker.log_metrics({"step_loss": 0.1 - step * 0.01}, step=step)
            time.sleep(0.1)
        print("✅ Logged step metrics")

    print("✅ MLflow integration test completed successfully!")
    print("🌐 Run 'make mlflow-ui' to view results in MLflow UI")


if __name__ == "__main__":
    test_mlflow_integration()
