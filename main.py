import argparse

from src.textSummariser.logging import logger
from src.textSummariser.pipeline.stage_1_data_ingestion_pipeline import (
    DataIngestionTrainingPipeline,
)
from src.textSummariser.pipeline.stage_2_data_transformation_pipeline import (
    DataTransformationTrainingPipeline,
)
from src.textSummariser.pipeline.stage_3_model_trainer_pipeline import (
    ModelTrainerTrainingPipeline,
)
from src.textSummariser.pipeline.stage_4_model_evaluation import (
    ModelEvaluationTrainingPipeline,
)


def run_data_ingestion():
    """Run data ingestion stage"""
    STAGE_NAME = "Data Ingestion stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion_pipeline = DataIngestionTrainingPipeline()
        data_ingestion_pipeline.initiate_data_ingestion()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


def run_data_transformation():
    """Run data transformation stage"""
    STAGE_NAME = "Data Transformation stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_transformation_pipeline = DataTransformationTrainingPipeline()
        data_transformation_pipeline.initiate_data_transformation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


def run_model_training():
    """Run model training stage"""
    STAGE_NAME = "Model Trainer stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_trainer_pipeline = ModelTrainerTrainingPipeline()
        model_trainer_pipeline.initiate_model_trainer()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


def run_quick_training():
    """Run quick training stage - actual training but with minimal steps"""
    STAGE_NAME = "Quick Model Training stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        logger.info("Running quick training with minimal steps for testing...")

        # Set environment variable to enable quick training mode
        import os

        os.environ["QUICK_TRAIN"] = "true"

        model_trainer_pipeline = ModelTrainerTrainingPipeline()
        model_trainer_pipeline.initiate_model_trainer()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


def run_mock_training():
    """Run mock training stage - creates dummy model files without actual training"""
    from pathlib import Path

    STAGE_NAME = "Mock Model Training stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        # Create mock model directory structure
        model_dir = Path("artifacts/model_trainer")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create mock model subdirectories
        mock_model_dir = model_dir / "pegasus-samsum-model"
        mock_tokenizer_dir = model_dir / "tokenizer"

        mock_model_dir.mkdir(exist_ok=True)
        mock_tokenizer_dir.mkdir(exist_ok=True)

        # Create dummy model files
        (mock_model_dir / "config.json").write_text(
            '{"model_type": "pegasus", "mock": true}'
        )
        (mock_model_dir / "pytorch_model.bin").write_text("mock model weights")

        # Create dummy tokenizer files
        (mock_tokenizer_dir / "tokenizer_config.json").write_text(
            '{"tokenizer_type": "pegasus", "mock": true}'
        )
        (mock_tokenizer_dir / "vocab.txt").write_text("mock vocab")

        logger.info("Created mock model and tokenizer files")
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e


def run_model_evaluation():
    """Run model evaluation stage"""
    STAGE_NAME = "Model Evaluation stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_evaluation = ModelEvaluationTrainingPipeline()
        model_evaluation.initiate_model_evaluation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


def main():
    parser = argparse.ArgumentParser(description="Text Summarisation MLOps Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "ingest", "transform", "train", "evaluate"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--mock-train",
        action="store_true",
        help="Skip actual training and create mock model files",
    )
    parser.add_argument(
        "--quick-train",
        action="store_true",
        help="Run actual training but with minimal steps for testing",
    )

    args = parser.parse_args()

    # If running training stage, ask user if they want to do actual training
    if args.stage in ["all", "train"] and not args.mock_train and not args.quick_train:
        response = (
            input(
                "Do you want to run actual model training? This may take a long time. (y/N): "
            )
            .lower()
            .strip()
        )
        if response not in ["y", "yes"]:
            print("Using mock training mode...")
            args.mock_train = True

    logger.info(f"Starting pipeline with stage: {args.stage}")
    if args.mock_train:
        logger.info("Mock training mode enabled - will skip actual training")
    elif args.quick_train:
        logger.info("Quick training mode enabled - minimal training steps for testing")

    if args.stage in ["all", "ingest"]:
        run_data_ingestion()

    if args.stage in ["all", "transform"]:
        run_data_transformation()

    if args.stage in ["all", "train"]:
        if args.mock_train:
            run_mock_training()
        elif args.quick_train:
            run_quick_training()
        else:
            run_model_training()

    if args.stage in ["all", "evaluate"]:
        if args.mock_train:
            logger.info("Skipping model evaluation in mock training mode")
        else:
            run_model_evaluation()

    logger.info("Pipeline execution completed!")


if __name__ == "__main__":
    main()
