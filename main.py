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
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x"
        )
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
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x"
        )
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
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x"
        )
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
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x"
        )
    except Exception as e:
        logger.exception(e)
        raise e


def main():
    parser = argparse.ArgumentParser(
        description="Text Summarization MLOps Pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "ingest", "transform", "train", "evaluate"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )

    args = parser.parse_args()

    logger.info(f"Starting pipeline with stage: {args.stage}")

    if args.stage in ["all", "ingest"]:
        run_data_ingestion()

    if args.stage in ["all", "transform"]:
        run_data_transformation()

    if args.stage in ["all", "train"]:
        run_model_training()

    if args.stage in ["all", "evaluate"]:
        run_model_evaluation()

    logger.info("Pipeline execution completed!")


if __name__ == "__main__":
    main()
