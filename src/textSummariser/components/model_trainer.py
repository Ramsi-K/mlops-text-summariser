from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
import torch
from datasets import load_from_disk
import os
from src.textSummariser.entity import ModelTrainerConfig
from src.textSummariser.utils.mlflow_utils import MLflowTracker
from src.textSummariser.logging import logger


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.mlflow_tracker = MLflowTracker(
            experiment_name="text-summarization-training"
        )

    def train(self):
        # Start MLflow run
        with self.mlflow_tracker.start_run(
            run_name="pegasus-samsum-training"
        ) as run:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            # Log system info
            self.mlflow_tracker.log_params(
                {
                    "device": device,
                    "model_checkpoint": self.config.model_ckpt,
                    "data_path": self.config.data_path,
                }
            )

            tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_ckpt
            ).to(device)
            seq2seq_data_collator = DataCollatorForSeq2Seq(
                tokenizer, model=model_pegasus
            )

            # loading the data
            dataset_samsum_pt = load_from_disk(self.config.data_path)

            # Log dataset info
            dataset_info = {
                "name": "samsum",
                "train_samples": len(dataset_samsum_pt["train"]),
                "val_samples": len(dataset_samsum_pt["validation"]),
                "test_samples": len(dataset_samsum_pt["test"]),
            }
            self.mlflow_tracker.log_dataset_info(dataset_info)

            # Log training parameters
            training_params = {
                "num_train_epochs": self.config.num_train_epochs,
                "warmup_steps": self.config.warmup_steps,
                "per_device_train_batch_size": self.config.per_device_train_batch_size,
                "weight_decay": self.config.weight_decay,
                "logging_steps": self.config.logging_steps,
                "evaluation_strategy": self.config.evaluation_strategy,
                "eval_steps": self.config.eval_steps,
                "save_steps": self.config.save_steps,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            }
            self.mlflow_tracker.log_params(training_params)

            trainer_args = TrainingArguments(
                output_dir=self.config.root_dir,
                num_train_epochs=self.config.num_train_epochs,
                warmup_steps=self.config.warmup_steps,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_train_batch_size,
                weight_decay=self.config.weight_decay,
                logging_steps=self.config.logging_steps,
                eval_strategy=self.config.evaluation_strategy,
                eval_steps=self.config.eval_steps,
                save_steps=self.config.save_steps,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                logging_dir=f"{self.config.root_dir}/logs",  # For tensorboard logs
                report_to=["mlflow"],  # Enable MLflow logging
                run_name="pegasus-samsum-training",
            )

            trainer = Trainer(
                model=model_pegasus,
                args=trainer_args,
                tokenizer=tokenizer,
                data_collator=seq2seq_data_collator,
                train_dataset=dataset_samsum_pt[
                    "train"
                ],  # Use train set, not test
                eval_dataset=dataset_samsum_pt["validation"],
            )

            # Train the model
            logger.info("Starting model training...")
            train_result = trainer.train()

            # Log training metrics
            if hasattr(train_result, "metrics"):
                self.mlflow_tracker.log_metrics(train_result.metrics)

            # Save model locally
            model_path = os.path.join(
                self.config.root_dir, "pegasus-samsum-model"
            )
            tokenizer_path = os.path.join(self.config.root_dir, "tokenizer")

            model_pegasus.save_pretrained(model_path)
            tokenizer.save_pretrained(tokenizer_path)

            # Log model to MLflow
            self.mlflow_tracker.log_model(
                model=model_pegasus,
                tokenizer=tokenizer,
                artifact_path="model",
                model_name="pegasus-samsum-summarizer",
            )

            # Log model artifacts
            self.mlflow_tracker.log_artifact(model_path, "model_files")
            self.mlflow_tracker.log_artifact(tokenizer_path, "tokenizer_files")

            logger.info("Model training completed and logged to MLflow")
