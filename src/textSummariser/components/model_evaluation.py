import evaluate
import pandas as pd
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.textSummariser.entity import ModelEvaluationConfig
from src.textSummariser.logging import logger
from src.textSummariser.utils.mlflow_utils import MLflowTracker


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.mlflow_tracker = MLflowTracker(
            experiment_name="text-summarisation-evaluation"
        )

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        model,
        tokenizer,
        batch_size=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        column_text="article",
        column_summary="highlights",
    ):
        article_batches = list(
            self.generate_batch_sized_chunks(dataset[column_text], batch_size)
        )
        target_batches = list(
            self.generate_batch_sized_chunks(dataset[column_summary], batch_size)
        )

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)
        ):
            inputs = tokenizer(
                article_batch,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8,
                num_beams=8,
                max_length=128,
            )
            """ parameter for length penalty ensures that the model does not generate sequences that are too long. """

            # Finally, we decode the generated texts,
            # replace the  token, and add the decoded texts with the references to the metric.
            decoded_summaries = [
                tokenizer.decode(
                    s,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                for s in summaries
            ]

            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        #  Finally compute and return the ROUGE scores.
        score = metric.compute()
        return score

    def evaluate(self):
        # Start MLflow run for evaluation
        with self.mlflow_tracker.start_run(run_name="pegasus-samsum-evaluation"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Starting model evaluation on device: {device}")

            # Log evaluation parameters
            eval_params = {
                "device": device,
                "model_path": self.config.model_path,
                "tokenizer_path": self.config.tokenizer_path,
                "data_path": self.config.data_path,
                "batch_size": 2,
                "max_samples": 10,  # Using first 10 samples for quick evaluation
            }
            self.mlflow_tracker.log_params(eval_params)

            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_path
            ).to(device)

            # loading data
            dataset_samsum_pt = load_from_disk(self.config.data_path)

            # Log dataset info
            dataset_info = {
                "name": "samsum_evaluation",
                "test_samples": len(dataset_samsum_pt["test"]),
                "samples_evaluated": 10,
            }
            self.mlflow_tracker.log_dataset_info(dataset_info)

            rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
            rouge_metric = evaluate.load("rouge")

            logger.info("Calculating ROUGE metrics on test dataset...")
            score = self.calculate_metric_on_test_ds(
                dataset_samsum_pt["test"][0:10],
                rouge_metric,
                model_pegasus,
                tokenizer,
                batch_size=2,
                column_text="dialogue",
                column_summary="summary",
            )

            # Extract ROUGE scores
            rouge_dict = {rn: score[rn] for rn in rouge_names}

            # Log metrics to MLflow
            self.mlflow_tracker.log_metrics(rouge_dict)

            # Log individual ROUGE components if available
            for rouge_name in rouge_names:
                if isinstance(rouge_dict[rouge_name], dict):
                    # If the score is a dict with precision, recall, fmeasure
                    for metric_type, value in rouge_dict[rouge_name].items():
                        self.mlflow_tracker.log_metrics(
                            {f"{rouge_name}_{metric_type}": value}
                        )
                else:
                    # If it's a single value
                    self.mlflow_tracker.log_metrics(
                        {rouge_name: rouge_dict[rouge_name]}
                    )

            # Save results to CSV
            df = pd.DataFrame(rouge_dict, index=["pegasus"])
            df.to_csv(self.config.metric_file_name, index=False)

            # Log the results CSV as artifact
            self.mlflow_tracker.log_artifact(
                self.config.metric_file_name, "evaluation_results"
            )

            logger.info("Model evaluation completed and logged to MLflow")
            logger.info(f"ROUGE Scores: {rouge_dict}")

            return rouge_dict
